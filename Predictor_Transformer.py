import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *

class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
        time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config
    
class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, 
                           input_shape=input_shape, 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v, 
                           input_shape=input_shape, 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out    

#############################################################################

class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  

        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        self.linear = Dense(input_shape[0][-1], 
                            input_shape=input_shape, 
                            kernel_initializer='glorot_uniform', 
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer 

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config

class Predictor_Transformer:
    def __init__(self,d_k=256,d_v=256,n_heads=12,ff_dim=256,model_path=None):
        self.d_k=d_k
        self.d_v=d_v
        self.n_heads=n_heads
        self.ff_dim=ff_dim
        self.model_path = model_path
    
    def build_model(self,input_shape,output_shape):
        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(input_shape[0])
        attn_layer1 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim)
        #attn_layer2 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim)
        #attn_layer3 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim)

        '''Construct model'''
        model_input = Input(shape=input_shape)
        x = time_embedding(model_input)
        x = Concatenate(axis=-1)([model_input, x])
        x = attn_layer1((x, x, x))
        #x = attn_layer2((x, x, x))
        #x = attn_layer3((x, x, x))
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(output_shape, activation='linear')(x)

        self.model = Model(inputs=model_input, outputs=out)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    
    def load_modell(self):
        self.model = load_model(self.model_path,compile=True)
    
    def train_model(self,x_train, y_train, x_test, y_test, batch_size, epochs):
        callbacks = [EarlyStopping(patience=3, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.01, verbose=1)]#,ModelCheckpoint(filepath='model'+str(i)+'.h5', save_best_only=True)]
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks = callbacks, verbose=0, validation_split=0.1)#, shuffle=True)
        '''evaluation = self.model.evaluate(x_test, y_test)
        return evaluation'''
    
    def predict(self,x):
        return self.model.predict(x)
    
    def evaluate(self,y_normaliser, x_train, x_test, real_y_train, real_y_test):
        def get_mape(y_true, y_pred): 
            """
            Compute mean absolute percentage error (MAPE)
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def get_wape(y_true, y_pred): 
            """
            Compute weighted absolute percentage error (WAPE)
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

        def get_mae(a, b):
            """
            Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
            Returns a vector of len = len(a) = len(b)
            """
            return np.mean(abs(np.array(a)-np.array(b)))

        def get_rmse(a, b):
            """
            Comp RMSE. a and b can be lists.
            Returns a scalar.
            """
            return np.sqrt(np.mean((np.array(a)-np.array(b))**2))
        
        predicted_train = self.predict(x_train)
        predicted_train = y_normaliser.inverse_transform(predicted_train)
        print('......Train RMSE...... , ',get_rmse(real_y_train , predicted_train))
        print('......Train MAPE...... , ',get_mape(real_y_train , predicted_train))
        print('......Train MAE...... , ',get_mae(real_y_train , predicted_train))
        print('......Train WAE...... , ',get_wape(real_y_train , predicted_train))

        '''predicted_test = self.predict(x_test)
        predicted_test = y_normaliser.inverse_transform(predicted_test)
        print('......Test RMSE...... , ',get_rmse(real_y_test , predicted_test))
        print('......Test MAPE...... , ',get_mape(real_y_test , predicted_test))
        print('......Test MAE...... , ',get_mae(real_y_test , predicted_test))
        print('......Test WAE...... , ',get_wape(real_y_test , predicted_test))

        x = np.append(x_train,x_test,axis=0)
        y = np.append(real_y_train,real_y_test,axis=0)
        predicted = self.predict(x)
        predicted = y_normaliser.inverse_transform(predicted)
        print('......Full RMSE...... , ',get_rmse(y , predicted))
        print('......Full MAPE...... , ',get_mape(y , predicted))
        print('......Full MAE...... , ',get_mae(y , predicted))
        print('......Full WAE...... , ',get_wape(y , predicted))'''

        return predicted_train,None#,predicted_test