import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *

def Inception_A(layer_in, c7):
    branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
    branch1x1 = BatchNormalization()(branch1x1_1)
    branch1x1 = ReLU()(branch1x1)

    branch5x5_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
    branch5x5 = BatchNormalization()(branch5x5_1)
    branch5x5 = ReLU()(branch5x5)
    branch5x5 = Conv1D(c7, kernel_size=5, padding='same', use_bias=False)(branch5x5)
    branch5x5 = BatchNormalization()(branch5x5)
    branch5x5 = ReLU()(branch5x5)  

    branch3x3_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
    branch3x3 = BatchNormalization()(branch3x3_1)
    branch3x3 = ReLU()(branch3x3)
    branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
    branch3x3 = BatchNormalization()(branch3x3)
    branch3x3 = ReLU()(branch3x3)
    branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
    branch3x3 = BatchNormalization()(branch3x3)
    branch3x3 = ReLU()(branch3x3) 

    branch_pool = AveragePooling1D(pool_size=(3), strides=1, padding='same')(layer_in)
    branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = ReLU()(branch_pool)
    outputs = Concatenate(axis=-1)([branch1x1, branch5x5, branch3x3, branch_pool])
    return outputs


def Inception_B(layer_in, c7):
    branch3x3 = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(layer_in)
    branch3x3 = BatchNormalization()(branch3x3)
    branch3x3 = ReLU()(branch3x3)  

    branch3x3dbl = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
    branch3x3dbl = BatchNormalization()(branch3x3dbl)
    branch3x3dbl = ReLU()(branch3x3dbl)  
    branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", use_bias=False)(branch3x3dbl)  
    branch3x3dbl = BatchNormalization()(branch3x3dbl)
    branch3x3dbl = ReLU()(branch3x3dbl)  
    branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(branch3x3dbl)    
    branch3x3dbl = BatchNormalization()(branch3x3dbl)
    branch3x3dbl = ReLU()(branch3x3dbl)   

    branch_pool = MaxPooling1D(pool_size=3, strides=2, padding="same")(layer_in)

    outputs = Concatenate(axis=-1)([branch3x3, branch3x3dbl, branch_pool])
    return outputs


def Inception_C(layer_in, c7):
    branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
    branch1x1 = BatchNormalization()(branch1x1_1)
    branch1x1 = ReLU()(branch1x1)   

    branch7x7_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
    branch7x7 = BatchNormalization()(branch7x7_1)
    branch7x7 = ReLU()(branch7x7)   
    branch7x7 = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7)
    branch7x7 = BatchNormalization()(branch7x7)
    branch7x7 = ReLU()(branch7x7)  
    branch7x7 = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7)  
    branch7x7 = BatchNormalization()(branch7x7)
    branch7x7 = ReLU()(branch7x7)   

    branch7x7dbl_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)  
    branch7x7dbl = BatchNormalization()(branch7x7dbl_1)
    branch7x7dbl = ReLU()(branch7x7dbl)  
    branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
    branch7x7dbl = BatchNormalization()(branch7x7dbl)
    branch7x7dbl = ReLU()(branch7x7dbl) 
    branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
    branch7x7dbl = BatchNormalization()(branch7x7dbl)
    branch7x7dbl = ReLU()(branch7x7dbl)  
    branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
    branch7x7dbl = BatchNormalization()(branch7x7dbl)
    branch7x7dbl = ReLU()(branch7x7dbl)  
    branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
    branch7x7dbl = BatchNormalization()(branch7x7dbl)
    branch7x7dbl = ReLU()(branch7x7dbl)  

    branch_pool = AveragePooling1D(pool_size=3, strides=1, padding='same')(layer_in)
    branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = ReLU()(branch_pool)  

    outputs = Concatenate(axis=-1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])
    return outputs

class Predictor_CNN_LSTM:
    def __init__(self,bilstm=False,model_path=None):
        self.bilstm=bilstm
        self.model_path = model_path
    
    def build_model(self,input_shape,output_shape):
        model_input = Input(shape=input_shape)

        x = Inception_A(model_input, 32)
        #x = Inception_A(x, 32)
        x = Inception_B(x, 32)
        #x = Inception_B(x, 32)
        x = Inception_C(x, 32)
        #x = Inception_C(x, 32)    
        
        if self.bilstm:
            x = Bidirectional(LSTM(50, return_sequences=True))(x)
            #x = Bidirectional(LSTM(64, return_sequences=True))(x)
        else:
            x = LSTM(50, return_sequences=True)(x)
            #x = LSTM(64, return_sequences=True)(x)
            
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        out = Dense(output_shape, activation="sigmoid")(conc)      

        self.model = Model(inputs=model_input, outputs=out)
        self.model.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape'])
    
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