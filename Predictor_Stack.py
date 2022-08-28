import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *

class Predictor_STACK:
    def __init__(self,predictors=None,model_path=None):
        self.predictors=predictors
        self.model_path = model_path
    
    def build_model(self,output_shape):
        
        outputs = []
        inputs = []
        for predictor in self.predictors:
            outputs.append(predictor.predictor.model.output)
            inputs.append(predictor.predictor.model.input)
        output = Concatenate(axis=-1)(outputs)
        output = Dense(64,activation='relu')(output)
        output = Dense(output_shape,activation='linear')(output)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    
    def load_modell(self):
        self.model = load_model(self.model_path,compile=True)
    
    def train_model(self,x_trains, y_train, x_tests, y_test, batch_size, epochs):
        callbacks = [EarlyStopping(patience=3, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.01, verbose=1)]#,ModelCheckpoint(filepath='model'+str(i)+'.h5', save_best_only=True)]
        self.model.fit(x_trains, y_train, batch_size=batch_size, epochs=epochs, callbacks = callbacks, verbose=0, validation_split=0.1)#, shuffle=True)
        '''evaluation = self.model.evaluate(x_tests, y_test)
        return evaluation'''
    
    def predict(self,x):
        return self.model.predict(x)
    
    def evaluate(self,y_normaliser, x_trains, x_tests, real_y_train, real_y_test):
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
        
        predicted_train = self.predict(x_trains)
        predicted_train = y_normaliser.inverse_transform(predicted_train)
        print('......Train RMSE...... , ',get_rmse(real_y_train , predicted_train))
        print('......Train MAPE...... , ',get_mape(real_y_train , predicted_train))
        print('......Train MAE...... , ',get_mae(real_y_train , predicted_train))
        print('......Train WAE...... , ',get_wape(real_y_train , predicted_train))

        '''predicted_test = self.predict(x_tests)
        predicted_test = y_normaliser.inverse_transform(predicted_test)
        print('......Test RMSE...... , ',get_rmse(real_y_test , predicted_test))
        print('......Test MAPE...... , ',get_mape(real_y_test , predicted_test))
        print('......Test MAE...... , ',get_mae(real_y_test , predicted_test))
        print('......Test WAE...... , ',get_wape(real_y_test , predicted_test))
        
        x = []
        for i in range(len(x_trains)):
            x.append(np.append(x_trains[i],x_tests[i],axis=0))
        y = np.append(real_y_train,real_y_test,axis=0)
        predicted = self.predict(x)
        predicted = y_normaliser.inverse_transform(predicted)
        print('......Full RMSE...... , ',get_rmse(y , predicted))
        print('......Full MAPE...... , ',get_mape(y , predicted))
        print('......Full MAE...... , ',get_mae(y , predicted))
        print('......Full WAE...... , ',get_wape(y , predicted))'''

        return predicted_train,None#,predicted_test