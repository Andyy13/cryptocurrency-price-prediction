import yfinance as yf
from datetime import date,timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from src.utilities.technical_indicators import *
from src.models.Predictor_LSTM import *
from src.models.Predictor_CNN_LSTM import *
from src.models.Predictor_Transformer import *
from src.models.Predictor_Stack import *
from sklearn import preprocessing

class Crypto_Predictor:
    def __init__(self,coin,model_name,start_date = date(2005,1,1),end_date = date.today(),add_technical=True,history_points=50,train_test_split=0.9,forecast_days=30,predictors=None,model_path=None):
        self.coin = coin
        self.model_name = model_name
        if self.model_name=='LSTM':
            self.predictor = Predictor_LSTM(model_path=model_path)
        if self.model_name=='BILSTM':
            self.predictor = Predictor_LSTM(True,model_path)
        if self.model_name=='CNN+LSTM':
            self.predictor = Predictor_CNN_LSTM(model_path=model_path)
        if self.model_name=='CNN+BILSTM':
            self.predictor = Predictor_CNN_LSTM(True,model_path)
        if self.model_name=='TRANSFORMER':
            self.predictor = Predictor_Transformer(model_path=model_path)
        if self.model_name=='STACK':
            self.predictors = predictors
            self.predictor = Predictor_STACK(predictors,model_path=model_path)
        self.start_date = start_date
        self.end_date = end_date
        self.add_technical = add_technical
        self.history_points = history_points
        self.split = train_test_split
        self.forecast_days = forecast_days
        self.model_path = model_path
        
    def download_dataset(self,start,end):
        data = yf.download(self.coin, start, end)
        data.index = pd.to_datetime(data.index)
        #data = data[['Close']]
        data = data.drop(['Adj Close','Volume'],axis = 1)
        data = data.dropna(axis=1)
        data = data.interpolate()
        if self.add_technical:
            data_ti = get_sma(data.copy())
            data_ti = get_ema(data_ti)
            data_ti = get_macd(data_ti)
            data_ti = get_stochastic_osc(data_ti)
            data_ti = get_rsi(data_ti)
            data_ti = get_atr(data_ti)
            data_ti = get_adx(data_ti)
            data_ti = get_bollinger_bands(data_ti)
            #data_ti.drop(['Open', 'High', 'Low', 'Turnover', 'sma_20', 'high_lag_1', 'low_lag_1', ], axis=1, inplace=True)
            tech_indicators = [ 'Close',
                                'sma_above20',
                                'sma_above50',
                                'sma_above100',
                                'sma_above200',
                                'sma_10above20',
                                'sma_10above50',
                                'sma_10above100',
                                'sma_10above200',
                                'sma_cut20',
                                'sma_cut50',
                                'sma_cut100',
                                'sma_cut200',
                                'sma_10cut20',
                                'sma_10cut50',
                                'sma_10cut100',
                                'sma_10cut200',
                                'sma_cut20down',
                                'sma_cut50down',
                                'sma_cut100down',
                                'sma_cut200down',
                                'sma_10cut20down',
                                'sma_10cut50down',
                                'sma_10cut100down',
                                'sma_10cut200down',
                                'ema_above20',
                                'ema_above50',
                                'ema_above100',
                                'ema_above200',
                                'ema_10above20',
                                'ema_10above50',
                                'ema_10above100',
                                'ema_10above200',
                                'ema_cut20',
                                'ema_cut50',
                                'ema_cut100',
                                'ema_cut200',
                                'ema_10cut20',
                                'ema_10cut50',
                                'ema_10cut100',
                                'ema_10cut200',
                                'ema_cut20down',
                                'ema_cut50down',
                                'ema_cut100down',
                                'ema_cut200down',
                                'ema_10cut20down',
                                'ema_10cut50down',
                                'ema_10cut100down',
                                'ema_10cut200down',
                                'macd',
                                'macd_crossover',
                                'macd_crossoverdown',
                                'stochastic_fast',
                                'stochastic_slow',
                                'stochastic_fastcutslow',
                                'stochastic_fastcutslowdown',
                                'stochastic_overs',
                                'stochastic_overb',
                                'rsi',
                                'rsi_ob',
                                'rsi_os',
                                'bollinger_upp_dist',
                                'bollinger_low_dist',
                                'bollinger_ob',
                                'bollinger_os'
                            ]
            data = data_ti[tech_indicators]
            data = data.interpolate()
            data = data.fillna(method='bfill')
        return data

    def split_dataset(self,data):
        n = int(data.shape[0] * self.split)
        train = data.iloc[:n]
        test = data.iloc[n-self.history_points:]    
        return train,test

    def preprocess_dataset(self,data,process,data_normaliser=None):

        if process.lower() == 'train':

            data_normaliser = preprocessing.MinMaxScaler()
            data_normalised = data_normaliser.fit_transform(data)

            x_train = np.array([data_normalised[i : i +  self.history_points].copy() for i in range(len(data_normalised) - self.history_points -  self.forecast_days)])
            y_train = np.squeeze(np.array([data_normalised[:,0][i + self.history_points:i + self.history_points + self.forecast_days].copy() for i in range(len(data_normalised) - self.history_points - self.forecast_days)]))
            real_y_train = np.squeeze(np.array([data[['Close']].to_numpy()[i + self.history_points:i + self.history_points + self.forecast_days] for i in range(len(data) - self.history_points - self.forecast_days)]))
            if len(y_train.shape) == 1:
                y_train = np.expand_dims(y_train,axis=-1)
                real_y_train = np.expand_dims(real_y_train,axis=-1)
            y_normaliser = preprocessing.MinMaxScaler()
            y_normaliser.fit(real_y_train)

            return data_normaliser,x_train,y_train,real_y_train,y_normaliser,x_train[0].shape,y_train.shape[1]

        elif process.lower() == 'test':

            data_normalised = data_normaliser.transform(data)
            x_test = np.array([data_normalised[i : i + self.history_points].copy() for i in range(len(data_normalised) - self.history_points - self.forecast_days)])
            y_test = np.squeeze(np.array([data_normalised[:,0][i + self.history_points:i + self.history_points + self.forecast_days].copy() for i in range(len(data_normalised) - self.history_points - self.forecast_days)]))
            real_y_test = np.squeeze(np.array([data[['Close']].to_numpy()[i + self.history_points:i + self.history_points + self.forecast_days] for i in range(len(data) - self.history_points - self.forecast_days)]))
            if len(y_test.shape) == 1:
                y_test = np.expand_dims(y_test,axis=-1)
                real_y_test = np.expand_dims(real_y_test,axis=-1)
            return x_test,y_test,real_y_test

        elif process.lower() == 'forecast':

            data_normalised = data_normaliser.transform(data)
            x_forecast = np.array([data_normalised[i : i + self.history_points].copy() for i in range(len(data_normalised) - self.history_points - self.forecast_days)])
            return x_forecast

    def forecast_price(self,start,plot = True):
        
        if self.model_name == 'STACK':
            x = []
            for predictor in self.predictors:
                lookback_date = start-timedelta(days=predictor.history_points*10)
                data = self.download_dataset(lookback_date,start)
                data_normalised = self.data_normaliser.transform(data)
                x_normalised = np.array([data_normalised[-predictor.history_points:]])
                x.append(x_normalised)
            output = self.predictor.predict(x)
            output = self.y_normaliser.inverse_transform(output)
            output = output.T
            prediction_dates = pd.date_range(start, periods=self.forecast_days).tolist()
            forecast = pd.DataFrame(output,index = prediction_dates,columns=['Pred_Close'])
            if plot:
                self.plot('forecast',forecast=forecast)
            return forecast
        else:
            lookback_date = start-timedelta(days=self.history_points*10)
            data = self.download_dataset(lookback_date,start)
            data_normalised = self.data_normaliser.transform(data)
            x_normalised = np.array([data_normalised[-self.history_points:]])
            output = self.predictor.predict(x_normalised)
            output = self.y_normaliser.inverse_transform(output)
            output = output.T
            prediction_dates = pd.date_range(start, periods=self.forecast_days).tolist()
            forecast = pd.DataFrame(output,index = prediction_dates,columns=['Pred_Close'])
            if plot:
                self.plot('forecast',forecast=forecast)
            return forecast

    def plot(self,process,forecast=None):

        if process=='train':
            plt.gcf().set_size_inches(22, 15, forward=True)
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(self.train.index,self.train.iloc[:,0], label='Train data')
            plt.plot(self.train.index[self.highest_hp:-self.forecast_days],self.predicted_train[:,0], label='Train predictions')
            plt.legend()
            plt.show()
        elif process=='test':
            plt.gcf().set_size_inches(22, 15, forward=True)
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(self.test.index,self.test.iloc[:,0], label='Test data')
            plt.plot(self.test.index[self.highest_hp:-self.forecast_days],self.predicted_test[:,0], label='Test predictions')
            plt.legend()
            plt.show()
        elif process=='full':
            plt.gcf().set_size_inches(22, 15, forward=True)
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(self.train.index,self.train.iloc[:,0], 'blue', label='Train data')
            plt.plot(self.test.index,self.test.iloc[:,0], 'blue', label='Test data')
            plt.plot(self.train.index[self.highest_hp:-self.forecast_days],self.predicted_train[:,0], 'orange', label='Train predictions')
            plt.plot(self.test.index[self.highest_hp:-self.forecast_days],self.predicted_test[:,0], 'green', label='Test predictions')
            plt.legend()
            plt.show()
        elif process=='forecast':
            plt.gcf().set_size_inches(22, 15, forward=True)
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(forecast.index,forecast['Pred_Close'], label='Forecast_'+str(self.forecast_days)+'_days')
            plt.show()
        elif process=='full+forecast':
            plt.gcf().set_size_inches(22, 15, forward=True)
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(self.train.index,self.train.iloc[:,0], 'blue', label='Train data')
            #plt.plot(self.test.index,self.test.iloc[:,0], 'blue', label='Test data')
            plt.plot(self.train.index[self.highest_hp:-self.forecast_days],self.predicted_train[:,0], 'orange', label='Train predictions')
            #plt.plot(self.test.index[self.highest_hp:-self.forecast_days],self.predicted_test[:,0], 'green', label='Test predictions')
            plt.plot(forecast.index,forecast['Pred_Close'], 'red', label='Forecast_'+str(self.forecast_days)+'_days')
            plt.legend()
            plt.show()
            
    def run(self,batch_size=32,epochs=50,start_pred_date=date.today(),plot=True):
        if self.model_name == 'STACK':
            for predictor in self.predictors:
                if predictor.forecast_days!=self.forecast_days:
                    return None
                
            print('-----------Creating Datasets and Different Models-----------')
            data = self.download_dataset(self.start_date,self.end_date)
            self.train,self.test = self.split_dataset(data)
            
            variable_dict = {'x_train':None, 'input_shape':None, 'output_shape':None,'x_test':None}
            predictor_dict = {'predictor'+str(i):variable_dict.copy() for i in range(len(self.predictors))}
            x_trains=[]
            x_tests=[]
            highest = 0
            self.highest_hp = 0
            for i,predictor in enumerate(self.predictors):
                data_normaliser,predictor_dict['predictor'+str(i)]['x_train'],y_train,real_y_train,y_normaliser,predictor_dict['predictor'+str(i)]['input_shape'],predictor_dict['predictor'+str(i)]['output_shape'] = predictor.preprocess_dataset(self.train,'train')
                predictor_dict['predictor'+str(i)]['x_test'],y_test,real_y_test = predictor.preprocess_dataset(self.test,'test',data_normaliser)
                if predictor.history_points+predictor.forecast_days>highest:
                    highest = predictor.history_points+predictor.forecast_days
                    self.highest_hp = predictor.history_points
                predictor.predictor.build_model(predictor_dict['predictor'+str(i)]['input_shape'],predictor_dict['predictor'+str(i)]['output_shape'])
                for layer in predictor.predictor.model.layers:
                    layer._name = layer._name + str(i)
            for i,predictor in enumerate(self.predictors):
                predictor_dict['predictor'+str(i)]['x_train'] = predictor_dict['predictor'+str(i)]['x_train'][-len(self.train)+highest:]
                y_train = y_train[-len(self.train)+highest:]
                real_y_train = real_y_train[-len(self.train)+highest:]
                predictor_dict['predictor'+str(i)]['x_test'] = predictor_dict['predictor'+str(i)]['x_test'][-len(self.test)+highest:]
                y_test = y_test[-len(self.test)+highest:]
                real_y_test = real_y_test[-len(self.test)+highest:]
                x_trains.append(predictor_dict['predictor'+str(i)]['x_train'])
                x_tests.append(predictor_dict['predictor'+str(i)]['x_test'])
            self.data_normaliser = data_normaliser
            self.y_normaliser = y_normaliser
            print('-----------Datasets and Different Models created-----------')
            if self.model_path is None:
                print('-----------Creating Stack Model-----------')
                self.predictor.build_model(self.forecast_days)
                print('-----------Stack Model created-----------')
                print('-----------Training Model-----------')
                evaluation = self.predictor.train_model(x_trains, y_train, x_tests, y_test, batch_size, epochs)
                print('-----------Model trained-----------')
            else:
                print('-----------Loading Model-----------')
                self.predictor.load_model()
                print('-----------Model loaded-----------')
            print('-----------Model Evaluation-----------')
            self.predicted_train,self.predicted_test = self.predictor.evaluate(self.y_normaliser, x_trains, x_tests, real_y_train, real_y_test)
            print('-----------Forecasting-----------')
            forecast = self.forecast_price(start_pred_date,False)
            if plot:
                print('-----------Plotting-----------')
                self.plot('train')
                self.plot('test')
                self.plot('full')
                self.plot('forecast',forecast)
                self.plot('full+forecast',forecast)
        else:
            print('-----------Creating Dataset-----------')
            self.highest_hp = self.history_points
            data = self.download_dataset(self.start_date,self.end_date)
            self.train,self.test = self.split_dataset(data)
            self.data_normaliser,x_train,y_train,real_y_train,self.y_normaliser,input_shape,output_shape = self.preprocess_dataset(self.train,'train')
            x_test,y_test,real_y_test = None,None,None#self.preprocess_dataset(self.test,'test',self.data_normaliser)
            print('-----------Dataset created-----------')
            if self.model_path is None:
                print('-----------Creating Model-----------')
                self.predictor.build_model(input_shape,output_shape)
                print('-----------Model created-----------')
                print('-----------Training Model-----------')
                evaluation = self.predictor.train_model(x_train, y_train, x_test, y_test, batch_size, epochs)
                print('-----------Model trained-----------')
            else:
                print('-----------Loading Model-----------')
                self.predictor.load_model()
                print('-----------Model loaded-----------')
            print('-----------Model Evaluation-----------')
            self.predicted_train,self.predicted_test = self.predictor.evaluate(self.y_normaliser, x_train, x_test, real_y_train, real_y_test)
            print('-----------Forecasting-----------')
            forecast = self.forecast_price(start_pred_date,False)
            if plot:
                print('-----------Plotting-----------')
                self.plot('train')
                self.plot('test')
                self.plot('full')
                self.plot('forecast',forecast)
                self.plot('full+forecast',forecast)