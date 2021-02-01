import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.layers import Flatten, Dense, Conv1D, MaxPooling2D
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.layers import Input, LSTM,Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import pandas as pd

class RNN():
    def __init__(self):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        if len(tf.config.experimental.list_physical_devices('GPU'))<1:
            print('No working GPU, need to stop here')
            return
        else:   
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession

            config = ConfigProto()
            config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)
            tf.config.experimental.list_physical_devices('GPU')
            self.run()

    def run(self):
        model, MAE,RMSE, X_train, y_train, X_test, y_test = self.get_score((10,100))
        print('RNN MAE:',MAE, 'RNN RMSE:',RMSE)
    
    def tomatrix(self,vectorSeries, sequence_length):
        matrix=[]
        for i in range(len(vectorSeries)-sequence_length+1):
            matrix.append(vectorSeries[i:i+sequence_length])
        return matrix

    
    def get_model(self,units1, units2):
        # build the model
        model = Sequential()
        # layer 1: LSTM
        model.add(LSTM(input_dim=1,units=units1, return_sequences=True))
        model.add(Dropout(0.2))
        # layer 2: LSTM
        model.add(LSTM(units=units2, return_sequences=False))
        model.add(Dropout(0.2))
        # layer 3: dense
        # linear activation: a(x) = x
        model.add(Dense(units=1, activation='linear'))
        #
        return model


    def get_score(self,hyperparameters):
        
        hyperparameters = np.array(hyperparameters,dtype=int)
        sequence_length, units = hyperparameters

        # random seed to make sure we always obtain equal results
        np.random.seed(1234)

        # load the data
        RVOL = np.sqrt(252*pd.read_pickle('./datafiles/asml_RK.pickle'))

        # convert the vector to a 2D matrix
        matrix_RVOL = self.tomatrix(RVOL.values.flatten(), sequence_length)
        
        # shift all data by mean
        matrix_RVOL = np.array(matrix_RVOL)
        shifted_value = matrix_RVOL.mean()
        matrix_RVOL -= shifted_value
        print ("Data  shape: ", matrix_RVOL.shape)

        # split dataset
        train_row = int(round(0.9 * matrix_RVOL.shape[0]))
        train_set = matrix_RVOL[:train_row, :]

        np.random.shuffle(train_set)
        # the training set
        X_train = train_set[:, :-1]
        # the last column is the true value to compute the mean-squared-error loss
        y_train = train_set[:, -1] 
        # the test set
        X_test = matrix_RVOL[train_row:, :-1]
        y_test = matrix_RVOL[train_row:, -1]
        test_dates = RVOL.index[train_row+sequence_length-1:]


        # the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = self.get_model(units, units)

        # compile the model
        model.compile(loss="mae", optimizer="rmsprop")

        # train the model
        model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.05, verbose=0)

        MAE = np.mean(np.abs(model(X_test).numpy().flatten() - y_test))
        MSE = np.mean((model(X_test).numpy().flatten() - y_test)**2)
        RMSE = np.sqrt(MSE)
        

        return model, MAE,RMSE, X_train, y_train, X_test, y_test
    def hyperopt(self):
        window_range = np.arange(6,30,2)
        unitrange = np.arange(30,120,10)
        
        MAEdf = pd.DataFrame({'w':[]}).set_index(['w'])
        RMSEdf = pd.DataFrame({'w':[]}).set_index(['w'])
        

        for w in window_range:
            for u in unitrange:
                MAE,RMSE = self.get_score((w,u))
                MAEdf.loc[w,u] = MAE
                RMSEdf.loc[w,u] = RMSE
        return RMSEdf, MAEdf
