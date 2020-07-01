'''
第一版：單一因子預測單一因子。
'''

"""
Factor：預測對象
step：預測步數
window：輸入長度
hours：預測長度(以小時記)
Target：
資料第1欄為PM10，第2欄為PM25(從0欄開始)
freq：時間序列分析的window大小。
一年共包含8760筆資料，時間序列分析後會在資料前後各刪除freq//2筆(因MA function)
訓練筆數為(資料筆數 - freq//2 - window + 1)

記得調整 to_supervised 1:3 or 8:10
"""

# multi headed multi-step cnn
import pandas as pd
import numpy as np
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from utils import save_model


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):	
	# calculate mse
	mse = mean_squared_error(actual, predicted)
	# calculate rmse
	rmse = sqrt(mse)
	return rmse

# convert history into inputs and outputs
def to_supervised(train, test, window, step, hours, Target, stage):
    
    if stage == 'training':
        Y_train = train[window:None, Target]
        X_train = []
        train_num = np.size(train, 0) - window
        # step over the entire history one time step at a time
        for i in range(train_num):
            PM = train[i:i + window, Target]
            X_train.append(PM)
        X_train = array(X_train)
        Y_train = array(Y_train)[:, np.newaxis]
        return X_train, Y_train
    
    elif stage == 'testing':
        Y_test = test[0:hours, Target]
        X_test = []
        test_num = hours
        shift = -(window + step - 1)
        temp = train[shift:None, :]
        test = np.concatenate((temp, test), axis=0)
        # step over the entire history one time step at a time
        for i in range(test_num):
            PM = test[i:i + window, Target]
            X_test.append(PM)         
        X_test = array(X_test)
        Y_test = array(Y_test)[:, np.newaxis]
        return X_test, Y_test

# train the model
def build_model(train, window, step, Target):
    # prepare training data
    train_x, train_y = to_supervised(train, test, window, step, hours, Target, 'training')
    # define parameters
    verbose, epochs, batch_size = 0, 25, 32
    n_timesteps, n_outputs = train_x.shape[1], train_y.shape[1]
    # create a channel for each variable
    model = Sequential()
    model.add(Dense(200, input_shape=(window,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # train_x = train_x.reshape(train_x.shape[0], train_x.shape[1])
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# make a forecast
def forecast(model, test_x, window, it):
    # retrieve last observations for input data
	input_x = test_x[it, :]
	# reshape into n input arrays
	input_x = input_x.reshape((1,input_x.shape[0]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, window, step, hours, Target):
    # fit model
    model = build_model(train, window, step, Target)
    # prepare testing data
    test_x, test_y = to_supervised(train, test, window, step, hours, Target, 'testing')
	# walk-forward validation over each week
    predictions = list()
    for i in range(hours):
        # predict the week
        yhat_sequence = forecast(model, test_x, window, i)
		# store the predictions
        predictions.append(yhat_sequence)
	# evaluate predictions days for each week
    predictions = array(predictions)
    rmse = evaluate_forecasts(test_y, predictions)
    #Save model
    save_model(model, './model/PM25_MLP_Single_Factor')
    return predictions, test_y, rmse


# setting parameters
Factor = 'PM25'
step = 1
window = 24
hours = 720
Target = 2

# load the new file
train = pd.read_excel("./data/106Zuoying.xlsx").values
test = pd.read_excel("./data/107Zuoying.xlsx").values

# evaluate model and get scores
predictions, test_y, rmse = evaluate_model(train, test, window, step, hours, Target)



