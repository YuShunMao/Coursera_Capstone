# -*- coding: utf-8 -*-
from sklearn.externals import joblib #jbolib模块
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
import numpy as np
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv

# setting parameters
Factor = 'PM25'
step = 1
window = 24
hours = 720
Target = 2

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

# make a forecast
def forecast(model, test_x, window, it):
    # retrieve last observations for input data
	input_x = test_x[it, :]
	# reshape into n input arrays
	input_x = input_x.reshape((1,input_x.shape[0]))
	# forecast the next week
	yhat = model.predict(input_x)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

def evaluate_model(train, test, window, step, hours, Target):
    # prepare testing data
    train_x, train_y = to_supervised(train, test, window, step, hours, Target, 'training')
    test_x, test_y = to_supervised(train, test, window, step, hours, Target, 'testing')
    
    model = KernelRidge(alpha=1.0)
    model.fit(train_x, train_y)

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
    joblib.dump(model, 'model/PM25_KRR_Single_Factor.pkl')
    return predictions, test_y, rmse

# load the new file
train = pd.read_excel("./data/106Zuoying.xlsx").values
test = pd.read_excel("./data/107Zuoying.xlsx").values

# evaluate model and get scores
predictions, test_y, rmse = evaluate_model(train, test, window, step, hours, Target)



