# -*- coding: utf-8 -*-

'''
第二版：強因子預測單一因子。
'''

import pandas as pd
import numpy as np
from numpy import array
from utilsV2 import evaluate_forecasts
from utilsV2 import load_model
from utilsV2 import to_supervised_multi


yhat = 0

# make a forecast
def forecastV1(PM10_model, PM25_model,  test_x, window, step, it):
    global yhat
    yhat_list = []
    
    # retrieve last observations for input data
    input_x = test_x[it, :, :]
    # reshape into n input arrays
    input_x = input_x.reshape((1,input_x.shape[0],input_x.shape[1]))
    
    for k in range(step):
        # forecast the next value
        PM10_yhat = PM10_model.predict(input_x, verbose=0)
        PM25_yhat = PM25_model.predict(input_x, verbose=0)
        # we only want the vector forecast
        PM10_yhat = PM10_yhat[0]
        PM25_yhat = PM25_yhat[0]
        yhat = array([PM10_yhat, PM25_yhat])
        input_x = np.delete(input_x, 0, 1)
        input_x_A = np.insert(input_x[:, :, 0], window - 1, PM10_yhat, 1)
        input_x_B = np.insert(input_x[:, :, 1], window - 1, PM25_yhat, 1)
        input_x = np.dstack((input_x_A, input_x_B))
        yhat_list.append(yhat)
    return yhat_list

# make a forecast
def forecastV2(PM10_model, PM25_model,  test_x, window, step, it):
    global yhat
    yhat_list = []
    
    # retrieve last observations for input data
    input_x = test_x[it, :, :]
    # reshape into n input arrays
    input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
    
    for k in range(step):
        # forecast the next value
        PM10_yhat = PM10_model.predict(input_x, verbose=0)
        PM25_yhat = PM25_model.predict(input_x, verbose=0)
        # we only want the vector forecast
        PM10_yhat = PM10_yhat[0]
        PM25_yhat = PM25_yhat[0]
        yhat = array([PM10_yhat, PM25_yhat])
        input_x = np.delete(input_x, 0, 2)
        input_x_A = np.insert(input_x[0, :], window - 1, PM10_yhat, 1)
        input_x_B = np.insert(input_x[1, :], window - 1, PM25_yhat, 1)
        input_x = [input_x_A, input_x_B]
        yhat_list.append(yhat)
    return yhat_list

# setting parameters
window = 24
step = 24
hours = 720
Target = 1
Factor = 'PM10'
Method = 'Strong'


PM10_CNN_model = load_model('./model/Monga/PM10_CNN_Strong_Factor')
PM25_CNN_model = load_model('./model/Monga/PM25_CNN_Strong_Factor')
PM10_LSTM_model = load_model('./model/Monga/PM10_LSTM_Strong_Factor')
PM25_LSTM_model = load_model('./model/Monga/PM25_LSTM_Strong_Factor')

PM10_list_performance = []
PM25_list_performance = []
Fixed_data = []

# load the new file
train = pd.read_excel("./data/106Monga.xlsx").values
test = pd.read_excel("./data/107Monga.xlsx").values

# split the data into training data and testing data
test_x, PM10_test_y, PM25_test_y, test_y = to_supervised_multi(train, test, window, step,
                                                             hours, Target, 'testing', Method)
predictions = list()

for j in range(hours + step - 1):
    
    # predict the trend and residual(PM10)
    yhat_sequence = forecastV1(PM10_LSTM_model, PM25_LSTM_model, test_x, window, step, j)
    yhat_sequence = array(yhat_sequence)
	# store the predictions(PM10)
    predictions.append(yhat_sequence)

# evaluate predictions days for each week
predictions = array(predictions)
predictions = np.squeeze(predictions, axis=(3,))

for i in range(24):
    
    if i == 0:
        M = None
    else:
        M = -i
        
    temp = predictions[23-i:M,i,:]
    Fixed_data.append(temp)


predictions = array(Fixed_data)
predictions = predictions.swapaxes(0,1)

test_y = np.hstack((PM10_test_y, PM25_test_y))

# Calculate RMSE、MSE、MAE and R2 Score between predicted values and observed values.
for step in range(0,24,1):
    PM10_rmse, PM10_mse, PM10_mae, PM10_R2 = evaluate_forecasts(test_y[:, 0], predictions[:, step, 0])
    PM25_rmse, PM25_mse, PM25_mae, PM25_R2 = evaluate_forecasts(test_y[:, 1], predictions[:, step, 1])
    PM10_performance = array([[PM10_rmse, PM10_mse, PM10_mae, PM10_R2]])
    PM25_performance = array([[PM25_rmse, PM25_mse, PM25_mae, PM25_R2]])
    PM10_list_performance.append(PM10_performance)    
    PM25_list_performance.append(PM25_performance)

P1 = array(PM10_list_performance)
P2 = array(PM25_list_performance)
P1 = P1.reshape((24,4))
P2 = P2.reshape((24,4))
d1 = pd.DataFrame(P1)
d2 = pd.DataFrame(P2)
d1.to_csv('PM10_LSTM_Strong.csv')
d2.to_csv('PM25_LSTM_Strong.csv')

