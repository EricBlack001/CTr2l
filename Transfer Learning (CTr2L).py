#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation,GRU
from keras import layers
from keras.optimizers import Adam
import numpy as np
import random
import os
import tensorflow as tf
df=pd.read_csv("/Users/apple/Desktop/guangdong.csv",parse_dates=["TradingDate"],index_col=[0])
scaler = MinMaxScaler(feature_range=(0,1))
df =scaler.fit_transform(df)
df_for_training_scaled=df[:1200]
df_for_testing_scaled=df[1200:]
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:-1])
            dataY.append(dataset[i,-1])
    return np.array(dataX),np.array(dataY)

x_train,y_train=createXY(df_for_training_scaled,5)
x_test,y_test=createXY(df_for_testing_scaled,5)
print(x_test.shape)
x_train = np.reshape(x_train, (x_train.shape[0], 5, x_train.shape[2]))
x_test  = np.reshape(x_test, (x_test.shape[0], 5, x_test.shape[2]))

model = Sequential()
model.add(GRU(5,return_sequences=True))
model.add(GRU(5))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam') 
model.optimizer.lr.assign(0.0005)
history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=200, 
                    validation_data=(x_test, y_test), 
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()
for layer in model.layers:    
    print(layer.trainable)
for layer in model.layers [:-2]:
    print(layer.name)
    layer.trainable = False
for layer in model.layers:    
    print(layer.trainable)
from keras.models import Model
from keras.models import Sequential
model2 = Sequential()
for layer in model.layers[:-2]:  # 跳过最后2层 
    model2.add(layer)
from keras.layers import Dense 
model2.add(GRU(10))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam') 
model2.optimizer.lr.assign(0.0005)
for layer in model2.layers:   
    print(layer.trainable)
df=pd.read_csv("/Users/apple/Desktop/shanghai.csv",parse_dates=["TradingDate"],index_col=[0])
scaler = MinMaxScaler(feature_range=(0,1))
df =scaler.fit_transform(df)
df_for_training_scaled=df[:-105]
df_for_testing_scaled=df[-105:]

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:-1])
            dataY.append(dataset[i,-1])
    return np.array(dataX),np.array(dataY)

x_train,y_train=createXY(df_for_training_scaled,5)
x_test,y_test=createXY(df_for_testing_scaled,5)
print(x_test.shape)
x_train = np.reshape(x_train, (x_train.shape[0], 5, x_train.shape[2]))
x_test  = np.reshape(x_test, (x_test.shape[0], 5, x_test.shape[2]))
from tensorflow.keras.callbacks import EarlyStopping

# 设置 EarlyStopping 回调
early_stop = EarlyStopping(monitor='val_loss',   
                           patience=10,         
                           min_delta=0.0171-0.0170, 
                           restore_best_weights=True)  

# 训练模型
history = model2.fit(x_train, y_train, 
                     batch_size=32, 
                     epochs=100, 
                     validation_data=(x_test, y_test), 
                     validation_freq=1
                     ) 
predicted = model.predict(x_test) 

# Since we scaled the entire dataset with all features, we need to reverse the scaling correctly.
# Create an empty array of the same shape as the original scaled data
predicted_with_full_shape = np.zeros((predicted.shape[0], df.shape[1]))

# Insert the predicted values into the last column of this array (since that's where the target variable was during scaling)
predicted_with_full_shape[:, -1] = predicted[:, 0]

# Now we can safely inverse transform
predicted_original_scale = scaler.inverse_transform(predicted_with_full_shape)

# The predicted values are now in their original scale, but we only care about the target column
# So we extract the last column
predicted_final = predicted_original_scale[:, -1]
TL=predicted_final

data = pd.read_csv("/Users/apple/Desktop/shanghai.csv", delimiter=",")
date = data['TradingDate'].tail(100)
vola=data['vola'].tail(100)
l1 =TL
plt.plot(date,l1, color='#F38181',linewidth=2.5,label='CTr2L')
plt.plot(date,vola, color='#95E1D3',linewidth=2.5,label='Historical Volatility')
plt.grid(alpha=0.8) 
plt.xticks(range(1,len(date),22))
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(fontsize='small')
plt.show()

