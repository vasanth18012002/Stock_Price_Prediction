import math
import pandas_datareader as web
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM


#Obtaining data,creating dataframe with CLose column , Converting dataframe into numpy values,finding the number of training rows 
df = web.DataReader('AAPL', data_source='yahoo', start ='2010-01-04', end ='2020-06-20')
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset) *.8) 


#Scaling the data 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)
 

#Create scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]
x_train=[]                                                #features
y_train = []                                              #target 
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Using LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


#COmpliling the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Create test data 
test_data = scaled_data[training_data_len - 60: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ] 
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


#Getting the predicted value of the model
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)


#Getting rmse 
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


print('Enter the date of the day before:\n')
end1=input()
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=end1)
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print (pred_price)