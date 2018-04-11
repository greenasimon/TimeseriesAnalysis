# Databricks notebook source
########################FB Stock Price Prediction using LSTM##############################################################

# COMMAND ----------

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import concatenate

# COMMAND ----------

#Dataset for fundamental analysis
filePath1 = '/FileStore/tables/FBPrices.csv' 
#Move the dataset into a spark Dataframe
stock_prices = spark.read.option("header","true"). option("inferSchema","true").csv(filePath1)

# COMMAND ----------

#####################Data Preprocessing################################################################################################

# COMMAND ----------

#Convert Spark Dataframe to a Pandas Dataframe
pd_fbstockprices = stock_prices.toPandas() 

pd_fbstockprices["temp_volume"] = pd_fbstockprices.volume
pd_fbstockprices.drop(['volume'], 1, inplace=True)
#Move the preditions variable 'close' to the last column
pd_fbstockprices["temp_close"] = pd_fbstockprices.close
pd_fbstockprices.drop(['close'], 1, inplace=True) #inplace : If True, do operation inplace and return None.
pd_fbstockprices=pd_fbstockprices.rename(columns = {'temp_close':'close','temp_volume':'volume'})

pd_fbstockprices.head()


# COMMAND ----------

#Droping first row as it is a repetition of the second row
pd_fbstockprices = pd_fbstockprices[1:]
#Order the data in the ascending order of the date
pd_fbstockprices.sort('date',inplace = True)
#Convert date column to date type and index it
pd_fbstockprices['date'] = pd.to_datetime(pd_fbstockprices['date'])
indexed_fbstocks = pd_fbstockprices.set_index('date')

indexed_fbstocks.head()

# COMMAND ----------

plt.plot(indexed_fbstocks["close"])
display(plt.show())

# COMMAND ----------

#LSTMs are sensitive to data scales when activation functions are used.Its a good practise to range the data between 0,1
#Normalising using MinMaxscaler
values = indexed_fbstocks.values.astype('float32')
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_stock = scaler.fit_transform(values)
print(scaled_stock)

# COMMAND ----------

#########################################Preparing Dataset for LSTM##########################################################


# COMMAND ----------

cols, cols_names = list(), list()
features = scaled_stock.shape[1] #Number of features
df = pd.DataFrame(scaled_stock)

#Timestep is the number of previous time steps to be used as input to predict the next time step
#We can change values to see which one gives the best results for our model
input_ts = 10

#Based on the value of the timestep variable 'input_ts', values at t-n, ... t-1 are appended to the dataset 
#for all the features to create an input sequence
#Hence, one input sequence will have (input_ts * features) number of variables 
for i in range(input_ts, 0, -1):
  cols.append(df.shift(i))
  cols_names += [('feature%d(t-%d)' % (j+1, i)) for j in range(features)]
  
#Sequence (t) is also appended which will be our output timestep, ie, the value of timestep 't' is 
#predicted using the data for timesteps from (t-1) to (t-input_timeSteps)
cols.append(df.shift(0))
cols_names += [('feature%d(t)' % (j+1)) for j in range(features)]

ModifiedSeq = concat(cols, axis=1)
ModifiedSeq .columns = cols_names
ModifiedSeq .dropna(inplace=True) # Omitting rows with any of the missing values
print(ModifiedSeq)

# COMMAND ----------

#####################################Training and Test data sets#############################################################
#We cannot use cross validation method here to validate our model because sequence is important in time series.##############
#Instead, we can split our past data into train data and test data.##########################################################

# COMMAND ----------

#Using 70% of the data for training and 30% for testing
train_size = int(len(ModifiedSeq.values) * 0.7)
test_size = len(ModifiedSeq.values) - train_size
train, test = ModifiedSeq.values[0:train_size,:], ModifiedSeq.values[train_size:len(ModifiedSeq.values),:]
print(len(train), len(test))

# COMMAND ----------

#Out of the total ((input_ts * features) + features) columns added by the above steps, the first (input_ts * features) columns
#that correspond to the timesteps (t-1) to (t-input_ts) will be our input features. Our prediction variable,
#which is the last column (that corresponds to 'close' price at timestep (t)) will be our 'y' variable.
trainX = train[:, :features*input_ts] 
trainY = train[:,-1]
    
testX = test[:, :features*input_ts] 
testY = test[:,-1]
 
#Input is reshaped to 3D [Number of rows, timesteps, Number of features] . Input requirement for LSTM   
trainX = np.reshape(trainX,(trainX.shape[0], input_ts,features))
testX = np.reshape(testX,(testX.shape[0], input_ts, features))

# COMMAND ----------

############################################# Build and fit LSTM Model #############################################

# COMMAND ----------

#Build the LSTM model
model = Sequential()
model.add(LSTM(250, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True, activation='linear')) #memory between batches
model.add(LSTM(250, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False, activation='linear'))
model.add(Dense(1))

#model.add(Dense(32,kernel_initializer="uniform",activation='relu'))  

model.compile(loss='mse', optimizer='adam') #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#Fit our model on the Training Set
ret = model.fit(trainX, trainY, epochs=100, batch_size=50, validation_split=0.2, verbose=2, shuffle=False)

# COMMAND ----------

# model.fit() method returns a History object. Its History.history attribute records training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values

fig = plt.figure()
plt.plot(ret.history['loss'],label='Train Loss')
plt.plot(ret.history['val_loss'],label='Test Loss')
plt.title('Loss at successive epochs')
plt.legend(loc='best')
display(plt.show())
plt.close(fig)

#From the plot, we can see that the model has comparable performance on both train and validation datasets.

# COMMAND ----------

# We scaled our dataset before feeding to the LSTM. As a result, our prediction variable also has a scaled value. Therefore, we should invert it to the orginal form and then compare with the actual value to get the rmse. This will give us an error measurement in the same unit as the original variable.

#Make predictions for our test data
predictions = model.predict(testX)

# COMMAND ----------

print(combined2)

# COMMAND ----------

#Invert the scales for predictions
testX = testX.reshape((testX.shape[0], input_ts*features))
combined1 = concatenate((predictions, testX[:, -4:]), axis=1)
inverted_predictions = scaler.inverse_transform(combined1)[:,0]

#Invert the scales for original data
testY = testY.reshape((len(testY), 1))
combined2 = concatenate((testY, testX[:, -4:]), axis=1)
inverted_original = scaler.inverse_transform(combined2)[:,0]
#Calculate RMSE
rmse = sqrt(mean_squared_error(inverted_original, inverted_predictions))
print('Test RMSE: %.3f' % rmse)

# COMMAND ----------

fig = plt.figure()
plt.plot(inverted_predictions, color='red', label='Prediction')
plt.plot(inverted_original, color='blue', label='Original')
plt.legend(loc='best')
display(plt.show())
plt.close(fig)

# COMMAND ----------


