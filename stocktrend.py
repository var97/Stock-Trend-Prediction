# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #.values is used for creating a numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #default feature range
"""We have to take new variable and apply fit_transform method from minmax scaler class. Here fit means it is going to
get the min and max of the input data so that it could apply values to the Normalisation formulae. And from that transform method
it is going to compute scaled stock prices for each of the stock prices from the training set according to the Normalisation Formulae"""
training_set_scaled = sc.fit_transform(training_set) 

# Creating a data structure with 60 timesteps and 1 output
"""60 timesteps means that for each time t rnn is going to look at 60 stock prices before time t that is stock prices between 60 days before time t and time t
and based on the trends it is capturing during the 60 periods time steps it is going to predict the next output"""
X_train = [] # inputs to the NN
y_train = [] # output to the NN
for i in range(60, 1258): #to populate X_train and y_train
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#Here we are getting 3d structure for X_train by adding third dimension as indicator.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN using stacked LSTMs

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

"""Initialising the RNN as a sequence of layers as opposed to computational graphs
here regressor is used because we are predicting continous values."""
regressor = Sequential()

"""Adding the first LSTM layer and some Dropout regularisation
  regressor is an object to sequential class. Sequential class contains add method which is used here.
  Then we add lstm layer using lstm class. 
  It contains three arguments: 1. No of Units/neurons 2.To create stacked(more than 1 layer) LSTM we have to set return_sequences = True
  3. input_shape contains three dimensions no of observations, no of timesteps and no of indicators, here we have taken last two dimensions because first one will
  be automatically considered."""
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#here we are using dropout regularization in order to prevent overfitting in the model.
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')#here loss is mse because we are doing regression here

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
"""Since here our dataset is less that's why we have set our epochs = 100, We can also increase no of epochs but it might 
leads to overfitting which is bad for our model."""



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values #converting it into numpy array using .values at end

# Getting the predicted stock price of 2017
"""Here to predict the stock price of particular day we have to take input as a stockprices of previous 60 days.
And while predicting the output some of the stock prices in Jan 2017 have their inputs as the stock prices in test set.
Therefore we have to concatenate both the dataset and forms the total dataset. """
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#here we have not used iloc method from pandas to get inputs so here there are chances that we can face format problem so we have to reshape the inputs using reshape function
inputs = inputs.reshape(-1,1)
#scaling the inputs
inputs = sc.transform(inputs)
#loading the X_test list with values for predicting the Jan 2017 stock prices
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
#adding third dimension in X_test 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#using predict method for predicting values Using X_test as input values
predicted_stock_price = regressor.predict(X_test)
#inverse_transform method is used for obtaining unscaled/original values.
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'black', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

"""
Conclusion: So here we can see that our model can closely predict the trends which are quite similar to the trends in the Google stock real prices.
"""
