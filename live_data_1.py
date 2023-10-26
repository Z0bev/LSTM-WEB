import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time

# define the ticker symbol
ticker = "MSFT"

# load the pre-trained model
model = load_model(r'C:\Users\zobev\Desktop\EYP\trained_model.h10')

# create a loop to continuously retrieve live data and make predictions
while True:
    # retrieve the live data from Yahoo Finance API
    live_data = yf.download(ticker, period="1d", interval="1m")
    live_data = live_data[['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close']]

    # add lagged features
    lagged_data = live_data.shift(1)
    lagged_data.columns = [f'{col}_lag1' for col in lagged_data.columns]
    live_data = pd.concat([live_data, lagged_data], axis=1)
    
    # remove the lagged features from the original data
    live_data = live_data.iloc[1:, :]

    # scale the input data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(live_data.values.reshape(live_data.shape[0], 12))

    # remove the last two rows of the scaled data to predict only 1 day ahead while looking at 2 days behind
    scaled_data = scaled_data[:-2, :]
    scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], 2, 12))

    # make predictions on the last row of the input data
    predictions = model.predict(scaled_data[-1:])

    print(predictions.shape)
    
    # assume that 'predictions' is the array with shape (n, 1)
    n = predictions.shape[0]
    remainder = n % 12
    if remainder != 0:
        predictions = predictions[:-remainder, :]

    # reshape the predictions array to shape (n, 12)
    predictions_reshaped = np.reshape(predictions, (-1, 12))

    # unscale the predicted values
    unscaled_predictions = scaler.inverse_transform(predictions_reshaped)
    
    # print the unscaled predictions
    print(unscaled_predictions)

    # wait for 1 minute before retrieving new data and making new predictions
    time.sleep(20)