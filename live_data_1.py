import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time

# define the ticker symbol
ticker = "MSFT"

# load the pre-trained model
model = load_model(r'/Users/samuelzobev/Downloads/End of year project/trained_model.h10')

# create a loop to continuously retrieve live data and make predictions
while True:
    # retrieve the live data from Yahoo Finance API
    live_data = yf.download(ticker, period="1d", interval="1m")
    live_data = live_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

    # calculate technical indicators
    live_data['RSI'] = ta.momentum.RSIIndicator(live_data['Close'], window=14).rsi()
    live_data['MA'] = ta.trend.SMAIndicator(live_data['Close'], window=20).sma_indicator()
    live_data['EMA'] = ta.trend.EMAIndicator(live_data['Close'], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(live_data['Close'], window=20, window_dev=2)
    live_data['BB_upper'] = bb.bollinger_hband()
    live_data['BB_middle'] = bb.bollinger_mavg()
    live_data['BB_lower'] = bb.bollinger_lband()

    # remove nan values
    live_data.dropna(inplace=True)

    # scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(live_data.values)

    # reshape the data to include time step
    scaled_data = scaled_data[-2:, :]  # select the last 2 time steps
    scaled_data = np.expand_dims(scaled_data, axis=0)

    # make predictions on the last sequence of the input data
    predictions = model.predict(scaled_data)

    print(predictions.shape)
    print(predictions)

    # wait for 1 minute before retrieving new data and making new predictions
    time.sleep(60)