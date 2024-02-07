import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import warnings
import datetime
from datetime import datetime, timedelta


# ignore warnings
warnings.filterwarnings('ignore')

# define the ticker symbol
global ticker
# initialize variables
current_price = 0
forecast_unscaled = 0
previous_day_close = 0
percentage_change = 0

# load the pre-trained model
model = load_model(r'trained_model.h12')

# create a loop to continuously retrieve live data and make predictions
def gldp(instrument, timeframe):
    
    instrument = input("Enter the ticker symbol: ")
    # retrieve the live data from Yahoo Finance API
    live_data = yf.download(instrument, period='1y', interval='1d')
    live_data = live_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    current_price = live_data['Close'].values[-1]

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

    # Fit a separate scaler for the Close price
    close_scaler = MinMaxScaler()
    close_scaler.fit(live_data[['Close']])
    
    # scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(live_data.values)

    # reshape the data to include time step
    scaled_data = scaled_data[-2:, :]  # select the last 2 time steps
    scaled_data = np.expand_dims(scaled_data, axis=0)
    
    signal = model.predict(scaled_data)
    
    # Make a prediction on the scaled data
    forecast = model.predict(scaled_data)

    # Reshape the prediction to match the shape of the original data
    forecast = np.reshape(forecast, (-1, 1))

    
    # Unscale the forecast using the separate scaler
    forecast_unscaled = close_scaler.inverse_transform(forecast)
        
    print("Current price: ", current_price)
    print("Forecast: ", forecast_unscaled)
   
    return current_price, forecast_unscaled


gldp(instrument= 'any', timeframe= '1d')

    

    