import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
import time
import logging

# Load trained LSTM model
model = load_model(r'trained_models/trained_model.h17')

def load_latest_data(symbol, period="30d"):
    data = yf.download(symbol, period=period, interval='5m')
    return data

def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MA'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()
    return data

def preprocess_data(data):
    data = data.dropna()
    features = data[['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close', 'RSI', 'MA', 'EMA', 'BB_upper', 'BB_middle', 'BB_lower']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    return scaled_data, scaler

def create_lstm_input(scaled_data, num_timesteps=10):
    X = []
    for i in range(num_timesteps, len(scaled_data)):
        X.append(scaled_data[i-num_timesteps:i])
    return np.array(X)

def generate_signals(model, X):
    return model.predict(X)

def get_latest_signal(data, signals, num_timesteps=10, take_profit_pct=0.05, stop_loss_pct=0.1, risk_factor=1):
    current_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    ma = data['MA'].iloc[-1]
    ema = data['EMA'].iloc[-1]
    bb_upper = data['BB_upper'].iloc[-1]
    bb_lower = data['BB_lower'].iloc[-1]
    signal = signals[-1]
    
    # Calculate thresholds
    rsi_buy_threshold = 30 * risk_factor
    rsi_sell_threshold = 70 / risk_factor
    bb_buy_threshold = bb_lower * risk_factor
    bb_sell_threshold = bb_upper / risk_factor

    # Check for buy signal
    if (current_price < bb_buy_threshold and 
        rsi < rsi_buy_threshold and 
        signal < 0.85 and 
        current_price < ma and 
        current_price < ema):
        take_profit = current_price * (1 + take_profit_pct)
        stop_loss = current_price * (1 - stop_loss_pct)
        return {'timestamp': data.index[-1], 'action': 'buy', 'price': current_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    
    # Check for sell signal
    elif (current_price > bb_sell_threshold and 
          rsi > rsi_sell_threshold and 
          signal > 0.05 and 
          current_price > ma and 
          current_price > ema):
        take_profit = current_price * (1 - take_profit_pct)
        stop_loss = current_price * (1 + stop_loss_pct)
        return {'timestamp': data.index[-1], 'action': 'sell', 'price': current_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    
    return None

def run_signal_generator(symbol, risk_factor=1.4, interval=60):
    while True:
        # Get the latest data
        data = load_latest_data(symbol)
        
        # Calculate indicators and generate signals
        data = calculate_indicators(data)
        scaled_data, scaler = preprocess_data(data)
        X = create_lstm_input(scaled_data)
        signals = generate_signals(model, X)
        
        # Get the latest trade signal
        latest_signal = get_latest_signal(data, signals, risk_factor=risk_factor)
        
        # Output new signal if available
        if latest_signal:
            print(f"Signal: {latest_signal['action']} {symbol} at {latest_signal['timestamp']}")
            print(f"Price: ${latest_signal['price']:.2f}")
            print(f"Take Profit: ${latest_signal['take_profit']:.2f}")
            print(f"Stop Loss: ${latest_signal['stop_loss']:.2f}")
            print("---")
        else:
            print(f"No new signal at {datetime.now()}")
        
        # Wait for the next interval
        time.sleep(interval)

if __name__ == "__main__":
    symbol = 'JPY=X'
    risk_factor = 1.4
    run_signal_generator(symbol, interval=300)