import pandas as pd
import numpy as np
import math
import ta
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

APCA_API_KEY_ID=('PK6E1S3XXEJ6SIFT9YK0')
APCA_API_SECRET_KEY=('Lrd8mLgbruIfThxbyYIkQF1ITfmnzP8N0cSMOhzO')
BASE_URL='https://paper-api.alpaca.markets'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL, api_version='v2')

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Load trained LSTM model
model = load_model(r'trained_models\trained_model.h17')

def load_data(symbol, start_date, end_date):
    try:
        # Format dates to 'YYYY-MM-DD'
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_date_str,
            end=end_date_str
        ).df
        df = bars[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Adj Close'] = df['Close']
        return df
    except tradeapi.rest.APIError as e:
        print(f"Alpaca API error: {e}")
        print("Falling back to Yahoo Finance data...")
        data = yf.download(symbol, start=start_date, end=end_date, interval='5m')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        return data

def calculate_indicators(data):
    data = data.copy()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MA'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data.loc[:, 'BB_upper'] = bb.bollinger_hband()
    data.loc[:, 'BB_middle'] = bb.bollinger_mavg()
    data.loc[:, 'BB_lower'] = bb.bollinger_lband()
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

def calculate_thresholds(rsi, bb_upper, bb_lower, risk_factor):
    rsi_buy_threshold = 30 * risk_factor
    rsi_sell_threshold = 70 / risk_factor
    bb_buy_threshold = bb_lower * risk_factor
    bb_sell_threshold = bb_upper / risk_factor
    return rsi_buy_threshold, rsi_sell_threshold, bb_buy_threshold, bb_sell_threshold

def should_buy(current_price, rsi, bb_buy_threshold, rsi_buy_threshold, signal):
    score = 0
    total_weight = 0
    
    conditions = [
        (current_price < bb_buy_threshold, 1.0),  
        (rsi < rsi_buy_threshold, 1.0),  
        (signal < 0.85, 1)     
    ]
    
    for condition, weight in conditions:
        total_weight += weight
        if condition:
            score += weight
    
    return score / total_weight >= 0.9

def should_sell(current_price, rsi, bb_sell_threshold, rsi_sell_threshold, signal):
    score = 0
    total_weight = 0
    
    conditions = [
        (current_price > bb_sell_threshold, 1.0),  
        (rsi > rsi_sell_threshold, 1.0),  
        (signal > 0.05, 1),       
    ]
    
    for condition, weight in conditions:
        total_weight += weight
        if condition:
            score += weight
    
    return score / total_weight >= 0.9

def get_trade_signals(data, signals, scaler, num_timesteps=10, take_profit_pct=0.05, stop_loss_pct=0.10, risk_factor=1):
    trades = []
    for i in range(num_timesteps, len(signals) + num_timesteps):
        current_price = data['Close'].iloc[i]
        rsi = data['RSI'].iloc[i]
        ma = data['MA'].iloc[i]
        ema = data['EMA'].iloc[i]
        bb_upper = data['BB_upper'].iloc[i]
        bb_middle = data['BB_middle'].iloc[i]
        bb_lower = data['BB_lower'].iloc[i]
        signal = signals[i - num_timesteps]
        
        # Calculate thresholds
        rsi_buy_threshold, rsi_sell_threshold, bb_buy_threshold, bb_sell_threshold = calculate_thresholds(rsi, bb_upper, bb_lower, risk_factor)

        # Check for buy signal
        if should_buy(current_price, rsi, bb_buy_threshold, rsi_buy_threshold, signal) and current_price < ma and current_price < ema:
            take_profit = current_price * (1 + take_profit_pct)
            stop_loss = current_price * (1 - stop_loss_pct)
            trades.append({'index': i, 'action': 'buy', 'price': current_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'executed': False})
            #print(f"Buy Signal Generated at Index: {i}, Price: {current_price}, 'take_profit': {take_profit}, 'stop_loss': {stop_loss}")
            
            # Check for sell signal
        elif should_sell(current_price, rsi, bb_sell_threshold, rsi_sell_threshold, signal) and current_price > ma and current_price > ema:
                take_profit = current_price * (1 - take_profit_pct)
                stop_loss = current_price * (1 + stop_loss_pct)
                trades.append({'index': i, 'action': 'sell', 'price': current_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'executed': False})
                #print(f"Sell Signal Generated at Index: {i}, Price: {current_price}, 'take_profit': {take_profit}, 'stop_loss': {stop_loss}")
    
    return trades

def place_trade(action, symbol, qty):
    if action == 'buy':
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        print(f"Placed buy order for {qty} shares of {symbol}")
    elif action == 'sell':
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Placed sell order for {qty} shares of {symbol}")

def exec_trades(trades, data, initial_balance=10000, investment_fraction=0.1):
    balance = initial_balance
    portfolio_value = []
    position = None
    units = 0
    entry_price = 0
    
    for trade in trades:
        index = trade['index']
        action = trade['action']
        price = trade['price']
        take_profit = trade['take_profit']
        stop_loss = trade['stop_loss']
        trade['executed'] = False

        if action == 'buy' and position is None:
            # Open a long position
            amount_to_invest = balance * investment_fraction
            units = amount_to_invest // price
            entry_price = price
            balance -= units * price
            position = 'long'
            place_trade('buy', 'AAPL', units)  # Example symbol
            print(f"Bought {units} units at ${entry_price:.2f} (Signal Index: {index})")
            

        elif action == 'sell' and position is None:
            # Open a short position
            amount_to_invest = balance * investment_fraction
            units = amount_to_invest // price
            entry_price = price
            balance += units * price
            position = 'short'
            place_trade('sell', 'AAPL', units)  # Example symbol
            print(f"Sold short {units} units at ${entry_price:.2f} (Signal Index: {index})")
            

        # Monitor the position for take profit or stop loss
        if position == 'long':
            for i in range(index, len(data)):
                current_price = data['Close'].iloc[i]
                if current_price >= take_profit or current_price <= stop_loss:
                    proceeds = units * current_price
                    balance += proceeds
                    place_trade('sell', 'AAPL', units)  # Example symbol
                    print(f"Closed long position at ${current_price:.2f} (Signal Index: {i})")
                    position = None
                    break

        elif position == 'short':
            for i in range(index, len(data)):
                current_price = data['Close'].iloc[i]
                if current_price <= take_profit or current_price >= stop_loss:
                    proceeds = units * current_price
                    balance -= proceeds
                    place_trade('buy', 'AAPL', units)  # Example symbol
                    print(f"Closed short position at ${current_price:.2f} (Signal Index: {i})")
                    position = None
                    break

def main(symbol, start_date, end_date, risk_factor):
    data = load_data(symbol, start_date, end_date)
    data = calculate_indicators(data)
    scaled_data, scaler = preprocess_data(data)
    X = create_lstm_input(scaled_data)
    signals = generate_signals(model, X)
    trades = get_trade_signals(data, signals, scaler, risk_factor=risk_factor)
    portfolio_value, final_balance, overall_pnl_percent = exec_trades(trades, data)
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Overall PnL%: {overall_pnl_percent:.2f}%")
    return portfolio_value, final_balance

if __name__ == "__main__":
    symbol = 'SPY'  
    risk_factor = 1.4 
    portfolio_value, final_balance = main(symbol, start_date, end_date, risk_factor)