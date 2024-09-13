import pandas as pd
import numpy as np
import math
import ta
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Load trained LSTM model
model = load_model(r'trained_models\trained_model.h17')

def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')
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
            print(f"Buy Signal Generated at Index: {i}, Price: {current_price}, 'take_profit': {take_profit}, 'stop_loss': {stop_loss}")
            
            # Check for sell signal
        elif should_sell(current_price, rsi, bb_sell_threshold, rsi_sell_threshold, signal) and current_price > ma and current_price > ema:
                take_profit = current_price * (1 - take_profit_pct)
                stop_loss = current_price * (1 + stop_loss_pct)
                trades.append({'index': i, 'action': 'sell', 'price': current_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'executed': False})
                print(f"Sell Signal Generated at Index: {i}, Price: {current_price}, 'take_profit': {take_profit}, 'stop_loss': {stop_loss}")
    
    return trades

def exec_trades(trades, data, initial_balance=100000, investment_fraction=0.1):
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
            print(f"Bought {units} units at ${entry_price:.2f} (Signal Index: {index})")
            

        elif action == 'sell' and position is None:
            # Open a short position
            amount_to_invest = balance * investment_fraction
            units = amount_to_invest // price
            entry_price = price
            balance += units * price
            position = 'short'
            print(f"Sold short {units} units at ${entry_price:.2f} (Signal Index: {index})")
            

        # Monitor the position for take profit or stop loss
        if position == 'long':
            for i in range(index, len(data)):
                current_price = data['Close'].iloc[i]
                if current_price >= take_profit or current_price <= stop_loss:
                    proceeds = units * current_price
                    balance += proceeds
                    print(f"Sold {units} units at ${current_price:.2f}, Profit: ${proceeds - (units * entry_price):.2f} (Signal Index: {index})")
                    position = None
                    units = 0
                    entry_price = 0
                    trade['executed'] = True
                    break

        elif position == 'short':
            for i in range(index, len(data)):
                current_price = data['Close'].iloc[i]
                if current_price <= take_profit or current_price >= stop_loss:
                    cost = units * current_price
                    balance -= cost
                    print(f"Bought back {units} units at ${current_price:.2f}, Profit: ${(units * entry_price) - cost:.2f} (Signal Index: {index})")
                    position = None
                    units = 0
                    entry_price = 0
                    trade['executed'] = True
                    break

        # Update portfolio value
        portfolio_value.append(balance + (units * data['Close'].iloc[index] if position == 'long' else 0))

    final_balance = balance + (units * data['Close'].iloc[-1] if position == 'long' else -units * data['Close'].iloc[-1])
    overall_pnl_percent = math.log(final_balance / initial_balance) * 100
    return portfolio_value, final_balance, overall_pnl_percent


def plot_executed_trades(data, trades):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    
    executed_buy_trades = [trade for trade in trades if trade['action'] == 'buy' and trade['executed']]
    executed_sell_trades = [trade for trade in trades if trade['action'] == 'sell' and trade['executed']]
    
    plt.scatter(data.index[[trade['index'] for trade in executed_buy_trades]], 
                data['Close'][[trade['index'] for trade in executed_buy_trades]], 
                marker='^', color='g', label='Executed Buy Trade', alpha=1)
    
    plt.scatter(data.index[[trade['index'] for trade in executed_sell_trades]], 
                data['Close'][[trade['index'] for trade in executed_sell_trades]], 
                marker='v', color='r', label='Executed Sell Trade', alpha=1)
    
    plt.title('Executed Trades')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    plot_executed_trades(data, trades)
    

    return portfolio_value, final_balance
if __name__ == "__main__":
    symbol = 'JPY=X'  
    risk_factor = 1.4 
    portfolio_value, final_balance = main(symbol, start_date, end_date, risk_factor)