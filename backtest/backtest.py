import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load your trained LSTM model
model = load_model('/Users/samuelzobev/Downloads/End of year project/trained_models/trained_model.h14')

# Load your historical data
data = pd.read_csv('/Users/samuelzobev/Downloads/End of year project/data_collection/AMZN.csv', date_parser=True)
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
data['MA'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
data['BB_upper'] = bb.bollinger_hband()
data['BB_middle'] = bb.bollinger_mavg()
data['BB_lower'] = bb.bollinger_lband()
data = data[['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close', 'RSI', 'MA', 'EMA', 'BB_upper', 'BB_middle', 'BB_lower']]
data = data.set_index('Date')

# Drop any rows with missing values
data = data.dropna()

# Select the features you want to use
features = data[['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close', 'RSI', 'MA', 'EMA', 'BB_upper', 'BB_middle', 'BB_lower']]

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Reshape the data into the format expected by the LSTM model
# LSTM expects input in the form of a 3D array: (num_samples, num_timesteps, num_features)
# Here im using a sliding window of 10 timesteps
num_timesteps = 10
X = []
for i in range(num_timesteps, len(scaled_data)):
    X.append(scaled_data[i-num_timesteps:i])

X = np.array(X)

# Use your LSTM model to generate trading signals
signals = model.predict(X)

# Initialize variables
holdings = 0
capital = 10000  
last_sell_price = float('inf')
trades = []
last_buy_price = 0

# Loop over  data
for i in range(len(signals)):
    # Get the current price
    current_price = data['Close'].iloc[i] 

    # Get the trading signal
    signal = signals[i]

    # Get the moving average, RSI, and Bollinger Bands
    ma = data['MA'].iloc[i]
    rsi = data['RSI'].iloc[i]
    bb_upper = data['BB_upper'].iloc[i]
    bb_lower = data['BB_lower'].iloc[i]

    # Decide whether to buy, sell, or hold
    # Buy if the price is below the lower Bollinger Band, the RSI is less than 30 (indicating oversold conditions), and the LSTM signal is less than 0.4
    if current_price < bb_lower and rsi < 30 and signal < 0.5 and capital > current_price:
        stocks_bought = (capital * 0.1) // current_price
        holdings += stocks_bought
        capital -= stocks_bought * current_price
        trades.append({'index': i, 'action': 'buy', 'price': current_price, 'stocks_bought': stocks_bought, 'capital': capital})
    # Sell if the price is above the upper Bollinger Band, the RSI is greater than 70 (indicating overbought conditions), and the LSTM signal is greater than 0.7
    elif current_price > bb_upper and rsi > 70 and signal > 0.7 and holdings > 0:
        stocks_sold = holdings * 0.4
        capital += stocks_sold * current_price
        holdings -= stocks_sold
        trades.append({'index': i, 'action': 'sell', 'price': current_price, 'stocks_sold': stocks_sold, 'capital': capital})

# Calculate PnL
realized_pnl = capital - 10000
unrealized_pnl = (holdings * data['Close'].iloc[-1]) + capital - 10000

#Calculate total PnL in percentage
total_pnl = (realized_pnl + unrealized_pnl)/10000 * 100

# Print PnL
print(f'Realized Profit: {realized_pnl}')
print(f'Unrealized Profit: {unrealized_pnl}')
print(f'P/L: {total_pnl}%')

# Print all the trades
for trade in trades:
    print(trade)

# Print current holdings
print(f'Current holdings: {holdings}')

# Plot the trades on the price chart
buy_signals = [trade['index'] for trade in trades if trade['action'] == 'buy']
sell_signals = [trade['index'] for trade in trades if trade['action'] == 'sell']
plt.plot(data['Close'], label='Close Price')
plt.scatter(buy_signals, data['Close'].iloc[buy_signals], color='g', marker='^', label='Buy')
plt.scatter(sell_signals, data['Close'].iloc[sell_signals], color='r', marker='v', label='Sell')
plt.legend()
plt.show()