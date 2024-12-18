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
import time
import logging
import pytz
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
import os
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get absolute path to .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
logger.debug(f"Looking for .env file at: {env_path}")

if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")

# Read with explicit UTF-8 encoding
with open(env_path, 'r', encoding='utf-8-sig') as f:
    env_contents = f.read()
    logger.debug(f"Raw .env contents:\n{env_contents}")

load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')

APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY') 
BASE_URL = os.getenv('APCA_BASE_URL')

logger.debug(f"Loaded API Key: {APCA_API_KEY_ID}")
logger.debug(f"Loaded Base URL: {BASE_URL}")

if not all([APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL]):
    raise ValueError("Environment variables not loaded correctly")

api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL, api_version='v2')

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Load trained LSTM model
model = load_model(r'trained_models\trained_model.h17')

def retry_api_call(func, max_retries=5, initial_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except (RequestException, ProtocolError) as e:
            wait = initial_delay * (2 ** retries)
            logging.warning(f"API call failed: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
            retries += 1
    
    logging.error(f"API call failed after {max_retries} retries.")
    raise Exception("Max retries reached")

def load_data(symbol, start_date, end_date):
    def fetch_data():
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Hour,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        ).df
        df = bars[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Adj Close'] = df['Close']
        return df

    try:
        data = retry_api_call(fetch_data)
        if data.empty:
            raise ValueError("Received empty dataset from Alpaca")
        logging.info("Data fetched successfully from Alpaca API")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch data from Alpaca API: {e}")
        # Fallback to yfinance
        logging.info("Falling back to yfinance for data")
        data = yf.download(symbol, start=start_date, end=end_date, interval='1h')
        if data.empty:
            raise ValueError("Received empty dataset from yfinance")
        logging.info("Data fetched successfully from yfinance")
        return data

def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MA'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data.loc[:, 'BB_upper'] = bb.bollinger_hband()
    data.loc[:, 'BB_middle'] = bb.bollinger_mavg()
    data.loc[:, 'BB_lower'] = bb.bollinger_lband()
    print(data)
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

def get_trade_signals(data, signals, num_timesteps=10, take_profit_pct=0.005, stop_loss_pct=0.1, risk_factor=1):
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

def get_position(symbol):
    try:
        position = api.get_position(symbol)
        return {
            'symbol': position.symbol,
            'qty': int(position.qty),
            'side': 'long' if int(position.qty) > 0 else 'short',
            'avg_entry_price': float(position.avg_entry_price)
        }
    except tradeapi.rest.APIError as e:
        if 'position does not exist' in str(e):
            return None
        else:
            raise

# def place_bracket_order(symbol, qty, side, limit_price, stop_loss_price, take_profit_price):
#     try:

#         if side not in ['buy', 'sell']:
#             logging.error("Bracket orders must be entry orders (buy or sell)")
#             return None
        
#         order = api.submit_order(
#             symbol=symbol,
#             qty=qty,
#             side=side,
#             type='limit',
#             time_in_force='gtc',
#             limit_price=limit_price,
#             order_class='bracket',
#             stop_loss={'stop_price': stop_loss_price},
#             take_profit={'limit_price': take_profit_price}
#         )
#         logging.info(f"Bracket order placed: {order}")
#         return order
#     except tradeapi.rest.APIError as e:
#         logging.error(f"Error placing order: {str(e)}")
#         return None

def place_trade(action, symbol, units, take_profit, stop_loss):
    try:
        # Validate take profit limit price
        if take_profit < stop_loss + 0.01:
            logging.error("take_profit.limit_price must be >= base_price + 0.01")
            return None
        
        if action == 'buy':
            order = api.submit_order(
                symbol=symbol,
                qty=units,
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': take_profit},
                stop_loss={'stop_price': stop_loss}
            )
        elif action == 'sell':
            order = api.submit_order(
                symbol=symbol,
                qty=units,
                side='sell',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': take_profit},
                stop_loss={'stop_price': stop_loss}
            )
        print(f"Placed {action} order for {units} shares of {symbol} with take profit at {take_profit} and stop loss at {stop_loss}")
        logging.info(f"Trade placed: {order}")
        return order
    except tradeapi.rest.APIError as e:
        print(f"Error placing order: {e}")
        logging.error(f"Error placing order: {str(e)}")
        return None

def close_position(symbol, qty):
    try:
        position = api.get_position(symbol)
        available_qty = int(position.qty)
        
        if qty > available_qty:
            logging.error(f"Insufficient qty available for order (requested: {qty}, available: {available_qty})")
            return None
        
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        logging.info(f"Position closed: {order}")
        return order
    except tradeapi.rest.APIError as e:
        logging.error(f"Error closing position {position}: {str(e)}")
        return None

def get_account():
    return api.get_account()

def exec_trades(trades, data, symbol, initial_balance=100000, investment_fraction=0.1):
    account = get_account()
    balance = float(account.cash)
    trade_history = []
    
    for trade in trades:
        index = trade['index']
        action = trade['action']
        price = trade['price']
        take_profit = round(trade['take_profit'], 2)
        stop_loss = round(trade['stop_loss'], 2)

        current_position = get_position(symbol)

        if action == 'buy' and (current_position is None or current_position['side'] == 'short'):
            # Close any existing short position
            if current_position and current_position['side'] == 'short':
                close_position(symbol, current_position['qty'])
                pnl = (current_position['avg_entry_price'] - price) * current_position['qty']
                trade_history.append({'action': 'buy_to_cover', 'price': price, 'units': current_position['qty'], 'pnl': pnl})
                
            # Open a long position
            amount_to_invest = balance * investment_fraction
            units = int(amount_to_invest // price)
            order = place_trade('buy', symbol, units, take_profit, stop_loss)
            if order:
                print(f"Bought {units} units at ${price:.2f} (Signal Index: {index})")
                trade_history.append({'action': 'buy', 'price': price, 'units': units})
            

        elif action == 'sell' and (current_position is None or current_position['side'] == 'long'):
            # Close any existing long position
            if current_position and current_position['side'] == 'long':
                close_position(symbol, current_position['qty'])
                pnl = (price - current_position['avg_entry_price']) * current_position['qty']
                trade_history.append({'action': 'sell', 'price': price, 'units': current_position['qty'], 'pnl': pnl})
            
            # Open a short position
            amount_to_invest = balance * investment_fraction
            units = int(amount_to_invest // price)
            order = place_trade('sell', symbol, units, take_profit, stop_loss)
            if order:
                print(f"Sold short {units} units at ${price:.2f} (Signal Index: {index})")
                trade_history.append({'action': 'sell_short', 'price': price, 'units': units})

        # Update account balance
        account = get_account()
        balance = float(account.cash)

    # Close any remaining positions at the end
    final_position = get_position(symbol)
    if final_position:
        close_position(symbol, final_position['qty'])
        final_price = data['Close'].iloc[-1]
        if final_position['side'] == 'long':
            pnl = (final_price - final_position['avg_entry_price']) * final_position['qty']
        else:
            pnl = (final_position['avg_entry_price'] - final_price) * final_position['qty']
        trade_history.append({'action': 'close_final_position', 'price': final_price, 'units': final_position['qty'], 'pnl': pnl})

    account = get_account()
    final_balance = float(account.portfolio_value)
    overall_pnl = final_balance - initial_balance
    overall_pnl_percent = (overall_pnl / initial_balance) * 100

    return final_balance, overall_pnl_percent, trade_history

def is_market_open():
    clock = api.get_clock()
    return clock.is_open

def wait_for_market_open():
    clock = api.get_clock()
    if not clock.is_open:
        time_to_open = clock.next_open - clock.timestamp
        logging.info(f'Market is closed. Waiting for {time_to_open.total_seconds() / 60:.2f} minutes until market opens.')
        time.sleep(time_to_open.total_seconds())

def get_trading_data(symbol, lookback_days=30):
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=lookback_days)
    return load_data(symbol, start_date, end_date)

def run_trading_loop(symbol, risk_factor, investment_fraction=0.1):
    while True:
        try:
            if not is_market_open():
                wait_for_market_open()
            
            # Get recent trading data
            data = get_trading_data(symbol)
            
            if data.empty:
                logging.warning("Received empty dataset. Retrying in 5 minutes...")
                time.sleep(300)
                continue
            
            # Calculate indicators and generate signals
            data = calculate_indicators(data)
            
            try:
                scaled_data, scaler = preprocess_data(data)
            except ValueError as e:
                logging.error(f"Preprocessing error: {str(e)}")
                time.sleep(300)
                continue
            
            X = create_lstm_input(scaled_data)
            signals = generate_signals(model, X)
            
            # Get trade signals
            trades = get_trade_signals(data, signals, risk_factor=risk_factor)
            
            # Execute trades
            balance, pnl_percent, trade_history = exec_trades(trades, data, symbol, investment_fraction=investment_fraction)
            
            # Log results
            logging.info(f"Current Balance: ${balance:.2f}, PnL%: {pnl_percent:.2f}%")
            
            # Wait for a short period before next iteration
            time.sleep(60)  

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {str(e)}")
            logging.error("Traceback:", exc_info=True)
            time.sleep(300)

logging.basicConfig(
    filename='trading_bot.log',  
    level=logging.DEBUG,         
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)

def main():
    symbol = 'SPY'
    risk_factor = 1.4
    print("Starting Trading bot, check logs")
    run_trading_loop(symbol, risk_factor)

if __name__ == "__main__":
    main()