import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import warnings
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import mean_absolute_percentage_error

app = Flask(__name__, static_folder=r'C:\Users\zobev\LSTM-WEB\static', template_folder=r'C:\Users\zobev\LSTM-WEB\template')

# load the pre-trained model
model = load_model(r'trained_models\trained_model.h17')

@app.route('/')
def home():
    return render_template('index.html')

warnings.filterwarnings('ignore')

def get_live_data_and_predict(instrument, timeframe):
    # retrieve the live data from Yahoo Finance API
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year of historical data
    live_data = yf.download(instrument, start=start_date, end=end_date, interval='1d')
    live_data = live_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # get the current price
    current_price = live_data['Close'][-1]

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

    # Calculate accuracy
    accuracy = calculate_accuracy(live_data, scaled_data, close_scaler)

    # reshape the data to include time step
    scaled_data = scaled_data[-10:, :]
    scaled_data = np.expand_dims(scaled_data, axis=0)
    
    # Make predictions based on timeframe
    if timeframe == '1day':
        forecast_steps = 1
    elif timeframe == '1week':
        forecast_steps = 5  
    elif timeframe == '1month':
        forecast_steps = 22
    else:
        raise ValueError("Invalid timeframe. Please choose from '1day', '1week', '1month'.")
    
    forecasts = []
    for _ in range(forecast_steps):
        forecast = model.predict(scaled_data)
        forecasts.append(forecast[0, 0])
        scaled_data = np.roll(scaled_data, -1, axis=1)
        scaled_data[0, -1, :] = np.append(scaled_data[0, -1, 1:], forecast)
    
    # Unscale the forecasts
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_unscaled = close_scaler.inverse_transform(forecasts)
    
    # Prepare historical prices
    historical_prices = live_data['Close'].reset_index().rename(columns={'Date': 'date', 'Close': 'close'}).to_dict('records')

    return current_price, forecasts_unscaled.flatten().tolist(), historical_prices, accuracy

def calculate_accuracy(live_data, scaled_data, close_scaler):
    # Use the last 30 days for accuracy calculation
    test_data = scaled_data[-30:]
    actual_prices = live_data['Close'].iloc[-30:].values

    predictions = []
    for i in range(len(test_data) - 10):
        input_data = test_data[i:i+10]
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data)
        predictions.append(prediction[0, 0])

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_unscaled = close_scaler.inverse_transform(predictions)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(actual_prices[10:], predictions_unscaled)
    
    # Convert MAPE to accuracy percentage
    accuracy = (1 - mape) * 100

    return round(accuracy, 2)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    instrument = data['instrument']
    timeframe = data['timeframe']
    current_price, forecast_unscaled, historical_prices, accuracy = get_live_data_and_predict(instrument, timeframe)
    
    return jsonify({
        'current_price': float(current_price),
        'prediction': forecast_unscaled,
        'historical_prices': historical_prices,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)