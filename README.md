# LSTM-WEB
An automated trading bot using LSTM neural networks to predict market movements and execute trades via Alpaca API.

# Dependencies
- Python 3.11+
- pandas
- numpy
- tensorflow
- alpaca-trade-api
- scikit-learn
- ta (Technical Analysis)
- yfinance
- python-dotenv

# Setup
- pip install pandas numpy tensorflow alpaca-trade-api scikit-learn ta yfinance python-dotenv

# Create .env file in root directory:
- APCA_API_KEY_ID=your_api_key
- APCA_API_SECRET_KEY=your_secret_key
- APCA_BASE_URL=https://paper-api.alpaca.markets


# Features
- LSTM-based price prediction
- Technical indicator integration
- Automated trade execution
- Position management
- Risk management
- Real-time market data
- Paper trading support

# The bot uses:
- LSTM predictions
- RSI
- Moving Averages
- MACD
- Bollinger Bands
- Error Handling
- API retry mechanism
- Logging system
- Fallback to yfinance data
