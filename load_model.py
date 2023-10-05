import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

latest_data = yf.download('MSFT', period='1h')
latest_data = latest_data[['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close']]
latest_data = latest_data.iloc[-1].values.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(latest_data)


# prepare the data

window_size = 2

def prepare_data(data, window_size):
    if len(data) < window_size:
        return data
    X, Y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        Y.append(data[i, 1]) # predict the Close value
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], data.shape[1]) # reshape X to include all columns
    return X, Y

latest_data = prepare_data(latest_data, window_size)
latest_data = scaler.transform(latest_data)

# load the model
model = load_model('trained_model.h8')

# make the prediction
prediction = model.predict(latest_data)

# inverse transform the prediction
prediction = scaler.inverse_transform(prediction)
