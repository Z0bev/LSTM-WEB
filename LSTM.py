import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Activation


# load the data
data = pd.read_csv('MSFT.csv', date_parser=True)
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
data['MA'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
data['BB_upper'] = bb.bollinger_hband()
data['BB_middle'] = bb.bollinger_mavg()
data['BB_lower'] = bb.bollinger_lband()
data = data[['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close', 'RSI', 'MA', 'EMA', 'BB_upper', 'BB_middle', 'BB_lower']]
data = data.set_index('Date')

# remove NaN values
data = data.dropna()

print(data)

# split the data into training and test sets
train_size = int(len(data) * 0.7)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# scale the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# prepare the data for LSTM
def prepare_data(data, window_size):
    X, Y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        Y.append(data[i, 1]) # predict the Close value
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

window_size = 2
trainX, trainY = prepare_data(train_data, window_size)
testX, testY = prepare_data(test_data, window_size)

# build the LSTM model
model = Sequential()
model.add(LSTM(64, activation='sigmoid', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='sigmoid', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='sigmoid', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# train the model
history = model.fit(trainX, trainY, epochs=2000, batch_size=100, shuffle=True, validation_data=(testX, testY), verbose=2)

# plot the training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# make predictions on the test data
predictions = model.predict(testX)
for i in range(len(predictions)):
        print('predicted=%f, actual=%f' % (predictions[i], testY[i]))

print(predictions.shape)

# plot the predicted and actual prices
plt.plot(predictions, label='predicted Close', color='red')
plt.plot(testY, label='actual Close', color='green')
plt.legend()
plt.show()

# create dummy columns for predictions
dummy_cols_pred = np.zeros((predictions.shape[0], train_data.shape[1]-1))
predictions = np.hstack((dummy_cols_pred, predictions))
unscaled_predictions = scaler.inverse_transform(predictions)[:, -1]

# create dummy columns for actual values
dummy_cols_actual = np.zeros((testY.shape[0], train_data.shape[1]-1))
actual = np.hstack((dummy_cols_actual, testY.reshape(-1, 1)))
unscaled_actual = scaler.inverse_transform(actual)[:, -1]

# plot the predicted and actual prices (unscaled)
plt.plot(unscaled_predictions, label='predicted Close', color='red')
plt.plot(unscaled_actual, label='actual Close', color='green')
plt.legend()
plt.show()

# save the model
model.save('trained_model.h13')