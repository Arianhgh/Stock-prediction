import pandas as pd
import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
from IPython.display import display
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pd.options.mode.chained_assignment = None

# Download the data
df = yf.download(tickers=['MCD'], period='2y')

# Split the data
train_data = df[['Close']].iloc[:-200, :]
valid_data = df[['Close']].iloc[-200:, :]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
valid_data = scaler.transform(valid_data)

x_train, y_train = [], []

for i in range(60, train_data.shape[0]):
    x_train.append(train_data[i - 60: i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_valid = []

for i in range(60, valid_data.shape[0]):
    x_valid.append(valid_data[i - 60: i, 0])

x_valid = np.array(x_valid)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)

# Train the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=x_train.shape[1:]))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Generate the model predictions
y_pred = model.predict(x_valid)
y_pred = scaler.inverse_transform(y_pred)
y_pred = y_pred.flatten()

# Plot the model predictions
df.rename(columns={'Close': 'Actual'}, inplace=True)
df['Predicted'] = np.nan
df['Predicted'].iloc[-y_pred.shape[0]:] = y_pred
df[['Actual', 'Predicted']].plot(title='MCD')

display(df)
#save the plot
plt.savefig('stock4.png')
plt.show()
