
import pandas as pd
import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
from IPython.display import display
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# Step 1: Get historical price data for Apple from Yahoo Finance
from sklearn.preprocessing import MinMaxScaler

ticker = "AAPL"
start_date = "2022-01-01"
end_date = "2023-01-01"
apple_data = yf.download(ticker, start=start_date, end=end_date, interval="1h")

# Step 2: Calculate the average price change over the next 10 hours and label the data
lookback_hours = 30
prediction_hours = 10

def label_trend(avg_price_change):
    if avg_price_change > 0:
        return 1
    else:
        return 0

# Calculate the average price change for the next 10 hours
apple_data['Next_10h_Avg'] = apple_data['Close'].pct_change(prediction_hours).rolling(window=prediction_hours).mean()

# Shift the Next_10h_Avg values back to align them with the current data points
apple_data['Next_10h_Avg'] = apple_data['Next_10h_Avg'].shift(-prediction_hours)

# Label the data as Upward or Downward based on the average price change
apple_data['Label'] = apple_data['Next_10h_Avg'].apply(label_trend)

# Drop rows with NaN values resulting from the rolling calculations
apple_data.dropna(inplace=True)

# Print the datas first 100 rows
print(apple_data[['Close',  'Next_10h_Avg', 'Label']])

# Step 3: Split the data into training and testing sets
# Split the data into training and testing sets
train_data = apple_data[:int(apple_data.shape[0] * 0.8)]
test_data = apple_data[int(apple_data.shape[0] * 0.8):]

x_train = train_data[['Close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)

xtrain, ytrain = [], []

for i in range(20, x_train.shape[0]):
    xtrain.append(x_train[i - 20:i])
    #append the appropriate label
    ytrain.append(train_data['Label'].iloc[i])

xtrain, ytrain = np.array(xtrain), np.array(ytrain)
# Step 4: Data Preprocessing
# Reshape the data to fit the LSTM model's input shape
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

# Step 5: Model Architecture
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

# Step 6: Model Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=50, batch_size=16)

# Step 7: Prediction
xtest = test_data[['Close']]
xtest = scaler.transform(xtest)

# Convert xtest into intervals of 60
sequences = []
for i in range(20, len(xtest)):
    sequences.append(xtest[i - 20:i, 0])

# Reshape the sequences to fit the LSTM model's input shape
xtest_sequences = np.array(sequences)
xtest_sequences = np.reshape(xtest_sequences, (xtest_sequences.shape[0], xtest_sequences.shape[1], 1))

# Predict using the trained model
y_pred = model.predict(xtest_sequences)
y_pred_labels = [1 if pred > 0.5 else 0 for pred in y_pred]

# Compare the predicted labels with the actual labels
y_test = test_data['Label'].values.tolist()
print(y_pred_labels), print(y_test[20:])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy score: ", accuracy_score(y_test[20:], y_pred_labels))



