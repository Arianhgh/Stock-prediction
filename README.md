<h1>Predicting Stock Price Trends with LSTM</h1>

Stock4.py demonstrates how to use Long Short-Term Memory (LSTM) deep learning model with Keras to predict the trend (upward or downward) of a stock's price based on historical data.

Dataset and Features

The script uses historical price data for Apple Inc. (ticker symbol: AAPL) obtained from Yahoo Finance for the date range from January 1, 2022, to January 1, 2023. The dataset contains hourly stock price data and includes the 'Close' price, which will be the primary feature used for prediction.

Data Preprocessing

To train the LSTM model, the script calculates the average price change over the next 10 hours for each data point and labels the data as 'Upward' if the average price change is positive, and 'Downward' if it is negative. The data is then split into training and testing sets.

LSTM Model Architecture

The LSTM model is designed with three LSTM layers and a dense output layer. The architecture is as follows:

    LSTM layer with 100 units and return sequences (input shape: 20 time steps, 1 feature)
    Dropout layer to prevent overfitting
    LSTM layer with 100 units and return sequences
    Dropout layer
    LSTM layer with 50 units
    Dense output layer with sigmoid activation function (for binary classification)

Model Training and Prediction

The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is then trained on the training data for 50 epochs with a batch size of 16. After training, the model is used to make predictions on the test data.
Evaluation

The predictions are compared with the actual labels from the test data. The accuracy score is calculated using the scikit-learn library to evaluate the model's performance in predicting the stock price trends.
The best accuracy I have got till now is **0.7**

<h1>Stock Price Prediction with LSTM</h1>

Stock2.py demonstrates how to use Long Short-Term Memory (LSTM) neural networks with Keras to predict stock prices. The script uses historical stock price data for McDonald's Corporation (MCD) downloaded from Yahoo Finance.
Data Preparation

The script loads the stock price data and splits it into training and validation sets. The data is then scaled using MinMaxScaler to normalize it between 0 and 1, which is necessary for training neural networks.

For each data point in the training set, a window of 60 past stock prices is used as input (x_train), and the corresponding next stock price is the target (y_train). The same process is applied to the validation set (x_valid).
LSTM Model Architecture

The LSTM model consists of two LSTM layers with 50 units each. The first LSTM layer has the return_sequences=True argument, as it is followed by another LSTM layer. The final dense layer is used for the prediction.
Model Training

The model is compiled using mean squared error as the loss function and the Adam optimizer. It is then trained on the training data for 50 epochs with a batch size of 32.
Prediction and Visualization

The trained model is used to predict stock prices for the validation set. The predictions are then transformed back to their original scale using the inverse transform of the MinMaxScaler.

Finally, the actual and predicted stock prices are plotted using matplotlib, and the plot is saved as 'stock{2-4}.png' (results for different companies).
