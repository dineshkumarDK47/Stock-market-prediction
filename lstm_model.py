# lstm_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class StockLSTMModel:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        input_shape = (self.data.shape[1], 1)
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, time_step=1):
        data_scaled = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(data_scaled)-time_step-1):
            X.append(data_scaled[i:(i+time_step), 0])
            y.append(data_scaled[i + time_step, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y

    def train_model(self, X_train, y_train, epochs=50, batch_size=64):
        try:
            # Check the shapes and data types before training
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_train dtype:", X_train.dtype)
            print("y_train dtype:", y_train.dtype)

            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        except Exception as e:
            print("Error during training:", e)

    def predict(self, X_test):
        predicted_prices_scaled = self.model.predict(X_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices_scaled)
        return predicted_prices.flatten()

    def predict_future(self, X_test, num_days=10):
        # Use the trained model to predict future prices
        future_predictions = []

        for _ in range(num_days):
            next_prediction = self.model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))
            future_predictions.append(next_prediction[0, 0])

            # Update X_test with the latest prediction
            X_test = np.append(X_test, next_prediction[0].reshape(1, X_test.shape[1], 1), axis=0)

        # Inverse transform the predicted prices
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        print("Future Predictions:", future_predictions)  # Add this line

        return future_predictions.flatten()
    
    def evaluate_model(self, X_test, y_test):
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)

        mse = mean_squared_error(y_test, y_pred.flatten())
        return mse

    def plot_predictions(self, actual_data, predicted_data):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data, label='Actual Prices', marker='o')
        plt.plot(predicted_data, label='Predicted Prices', marker='o')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()
