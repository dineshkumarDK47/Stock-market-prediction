# linear_regression_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def train_linear_regression(df):
    # Assuming 'Date' is the index and 'Close' is the dependent variable
    X = np.array((df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).reshape(-1, 1)
    y = df['Close'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Get the last date from the test set for future predictions
    last_date = df.loc[X_test[:, 0].argmax(), 'Date']

    # Number of days for future predictions
    num_days = 30

    # Generate future dates for prediction
    future_dates = pd.date_range(last_date, periods=num_days + 1, freq='B')[1:]

    # Convert future dates to the format used in the plot
    predicted_dates = future_dates.strftime('%Y-%m-%d').tolist()

    # Convert predicted dates to datetime objects for model prediction
    future_dates_int = np.array((future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).reshape(-1, 1)

    # Make predictions for future prices
    future_predictions = model.predict(future_dates_int)

    # Save predictions to a dictionary
    predictions_dict = {'dates': predicted_dates, 'prices': future_predictions.tolist()}

    return predictions_dict
