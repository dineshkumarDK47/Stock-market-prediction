

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

    # Get the last date from the entire dataset for future predictions
    last_date = df['Date'].max()

    # Number of days for future predictions
    num_days = 10

    # Generate future dates for prediction starting from the day after the last date
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=num_days, freq='B')

    # Print the last date and future dates for debugging
    print(f"Last date: {last_date}")
    print(f"Future dates: {future_dates}")

    # Convert future dates to the format used in the plot
    predicted_dates = future_dates.strftime('%Y-%m-%d').tolist()

    # Convert predicted dates to datetime objects for model prediction
    future_dates_int = np.array((future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).reshape(-1, 1)

    # Make predictions for future prices
    future_predictions = model.predict(future_dates_int)

    # Convert predicted dates to string format for JavaScript
    predicted_dates_js = future_dates.strftime('%a, %d %b %Y %H:%M:%S GMT').tolist()

    # Save predictions to a dictionary
    predictions_dict = {'dates': predicted_dates_js, 'prices': future_predictions.tolist()}

    return predictions_dict

