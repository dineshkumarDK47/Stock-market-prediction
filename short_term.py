# linear_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


def train_linear_regression(df):

    # Assuming 'Date' is the index and 'Close' is the dependent variable
    X = np.array((df['Date'] - pd.Timestamp("1970-01-01"))).astype(int) // 10**9  # Convert to seconds
    X = X.reshape(-1, 1)
    y = df['Close'].values
   
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  train_size=0.8, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

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
    future_dates_int = np.array((future_dates - pd.Timestamp("1970-01-01"))).astype(int) // 10**9  # Convert to seconds
    future_dates_int = future_dates_int.reshape(-1, 1)

    # Make predictions for future prices
    future_dates_int_scaled = scaler.transform(future_dates_int.reshape(-1, 1))
    future_predictions = model.predict(future_dates_int_scaled)
    
    r_squared = r2_score(y_test, model.predict(X_test_scaled))

    # Convert predicted dates to string format for JavaScript
    predicted_dates_js = future_dates.strftime('%a, %d %b %Y %H:%M:%S GMT').tolist()

    # Save predictions to a dictionary
    predictions_dict = {'dates': predicted_dates_js, 'prices': future_predictions.tolist()}

    return predictions_dict, r_squared


