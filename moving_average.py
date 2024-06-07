# moving_average.py
import pandas as pd
from sklearn.metrics import r2_score

def train_moving_average(df):
    # Assuming 'Date' is the index and 'Close' is the dependent variable
    df = df[['Date', 'Close']].set_index('Date')

    # Calculate the rolling mean with a window size of 5 (you can adjust this)
    df['moving_average'] = df['Close'].rolling(window=5).mean()

    # Get the last date from the entire dataset for future predictions
    last_date = df.index.max()

    # Number of days for future predictions
    num_days = 25

    # Generate future dates for prediction starting from the day after the last date
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=num_days, freq='B')

    # Create a DataFrame with future dates and NaN values for 'moving_average'
    future_df = pd.DataFrame({'Date': future_dates, 'moving_average': [None] * num_days}).set_index('Date')

    # Print the last date and future dates for debugging
    print(f"Last date: {last_date}")
    print(f"Future dates: {future_dates}")
    actual_values = df['Close'][-num_days:]  # Last num_days actual closing prices
    predicted_values = df['moving_average'][-num_days:]
    r_squared = r2_score(actual_values, predicted_values)

    # Save predictions to a dictionary
    predictions_dict = {'dates': future_dates.strftime('%a, %d %b %Y %H:%M:%S GMT').tolist(),
                        'prices': df['moving_average'].tolist()[-num_days:]}

    return predictions_dict, r_squared
