# app.py
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import os
from lstm_model import StockLSTMModel

app = Flask(__name__)

excel_file_path = 'all_stock_data.xlsx'
lstm_model = None  # Initialize as None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    global lstm_model

    # Load the dataset
    df = pd.read_excel(excel_file_path)
    df = df[['Date', 'Close']]

    # Instantiate the LSTM model if not already created
    if lstm_model is None:
        lstm_model = StockLSTMModel(df)
        X_train, y_train = lstm_model.prepare_data(time_step=8)  # Adjust time_step as needed
        lstm_model.train_model(X_train, y_train, epochs=50, batch_size=64)

    return render_template('predict.html', stock_data=df)

# app.py

# ... (previous code)

@app.route('/process_predict', methods=['POST'])
def process_predict():
    global lstm_model
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Fetch data from yfinance
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    # Reset the index to make 'Date' a column
    stock_data.reset_index(inplace=True)

    # Add a new column for stock name
    stock_data['Stock Name'] = stock_name

    # Rearrange columns to have 'Date' first and 'Stock Name' last
    columns_order = ['Date'] + [col for col in stock_data.columns if col != 'Date' and col != 'Stock Name'] + ['Stock Name']
    stock_data = stock_data[columns_order]

    # Create or load the Excel file
    try:
        # If file exists, remove it
        os.remove(excel_file_path)
    except FileNotFoundError:
        pass

    # Save new data to Excel
    stock_data.to_excel(excel_file_path, index=False)

    # Instantiate the LSTM model if not already created
    if lstm_model is None:
        lstm_model = StockLSTMModel(stock_data)
        X_train, y_train = lstm_model.prepare_data(time_step=8)  # Adjust time_step as needed
        lstm_model.train_model(X_train, y_train, epochs=50, batch_size=64)

    # Additional logic to prepare data for prediction
    X_test, _ = lstm_model.prepare_data(time_step=8)  # Adjust time_step as needed

    # Make predictions using the LSTM model for the next 10 days
    future_predictions = lstm_model.predict_future(X_test, num_days=10)

    print("Future Predictions in app.py:", future_predictions)  # Add this line

    return render_template('result.html', excel_file=excel_file_path, stock_data=stock_data, future_predictions=future_predictions)

# ... (remaining code)

if __name__ == '__main__':
    app.run(debug=True)
