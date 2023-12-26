# app.py
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import os
from linear_regression import train_linear_regression

app = Flask(__name__)

excel_file_path = 'all_stock_data.xlsx'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    # Load the dataset
    df = pd.read_excel(excel_file_path)
    df = df[['Date', 'Close']]
    future_predictions = train_linear_regression(df)

    return render_template('predict.html', stock_data=df, future_predictions=future_predictions)

@app.route('/process_predict', methods=['POST'])
def process_predict():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Fetch data from yfinance
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    # Reset the index to make 'Date' a column
    stock_data.reset_index(inplace=True)

    # Add a new column for stock name
    stock_data['Stock Name'] = stock_name

    future_predictions = train_linear_regression(stock_data)

    # Save new data to Excel
    stock_data.to_excel(excel_file_path, index=False)

    return render_template('result.html', excel_file=excel_file_path, stock_data=stock_data,
                           future_predictions=future_predictions)

if __name__ == '__main__':
    app.run(debug=True)