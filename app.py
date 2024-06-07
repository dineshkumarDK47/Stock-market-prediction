# app.py
from flask import Flask, render_template, request

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import yfinance as yf
import pandas as pd
import os
from short_term import train_linear_regression
from moving_average import train_moving_average

app = Flask(__name__)

excel_file_path = 'all_stock_data.xlsx'

scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('HOME.html')


@app.route('/prediction_page')
def prediction_page():
    return render_template('index.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/manual')
def manual ():
    return render_template('manual.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')


@app.route('/right_side')
def right_side():
    return render_template('index.html')

@app.route('/left_side')
def left_side():
    return render_template('index.html')

@app.route('/predict_stock', methods=['GET'])
def predict_stock():
   
    df = pd.read_excel(excel_file_path)
    required_columns = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']
    if not all(column in df.columns for column in required_columns):
        return "Error: DataFrame is missing required columns."

    # Extract features and target variable
    X = df[['Open', 'High', 'Low', 'Volume']]  # Features
    y = df['Close'].values 
    
    # Train the random forest model and get future predictions
    future_predictions, r_squared = train_linear_regression(df)
    
    actual_prices = df['Close'].tolist()
   

    return render_template('predict.html', stock_data=df, future_predictions=future_predictions, actual_prices=actual_prices, r_squared=r_squared)

@app.route('/predict_stock_comp', methods=['GET'])
def predict_stock_comp():
   
    df = pd.read_excel(excel_file_path)
    required_columns = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']
    if not all(column in df.columns for column in required_columns):
        return "Error: DataFrame is missing required columns."

    # Extract features and target variable
    X = df[['Open', 'High', 'Low', 'Volume']]  # Features
    y = df['Close'].values 
    
    # Train the random forest model and get future predictions
    future_predictions, r_squared = train_linear_regression(df)
    
    actual_prices = df['Close'].tolist()
   

    return render_template('mini2.html', stock_data=df, future_predictions=future_predictions, actual_prices=actual_prices, r_squared=r_squared)

@app.route('/predict_stock_moving_average', methods=['GET'])
def predict_stock_moving_average():
    df = pd.read_excel(excel_file_path)
    future_predictions, r_squared = train_moving_average(df)
    actual_prices = df['Close'].tolist()
    stock_name = df['Stock Name'].iloc[0]

    return render_template('predict.html', stock_data=df, future_predictions=future_predictions, actual_prices=actual_prices, stock_name=stock_name,  r_squared=r_squared)



@app.route('/process_predict', methods=['POST'])
def process_predict():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    
    stock_data.reset_index(inplace=True)

    stock_data['Stock Name'] = stock_name

    future_predictions = train_linear_regression(stock_data)

    stock_data.to_excel(excel_file_path, index=False)

    return render_template('result.html', excel_file=excel_file_path, stock_data=stock_data,
                           future_predictions=future_predictions)

@app.route('/process_predict_1', methods=['POST'])
def process_predict_1():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    
    stock_data.reset_index(inplace=True)

    stock_data['Stock Name'] = stock_name

    future_predictions = train_linear_regression(stock_data)

    stock_data.to_excel(excel_file_path, index=False)

    return render_template('mini1.html', excel_file=excel_file_path, stock_data=stock_data,
                           future_predictions=future_predictions)
if __name__ == '__main__':
    app.run(debug=True)