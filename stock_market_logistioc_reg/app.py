import yfinance as yf
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Function to preprocess the data
def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    X = df[['Open', 'High', 'Low']]  # Features
    return X

# Function to make predictions
def predict_price_direction(df):
    X = preprocess_data(df)
    return model.predict(X)

# Streamlit web application script
def main():
    st.title('Stock Price Prediction ')

    # Sidebar for user input
    st.sidebar.header('User Input')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))
    ticker = st.sidebar.text_input('Enter Ticker Symbol', 'AAPL')

    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)

    if not df.empty:
        # Make predictions
        df['Predicted_Price_Direction'] = predict_price_direction(df)

        # Plotting original closing prices and predicted price direction
        st.subheader('Original Closing Prices vs. Predicted Price Direction')
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price', color=color)
        ax1.plot(df.index, df['Close'], color=color, label='Original Closing Prices')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Price Direction', color=color)
        ax2.plot(df.index, df['Predicted_Price_Direction'], color=color, linestyle='--', label='Predicted Price Direction')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        st.pyplot(fig)

    else:
        st.write("No data available for the selected ticker and date range.")

if __name__ == '__main__':
    main()
