# train_model.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date
import warnings
import os
import pickle

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Settings ---
STOCKS_TO_PROCESS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'BHEL.NS']
TIME_STEPS = 60

# --- Manual Indicator Functions (Copied from app.py) ---
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, period=20, std_dev=2):
    middle_band = data['Close'].rolling(window=period).mean()
    rolling_std = data['Close'].rolling(window=period).std()
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    return upper_band, middle_band, lower_band

def train_and_save_model(ticker):
    print(f"--- Starting training for {ticker} ---")
    try:
        # Create a dedicated folder for saved models if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

        # 1. Download and prepare data
        data = yf.download(ticker, start='2015-01-01', end=date.today().strftime('%Y-%m-%d'), progress=False)
        if data.empty:
            print(f"No data for {ticker}. Skipping.")
            return

        data['RSI_14'] = calculate_rsi(data)
        data['MACD_12_26_9'], _ = calculate_macd(data)
        data['BBU_20_2.0'], data['BBM_20_2.0'], data['BBL_20_2.0'] = calculate_bollinger_bands(data)
        data.dropna(inplace=True)

        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
        
        # 2. Scale data and SAVE THE SCALER
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(data[feature_columns])
        with open(f'saved_models/{ticker}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # 3. Prepare training sequences
        high_col_idx, low_col_idx, close_col_idx = [feature_columns.index(c) for c in ['High', 'Low', 'Close']]
        X_train, y_train = [], []
        for i in range(TIME_STEPS, len(scaled_features)):
            X_train.append(scaled_features[i-TIME_STEPS:i, :]); y_train.append(scaled_features[i, [high_col_idx, low_col_idx, close_col_idx]])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # 4. Build and train the model
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), Dropout(0.2),
            LSTM(units=100, return_sequences=False), Dropout(0.2), Dense(units=50), Dense(units=3)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"Training model for {ticker}...")
        model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=0)
        
        # 5. SAVE THE TRAINED MODEL
        model.save(f'saved_models/{ticker}_model.h5')
        print(f"--- Successfully trained and saved model for {ticker} ---")

    except Exception as e:
        print(f"!!! ERROR training {ticker}: {e} !!!")

# --- Main Execution Block ---
if __name__ == '__main__':
    for stock_ticker in STOCKS_TO_PROCESS:
        train_and_save_model(stock_ticker)
    print("\n--- Daily training complete for all stocks. ---")
