# app.py

import matplotlib
# --- THE FIX: Set the backend BEFORE importing pyplot ---
# This tells matplotlib to not use any GUI windows, which is essential for web servers.
matplotlib.use('Agg')

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt # Now this import is safe
import warnings
import os
from newsapi import NewsApiClient
import nltk
from flask import Flask, request, jsonify, render_template
import time

# --- A. Initial Setup: NLTK Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- B. Suppress TensorFlow logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- 1. Flask App Initialization ---
app = Flask(__name__)
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# --- 2. Settings (Unchanged) ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '993551ddb5b64cdbbdc0df8ce97c9bda')
TIME_STEPS = 60
PREDICTION_DAYS = 5
HISTORICAL_DAYS_TO_PLOT = 200
TREND_THRESHOLD_PERCENT = 2.0
DIVERGENCE_LOOKBACK = 30
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


# --- 3. All Helper Functions (Unchanged) ---
def calculate_pivot_points(data):
    last_day = data.iloc[-1]
    high, low, close = last_day['High'], last_day['Low'], last_day['Close']
    p = (high + low + close) / 3
    s1 = (2 * p) - high; r1 = (2 * p) - low
    s2 = p - (high - low); r2 = p + (high - low)
    return {'P': p, 'S1': s1, 'R1': r1, 'S2': s2, 'R2': r2}

def detect_reversal_signal(data, lookback_period):
    recent_data = data.tail(lookback_period).copy()
    if recent_data['Close'].iloc[-1] == recent_data['Close'].max():
        high_price_idx = recent_data['Close'].idxmax()
        second_high_price_series = recent_data['Close'].drop(high_price_idx).nlargest(1)
        if not second_high_price_series.empty and recent_data.loc[high_price_idx, 'RSI_14'] < recent_data.loc[second_high_price_series.index[0], 'RSI_14']:
            return "Bearish Divergence"
    if recent_data['Close'].iloc[-1] == recent_data['Close'].min():
        low_price_idx = recent_data['Close'].idxmin()
        second_low_price_series = recent_data['Close'].drop(low_price_idx).nsmallest(1)
        if not second_low_price_series.empty and recent_data.loc[low_price_idx, 'RSI_14'] > recent_data.loc[second_low_price_series.index[0], 'RSI_14']:
            return "Bullish Divergence"
    return "None"

def get_sentiment_analysis(ticker):
    if not NEWS_API_KEY: return "Neutral"
    try:
        query = ticker.replace('.NS', '')
        all_articles = newsapi.get_everything(q=query, from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'), language='en', sort_by='relevancy', page_size=20)
        if not all_articles['articles']: return "Neutral"
        analyzer = SentimentIntensityAnalyzer()
        compound_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in all_articles['articles'] if article['title']]
        if not compound_scores: return "Neutral"
        avg_score = sum(compound_scores) / len(compound_scores)
        if avg_score >= 0.05: return "Positive"
        elif avg_score <= -0.05: return "Negative"
        else: return "Neutral"
    except:
        return "Neutral"

# --- 4. The Core Processing Function (Unchanged) ---
def process_stock_data(ticker):
    try:
        data = yf.download(ticker, start='2015-01-01', end=date.today().strftime('%Y-%m-%d'), progress=False)
        if data.empty: return {"error": f"No data found for {ticker}."}
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

        pivot_levels = calculate_pivot_points(data)
        data.ta.rsi(append=True); data.ta.macd(append=True); data.ta.bbands(length=20, std=2.0, append=True)
        data.dropna(inplace=True)
        last_actual_close = data['Close'].iloc[-1]
        reversal_signal = detect_reversal_signal(data, DIVERGENCE_LOOKBACK)
        
        data.reset_index(inplace=True)
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(data[feature_columns])
        high_col_idx, low_col_idx, close_col_idx = [feature_columns.index(c) for c in ['High', 'Low', 'Close']]
        X_train, y_train = [], []
        for i in range(TIME_STEPS, len(scaled_features)):
            X_train.append(scaled_features[i-TIME_STEPS:i, :]); y_train.append(scaled_features[i, [high_col_idx, low_col_idx, close_col_idx]])
        X_train, y_train = np.array(X_train), np.array(y_train)
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), Dropout(0.2),
            LSTM(units=100, return_sequences=False), Dropout(0.2), Dense(units=50), Dense(units=3)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=0)
        last_sequence = scaled_features[-TIME_STEPS:]
        prediction_input = np.reshape(last_sequence, (1, TIME_STEPS, X_train.shape[2]))
        predicted_scaled_values = []
        for _ in range(PREDICTION_DAYS):
            prediction_scaled = model.predict(prediction_input, verbose=0)[0]
            predicted_scaled_values.append(prediction_scaled)
            new_row = prediction_input[0, -1, :].copy(); new_row[high_col_idx], new_row[low_col_idx], new_row[close_col_idx] = prediction_scaled[0], prediction_scaled[1], prediction_scaled[2]
            prediction_input = np.reshape(np.vstack([prediction_input[0, 1:, :], new_row]), (1, TIME_STEPS, X_train.shape[2]))
        predicted_scaled_values = np.array(predicted_scaled_values)
        dummy_array = np.zeros((len(predicted_scaled_values), len(feature_columns))); dummy_array[:, [high_col_idx, low_col_idx, close_col_idx]] = predicted_scaled_values
        inversed_predictions = scaler.inverse_transform(dummy_array)
        predicted_highs, predicted_lows, predicted_closes = inversed_predictions[:, high_col_idx], inversed_predictions[:, low_col_idx], inversed_predictions[:, close_col_idx]

        final_predicted_close = predicted_closes[-1]
        percent_change = ((final_predicted_close - last_actual_close) / last_actual_close) * 100
        
        if percent_change > TREND_THRESHOLD_PERCENT: trend = "Upward"
        elif percent_change < -TREND_THRESHOLD_PERCENT: trend = "Downward"
        else: trend = "Sideways"
        
        sentiment_category = get_sentiment_analysis(ticker)

        if reversal_signal == "Bearish Divergence": recommendation = "Hold (Bearish Divergence)"
        elif reversal_signal == "Bullish Divergence": recommendation = "Hold (Bullish Divergence)"
        elif trend == "Upward" and sentiment_category == "Positive": recommendation = "Strong Buy"
        elif trend == "Upward": recommendation = "Buy"
        elif trend == "Downward" and sentiment_category == "Negative": recommendation = "Strong Sell"
        elif trend == "Downward": recommendation = "Sell"
        else: recommendation = "Hold"
        
        prediction_dates = [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start=pd.to_datetime(data['Date'].iloc[-1]) + timedelta(days=1), periods=PREDICTION_DAYS)]
        prediction_df = pd.DataFrame({
            'Date': prediction_dates, 'High': [f'{p:.2f}' for p in predicted_highs], 'Low': [f'{p:.2f}' for p in predicted_lows], 'Close': [f'{p:.2f}' for p in predicted_closes]
        })

        plot_path = f'static/images/{ticker}_plot.png'
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(10, 6))
        historical_plot_data = data.tail(HISTORICAL_DAYS_TO_PLOT)
        plt.plot(historical_plot_data['Date'], historical_plot_data['Close'], label='Historical Close', color='royalblue')
        plt.plot(pd.to_datetime(prediction_df['Date']), predicted_closes, label='Predicted Close', linestyle='--', color='orangered', marker='o')
        plt.fill_between(pd.to_datetime(prediction_df['Date']), predicted_highs, predicted_lows, color='orangered', alpha=0.2, label='Predicted High-Low Range')
        title_text = f'{ticker} Prediction\nTrend: {trend} | Sentiment: {sentiment_category} | Reversal: {reversal_signal}\nRecommendation: {recommendation}'
        plt.title(title_text, fontsize=14); plt.xlabel('Date'); plt.ylabel('Price (INR)')
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(plot_path); plt.close()

        return {
            "ticker": ticker, "prediction_table": prediction_df.to_dict(orient='records'),
            "summary": {"Trend": trend, "Sentiment": sentiment_category, "Reversal Signal": reversal_signal, "Recommendation": recommendation},
            "pivot_points": {k: f'{v:.2f}' for k, v in pivot_levels.items()},
            "plot_url": f'/{plot_path}?t={time.time()}'
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- 5. Flask API Routes (Unchanged) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    
    results = process_stock_data(ticker.upper())
    return jsonify(results)

# --- 6. Run the App (Unchanged) ---
if __name__ == '__main__':
    app.run(debug=True)
