# app.py - LIGHTWEIGHT PREDICTION SERVER

import matplotlib
matplotlib.use('Agg')

import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import warnings
import os
from newsapi import NewsApiClient
import nltk
from flask import Flask, request, jsonify, render_template
import time
import sys
import pickle

# --- (Initial setup is the same) ---
try: nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
app = Flask(__name__)
if not os.path.exists('static/images'): os.makedirs('static/images')

# --- (Settings are the same) ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'YOUR_API_KEY_HERE')
TIME_STEPS = 60
PREDICTION_DAYS = 5
DIVERGENCE_LOOKBACK = 30
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# --- (All helper functions for analysis are the same) ---
# ... (Copy all the calculate_..., detect_..., and get_sentiment_... functions from your previous app.py here) ...
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
def calculate_pivot_points(data):
    last_day = data.iloc[-1]
    high, low, close = last_day['High'], last_day['Low'], last_day['Close']
    p = (high + low + close) / 3
    s1 = (2 * p) - high; r1 = (2 * p) - low
    s2 = p - (high - low); r2 = p + (high - low)
    return {'P': p, 'S1': s1, 'R1': r1, 'S2': s2, 'R2': r2}
def detect_reversal_signal(data, lookback_period):
    recent_data = data.tail(lookback_period).copy()
    if recent_data.empty or 'RSI_14' not in recent_data.columns: return "None"
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
    if not NEWS_API_KEY or NEWS_API_KEY == 'YOUR_API_KEY_HERE': return "Neutral"
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

# --- NEW: Lightweight Prediction Function ---
def process_stock_data(ticker):
    try:
        # 1. Check if model files exist
        model_path = f'saved_models/{ticker}_model.h5'
        scaler_path = f'saved_models/{ticker}_scaler.pkl'
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {"error": f"No pre-trained model found for {ticker}. Please run the daily training script."}
        
        # 2. LOAD the pre-trained model and scaler
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # 3. Get the latest data for prediction
        data = yf.download(ticker, start='2015-01-01', end=date.today().strftime('%Y-%m-%d'), progress=False)
        if data.empty: return {"error": f"No data found for {ticker}."}
        
        # (Analysis is the same, but we don't dropna yet)
        pivot_levels = calculate_pivot_points(data)
        data['RSI_14'] = calculate_rsi(data)
        data['MACD_12_26_9'], _ = calculate_macd(data)
        data['BBU_20_2.0'], data['BBM_20_2.0'], data['BBL_20_2.0'] = calculate_bollinger_bands(data)
        
        reversal_signal = detect_reversal_signal(data.dropna(), DIVERGENCE_LOOKBACK)
        last_actual_close = data['Close'].iloc[-1]
        data.reset_index(inplace=True)

        # 4. Prepare the last sequence for prediction
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
        last_sequence_df = data[feature_columns].tail(TIME_STEPS)
        # We must use the LOADED scaler to transform this new data
        scaled_features = scaler.transform(last_sequence_df)
        prediction_input = np.reshape(scaled_features, (1, TIME_STEPS, len(feature_columns)))
        
        # 5. PREDICT (no training!)
        predicted_scaled_values = []
        high_col_idx, low_col_idx, close_col_idx = [feature_columns.index(c) for c in ['High', 'Low', 'Close']]
        for _ in range(PREDICTION_DAYS):
            prediction_scaled = model.predict(prediction_input, verbose=0)[0]
            predicted_scaled_values.append(prediction_scaled)
            new_row = prediction_input[0, -1, :].copy(); new_row[high_col_idx], new_row[low_col_idx], new_row[close_col_idx] = prediction_scaled[0], prediction_scaled[1], prediction_scaled[2]
            prediction_input = np.reshape(np.vstack([prediction_input[0, 1:, :], new_row]), (1, TIME_STEPS, len(feature_columns)))
        
        # (The rest of the logic is identical to your last version)
        # ... (Copy the result formatting, recommendation logic, and plotting code here) ...
        predicted_scaled_values = np.array(predicted_scaled_values)
        dummy_array = np.zeros((len(predicted_scaled_values), len(feature_columns))); dummy_array[:, [high_col_idx, low_col_idx, close_col_idx]] = predicted_scaled_values
        inversed_predictions = scaler.inverse_transform(dummy_array)
        predicted_highs, predicted_lows, predicted_closes = inversed_predictions[:, high_col_idx], inversed_predictions[:, low_col_idx], inversed_predictions[:, close_col_idx]
        final_predicted_close = predicted_closes[-1]
        percent_change = ((final_predicted_close - last_actual_close) / last_actual_close) * 100
        if percent_change > 2.0: trend = "Upward"
        elif percent_change < -2.0: trend = "Downward"
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
        prediction_df = pd.DataFrame({'Date': prediction_dates, 'High': [f'{p:.2f}' for p in predicted_highs], 'Low': [f'{p:.2f}' for p in predicted_lows], 'Close': [f'{p:.2f}' for p in predicted_closes]})
        plot_path = f'static/images/{ticker}_plot.png'
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(10, 6))
        historical_plot_data = data.tail(HISTORICAL_DAYS_TO_PLOT)
        plt.plot(historical_plot_data['Date'], historical_plot_data['Close'], label='Historical Close', color='royalblue')
        plt.plot(pd.to_datetime(prediction_df['Date']), predicted_closes, label='Predicted Close', linestyle='--', color='orangered', marker='o')
        plt.fill_between(pd.to_datetime(prediction_df['Date']), predicted_highs, predicted_lows, color='orangered', alpha=0.2, label='Predicted High-Low Range')
        title_text = f'{ticker} Prediction\nTrend: {trend} | Sentiment: {sentiment_category} | Reversal: {reversal_signal}\nRecommendation: {recommendation}'
        plt.title(title_text, fontsize=14); plt.xlabel('Date'); plt.ylabel('Price (INR)')
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(plot_path); plt.close()
        return {"ticker": ticker, "prediction_table": prediction_df.to_dict(orient='records'), "summary": {"Trend": trend, "Sentiment": sentiment_category, "Reversal Signal": reversal_signal, "Recommendation": recommendation}, "pivot_points": {k: f'{v:.2f}' for k, v in pivot_levels.items()}, "plot_url": f'/{plot_path}?t={time.time()}'}
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}", file=sys.stderr)
        return {"error": f"An unexpected error occurred. Check server logs for details."}

# --- (Flask routes are the same) ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json.get('ticker')
    if not ticker: return jsonify({"error": "Ticker symbol is required."}), 400
    results = process_stock_data(ticker.upper())
    return jsonify(results)
if __name__ == '__main__': app.run(debug=False)
