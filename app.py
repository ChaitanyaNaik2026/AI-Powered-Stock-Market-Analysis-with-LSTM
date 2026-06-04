import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.header('📈 Stock Market Predictor')

# ── User Inputs ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    stock = st.text_input('Enter Stock Symbol', 'GOOG')
with col2:
    forecast_days = st.slider('Forecast Days', min_value=7, max_value=60, value=30)

start = '2021-01-01'
end = datetime.today().strftime('%Y-%m-%d')   # ✅ CHANGE 1: always use today's date

# ── Load Model ───────────────────────────────────────────────────────────────
try:
    model = load_model(r'C:\Users\LUCKY NAIK\Desktop\Stock\Untitled.ipynb')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Fetch Stock Data ──────────────────────────────────────────────────────────
data = yf.download(stock, start=start, end=end)

if data.empty:
    st.error("No stock data available. Check the stock symbol or try a different one.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

# ── Train / Test Split ────────────────────────────────────────────────────────
split_idx = int(len(data) * 0.80)
data_train = data[['Close']].iloc[:split_idx]
data_test  = data[['Close']].iloc[split_idx:]

if len(data_train) == 0 or len(data_test) == 0:
    st.error("Not enough data for training or testing.")
    st.stop()

# ── Scaling ───────────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)
data_test_scale  = scaler.transform(data_test)

if data_test_scale.shape[0] <= 100:
    st.error("Not enough test data points for predictions.")
    st.stop()

# ── Prepare Sequences ─────────────────────────────────────────────────────────
x, y = [], []
for i in range(100, len(data_test_scale)):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# ── Backtest Predictions ──────────────────────────────────────────────────────
predict = model.predict(x)

scale = 1 / scaler.scale_[0]
predict_rescaled = predict.flatten() * scale
y_rescaled       = y.flatten() * scale

predicted_dates = data.index[split_idx:][100:]
predicted_actual_df = pd.DataFrame({
    'Date':            predicted_dates,
    'Predicted Price': predict_rescaled,
    'Actual Price':    y_rescaled,
})

st.subheader('Predicted vs Actual Prices')
st.write(predicted_actual_df)

# ── Accuracy ──────────────────────────────────────────────────────────────────
mape     = mean_absolute_percentage_error(y_rescaled, predict_rescaled)
accuracy = 100 - (mape * 100)

st.subheader('Model Accuracy')
if not np.isinf(accuracy):
    st.metric("Model Accuracy (100 - MAPE)", f"{accuracy:.2f}%")
else:
    st.write("Model Accuracy: N/A (MAPE is infinite)")

# ── CHANGE 2: Combined Moving Averages Chart ──────────────────────────────────
st.subheader('Price vs Moving Averages (MA50 / MA100 / MA200)')

ma_50  = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

fig_ma, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data['Close'], color='#2ecc71', linewidth=1.2, label='Close Price', alpha=0.9)
ax.plot(data.index, ma_50,        color='#f39c12', linewidth=1.2, label='MA 50',  linestyle='--')
ax.plot(data.index, ma_100,       color='#3498db', linewidth=1.2, label='MA 100', linestyle='--')
ax.plot(data.index, ma_200,       color='#e74c3c', linewidth=1.2, label='MA 200', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f'{stock} — Price vs Moving Averages')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig_ma.autofmt_xdate()
st.pyplot(fig_ma)

# ── Backtest Plot ─────────────────────────────────────────────────────────────
st.subheader('Original Price vs Predicted Price')

fig_bt, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(predicted_dates, predict_rescaled, color='#e74c3c', label='Predicted Price', linewidth=1.5)
ax2.plot(predicted_dates, y_rescaled,       color='#2ecc71', label='Actual Price',    linewidth=1.5)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (USD)')
ax2.set_title(f'{stock} — Actual vs Predicted')
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig_bt.autofmt_xdate()
st.pyplot(fig_bt)

# ── CHANGE 3: 30-Day Future Forecast ─────────────────────────────────────────
st.subheader(f'🔮 {forecast_days}-Day Future Price Forecast')

# Seed the forecast with the last 100 scaled close prices from the full dataset
last_100_scaled = scaler.transform(data[['Close']].iloc[-100:])
forecast_input  = last_100_scaled.reshape(1, 100, 1).copy()

forecast_scaled = []
for _ in range(forecast_days):
    next_pred = model.predict(forecast_input, verbose=0)[0, 0]
    forecast_scaled.append(next_pred)
    # Slide the window forward
    forecast_input = np.roll(forecast_input, -1, axis=1)
    forecast_input[0, -1, 0] = next_pred

forecast_prices = np.array(forecast_scaled).reshape(-1, 1) * scale

last_date      = data.index[-1]
forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
# Skip weekends for cleaner display
forecast_dates = [d for d in forecast_dates if d.weekday() < 5][:forecast_days]
forecast_prices = forecast_prices[:len(forecast_dates)]

forecast_df = pd.DataFrame({
    'Date':             forecast_dates,
    'Forecasted Price': forecast_prices.flatten(),
})
forecast_df.set_index('Date', inplace=True)

fig_fc, ax3 = plt.subplots(figsize=(12, 5))
# Show the last 60 days of actuals for context
hist_window = data['Close'].iloc[-60:]
ax3.plot(hist_window.index, hist_window.values,    color='#2ecc71', label='Historical Close', linewidth=1.5)
ax3.plot(forecast_df.index, forecast_df['Forecasted Price'],
         color='#9b59b6', label='Forecast', linewidth=2, linestyle='--', marker='o', markersize=3)
ax3.axvline(x=last_date, color='gray', linestyle=':', linewidth=1, label='Today')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price (USD)')
ax3.set_title(f'{stock} — {forecast_days}-Day Price Forecast')
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig_fc.autofmt_xdate()
st.pyplot(fig_fc)

st.write(forecast_df)