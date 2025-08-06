import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

st.header('Stock Market Predictor')

# User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2021-01-01'
end = '2025-07-14'

# Load the LSTM Model with Error Handling
try:
    model = load_model(r'C:\Users\LUCKY NAIK\Desktop\Stock\Stock Predictions Model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Fetch Stock Data
data = yf.download(stock, start=start, end=end)

# Check if data is empty
if data.empty:
    st.error("No stock data available. Check the stock symbol or try a different one.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

# Split Data
split_idx = int(len(data) * 0.80)
data_train = data[['Close']].iloc[:split_idx]
data_test = data[['Close']].iloc[split_idx:]

# Ensure there is enough data to proceed
if len(data_train) == 0 or len(data_test) == 0:
    st.error("Not enough data for training or testing. Try a different stock symbol or date range.")
    st.stop()

# Apply MinMax Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)
data_test_scale = scaler.transform(data_test)

# Ensure data_test_scale has at least 101 entries
if data_test_scale.shape[0] <= 100:
    st.error("Not enough test data points for predictions.")
    st.stop()

# Prepare Input Data for Model
x, y = [], []
for i in range(100, len(data_test_scale)):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make Predictions
predict = model.predict(x)

# Rescale back to original price range
if scaler.scale_ is not None:
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale
else:
    st.error("Error in scaling transformation.")
    st.stop()

# Prepare Data for Display
predicted_dates = data.index[split_idx:][100:]
predicted_actual_df = pd.DataFrame({
    'Date': predicted_dates,
    'Predicted Price': predict.flatten(),
    'Actual Price': y.flatten()
})

st.subheader('Predicted vs Actual Prices')
st.write(predicted_actual_df)

# Calculate Model Accuracy (MAPE)
mape = mean_absolute_percentage_error(y, predict)
accuracy = 100 - (mape * 100)

st.subheader('Model Accuracy')
if not np.isinf(accuracy):
    st.write(f"Model Accuracy: {accuracy:.2f}%")
else:
    st.write("Model Accuracy: N/A (MAPE is infinite)")

# Moving Averages and Plots
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()

fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data['Close'], 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Price vs MA50')
st.pyplot(fig1)

st.subheader('Price vs MA100')
ma_100_days = data['Close'].rolling(100).mean()

fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data['Close'], 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Price vs MA100')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()

fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Price vs MA100 vs MA200')
st.pyplot(fig3)

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predicted_dates, predict.flatten(), 'r', label='Predicted Price')
plt.plot(predicted_dates, y.flatten(), 'g', label='Actual Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Original Price vs Predicted Price')
st.pyplot(fig4)
