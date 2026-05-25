# 📈 Stock Market Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![yFinance](https://img.shields.io/badge/yFinance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)

**📄 Published Research:** AI-Powered Stock Market Analysis with LSTM (IJFMR)

---

## 🎯 Problem Statement

Stock price prediction is challenging due to market volatility, non-linear trends, and continuously changing market conditions. This project explores the use of **Long Short-Term Memory (LSTM)** neural networks to learn historical stock price patterns and generate future price forecasts.

The goal is to provide investors and learners with a data-driven approach to understanding stock market trends using deep learning techniques.

---

## 🛠️ Technical Stack

### 🤖 Machine Learning & Deep Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![LSTM](https://img.shields.io/badge/LSTM-Neural_Network-8A2BE2?style=for-the-badge&logo=pytorch&logoColor=white)

### 📊 Data Processing & Analysis
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![yFinance](https://img.shields.io/badge/yFinance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)

### 📈 Visualization & Dashboard
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### ⚙️ Development Tools
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

---

## 🧠 Project Workflow

### 1. Data Collection

* Fetches historical stock data using Yahoo Finance (`yfinance`)
* Supports multiple stock symbols such as AAPL, GOOG, TSLA, and more
* Uses stock closing prices for forecasting

### 2. Data Preprocessing

* Handles missing values
* Applies MinMax Scaling
* Creates 100-day sliding window sequences
* Splits data into training and testing datasets

### 3. Model Development

* LSTM-based deep learning architecture
* Trained on historical stock price patterns
* Generates future stock price predictions

### 4. Visualization

* Actual vs Predicted Price Comparison
* Moving Averages (MA50, MA100, MA200)
* Interactive Streamlit dashboard

---

## 📸 Dashboard Preview

> Add a screenshot after deployment

```text
images/dashboard.png
```

---

## 🏗️ System Architecture

```text
User Input (Stock Symbol)
            │
            ▼
 Historical Data (yFinance)
            │
            ▼
 Data Preprocessing
 (Scaling & Sequence Creation)
            │
            ▼
      LSTM Model
            │
            ▼
     Predictions
            │
            ▼
 Streamlit Dashboard
```

---

## ✨ Features

✅ Real-time stock data retrieval

✅ LSTM-based stock price forecasting

✅ Historical trend visualization

✅ Moving Average Analysis (MA50, MA100, MA200)

✅ Actual vs Predicted Price Comparison

✅ Interactive Streamlit Dashboard

✅ Deep Learning-powered Time-Series Forecasting

---

## 📊 Model Capabilities

* Forecasts stock prices using historical market data
* Captures short-term and medium-term trends
* Identifies price movement patterns from sequential data
* Visualizes prediction performance against real values
* Supports analysis of multiple stock symbols

---

## 📂 Project Structure

```text
stock-market-prediction/
│
├── app.py
├── Stock Predictions Model.keras
├── Untitled.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/ChaitanyaNaik2026/stock-market-prediction.git
cd stock-market-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

### Using the Application

1. Launch the Streamlit app.
2. Enter a stock symbol (AAPL, GOOG, TSLA, etc.).
3. View stock trends and moving averages.
4. Compare actual vs predicted prices.
5. Analyze future forecasts generated by the LSTM model.

---

## 📦 Requirements

```text
tensorflow>=2.10.0
keras>=2.10.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
streamlit>=1.20.0
matplotlib>=3.6.0
```

---

## 📚 What I Learned

* Building LSTM architectures for time-series forecasting
* Creating sliding-window sequences from stock market data
* Applying normalization techniques for financial datasets
* Evaluating prediction quality using forecasting metrics
* Deploying machine learning applications with Streamlit
* Understanding challenges associated with volatile financial markets

---

## 🔮 Future Improvements

🔸 Integrate news sentiment analysis

🔸 Compare LSTM with XGBoost and Prophet

🔸 Add portfolio tracking features

🔸 Implement stock recommendation insights

🔸 Deploy using FastAPI + Streamlit architecture

🔸 Add backtesting and trading simulations

---

## 👨‍💻 Developer

**Chaitanya Naik**

[![GitHub](https://img.shields.io/badge/GitHub-ChaitanyaNaik2026-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ChaitanyaNaik2026)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-naikchaitanya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/naikchaitanya)

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Stock market investments involve risk, and the predictions generated by this model should not be considered financial advice. Past performance does not guarantee future results.
