# 📈 Stock Market Prediction with LSTM

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock prices using historical data scraped via `yfinance`. Predictions and visualizations are shown using an interactive Streamlit dashboard.

---

## 🔧 Tech Stack

- **Languages:** Python  
- **Libraries & Tools:**  
  - `TensorFlow`, `Keras` – for LSTM model
  - `yfinance` – for stock data fetching
  - `Pandas`, `NumPy`, `scikit-learn` – for data preprocessing
  - `Matplotlib` – for plotting
  - `Streamlit` – for web app interface

---

## 📂 Project Files

Stock/
├── app.py # Streamlit app to run stock predictor
├── Stock Predictions Model.keras # Pre-trained LSTM model
├── Untitled.ipynb # Notebook version (optional)
└── README.md # Project documentation

---

## 🚀 How to Run This Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/stock-market-prediction.git
   cd stock-market-prediction

2. Install the required libraries
  - pip install -r requirements.txt

3. Run the Streamlit app
  - streamlit run app.py

4. Now on the Streamlit interface,
  Enter a stock symbol (like GOOG, AAPL, TSLA) in the input field to see the predictions.

📊 Features

Fetches historical stock data using yfinance

Scales and prepares data for LSTM prediction

Compares actual vs predicted prices

Calculates model accuracy (MAPE)

Shows Moving Averages: MA50, MA100, MA200

Streamlit-powered visualizations

Note: 
Ensure the .keras model file is placed correctly in the app directory or update the path in app.py



📬 Contact
Developer: Chaitanya Naik
GitHub: github.com/ChaitanyaNaik2026

---

