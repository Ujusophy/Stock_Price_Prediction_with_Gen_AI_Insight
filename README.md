# 📈 Stock Price Prediction with GenAI Insights

Predict and explain stock market trends by combining time-series forecasting (Prophet + LSTM) with market sentiment, economic indicators, and the power of generative AI using Groq + LLaMA 3.

---

## 🚀 Overview

This project demonstrates how to:

- Collect real-time stock and economic data
- Engineer predictive features from financial indicators and sentiment
- Forecast future stock prices using both Prophet and deep learning (LSTM via Darts)
- Backtest and evaluate model performance
- Use Groq + LLaMA 3 to **explain predictions in plain language**

---

## 🧠 Why This Project?

Forecasting stock prices isn’t just about numbers — it's about **understanding the “why” behind predictions**. By using Generative AI, we give data scientists and investors an intelligent assistant that makes models easier to trust, interpret, and refine.

---

## 🔧 Tools and Technologies

| Category              | Tools/Libs Used                                     |
|-----------------------|-----------------------------------------------------|
| **Programming**       | Python                                              |
| **Forecasting Models**| Prophet, Darts (LSTM)                               |
| **Generative AI**     | Groq API + LLaMA 3                                  |
| **LLM Integration**   | LlamaIndex, LangChain                               |
| **Data Sources**      | yfinance, FRED (via `pandas_datareader`)            |
| **Embeddings**        | sentence-transformers (`all-MiniLM-L6-v2`)         |
| **Visualization**     | Plotly                                              |
| **Data Handling**     | pandas, NumPy                                       |
| **Secrets Handling**  | dotenv                                              |

---

## 🔐 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/stock-genai-insight.git
cd stock-genai-insight
```

### 2. Create a .env file
Add your Groq API key:
```
GROQ_API_KEY=your_groq_key_here
```
💡 You can get your key from https://console.groq.com/keys

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```
jupyter notebook stock-predict.ipynb
```

---

How It Works

➤ Data Collection
- Pulls stock data via yfinance
- Fetches economic indicators (CPI, interest rates, unemployment) from FRED
- Uses Hugging Face transformers to analyze market sentiment from headlines

➤ Feature Engineering
- Calculates returns, rolling volatility
- Merges sentiment and macroeconomic context into a single dataset

➤ Forecasting
- Prophet: interpretable time-series predictions
- Darts LSTM: advanced deep learning forecasting with multivariate inputs

➤ Evaluation
- Calculates MAE (Prophet) and MAPE (Darts)
- Uses backtesting to simulate real-world performance

➤ Generative AI Insights
- Converts last 10 days of data into text-based summaries
- Embeds them using HuggingFaceEmbedding
- Uses LlamaIndex + Groq (LLaMA 3) to answer questions like:
`“Why did the model predict a drop in Apple’s stock?”`
