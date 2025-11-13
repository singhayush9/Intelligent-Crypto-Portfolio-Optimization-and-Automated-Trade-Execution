# Intelligent-Crypto-Portfolio-Optimization-and-Automated-Trade-Execution

An end-to-end AI-driven **algorithmic trading platform** that leverages **Deep Reinforcement Learning (DRL)** to automatically learn and optimize trading strategies.  
This project integrates a **Python-based backend** for model training and a **React + Vite frontend** for visual analytics and real-time trade visualization.

---

## 🚀 Features

- **DRL-powered trading bot** using PPO and custom reward functions  
- **Dynamic indicator selection** for technical analysis  
- **Custom trading environment** compatible with OpenAI Gym  
- **Performance visualization** and trade signal rendering  
- **Interactive frontend dashboard** built with React and Vite  
- **Support for live and backtesting modes**  
- **Market metric analysis** for volatility and trend strength  

---

## 🧩 Project Structure

```
Algortithmic_trading_using_DRL-main/
│
├── app5.py                   # Main entry point for training and running the bot
├── indicators.py             # Technical indicator calculations
├── market_metrics.py         # Market and volatility analysis functions
├── render.py                 # Visualization utilities for trading signals
├── ppo_trading_bot_enhanced.zip  # Pretrained PPO model
│
├── frontend/                 # React + Vite frontend dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/ (UI components & pages)
│
└── README.md                 # (Frontend documentation)
```

---

## ⚙️ Installation & Setup

### 🐍 Backend Setup (Python)
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Algortithmic_trading_using_DRL-main.git
   cd Algortithmic_trading_using_DRL-main
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python app5.py
   ```

---

### 💻 Frontend Setup (React + Vite)

1. Move to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run development server:
   ```bash
   npm run dev
   ```

4. The frontend will start at:
   ```
   http://localhost:5173/
   ```

---

## 🧠 Model Details

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Framework:** Stable Baselines3
- **Reward Function:** Combines profit ratio, Sharpe ratio, and volatility penalty
- **Indicators Used:** Moving Averages, RSI, MACD, Bollinger Bands, ATR
- **Training Data:** Historical market data (Yahoo Finance)

---

## 📊 Outputs & Visualization

- Equity curve and drawdown visualization  
- Buy/Sell signal rendering  
- Performance metrics (Cumulative Return, Sharpe Ratio, Max Drawdown)  
- Frontend dashboard for real-time strategy performance  

---

## 🔒 Ethical & Safety Considerations

- The model is designed **for educational and research purposes** only.  
- No guarantees of profitability — use simulated or paper trading environments.  
- Data privacy and API keys should be kept secure and **not committed to GitHub**.  

---

## 🧰 Technologies Used

| Component | Technology |
|------------|-------------|
| **Backend** | Python, Stable Baselines3, Pandas, NumPy, Matplotlib |
| **Frontend** | React, TypeScript, Vite, ShadCN/UI |
| **Visualization** | Plotly / Matplotlib |
| **Deployment (optional)** | FastAPI, Streamlit, or Flask |

---
