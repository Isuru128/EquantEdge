# EquantEdge

High-frequency, multi-asset quantitative trading engine that connects to MetaTrader 5, detects Liquidity Sweep price-action reversals on M1 timeframe, filters setups with an XGBoost ML classifier, and executes automated trades with dynamic ATR stop-loss and real-time breakeven trailing.

---

## 📁 Project Structure

```
EquantEdge/
├── assets/                 # Brand UI assets (app icon .ico & .png)
├── data/                   # Cached MT5 historical datasets (gitignored)
├── models/                 # Serialized XGBoost model artifacts & metadata
├── src/
│   ├── connect.py          # MT5 terminal connection & chunked historical candle loader
│   ├── dashboard.py        # Multi-asset Tkinter desktop trading terminal UI
│   ├── db.py               # Trade logging & Supabase analytics integration
│   ├── execution.py        # Order placement, multi-mode filling retry, & SL/TP modification
│   ├── ml_dataset.py       # Multi-asset dataset builder & breakeven-aware triple-barrier labeling
│   ├── ml_features.py      # Technical indicator engine & 19-feature ML extractor
│   ├── patterns.py         # Liquidity sweep, inside bar, pin bar, & candlestick pattern logic
│   ├── risk.py             # Position sizing & risk manager
│   ├── run_live.py         # Multi-symbol real-time live trading engine (M1)
│   ├── strategy.py         # Liquidity sweep strategy & ML inference pipeline
│   └── train_xgb.py        # XGBoost classifier trainer with CV & threshold evaluator
├── tests/
│   ├── test_ml_pipeline.py # Unit tests for ML feature extraction & model inference
│   └── test_strategy.py    # Unit tests for liquidity sweep buy/sell signal detection
├── EquantEdge.bat          # 1-click Windows desktop batch launcher
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation & setup guide
```

---

## 🚀 Quick Start (Local)

1. **Prerequisites**: [MetaTrader 5 Desktop](https://www.metatrader5.com/) logged into your demo account & Python 3.10+.
2. **Setup**:
   ```bash
   git clone https://github.com/Isuru128/EquantEdge.git
   cd EquantEdge
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Train Model on Latest MT5 Data (1-Minute M1)**:
   ```bash
   python -m src.ml_dataset --symbols GBPUSD,XAUUSD,EURUSD --tf 1 --candles 50000
   python -m src.train_xgb --threshold 0.55
   ```
4. **Launch Live Execution Bot**:
   ```bash
   python -m src.run_live
   ```
5. **Launch Visual Desktop Terminal**:
   ```bash
   python -m src.dashboard --live
   ```

---

## ☁️ Steps to Run on AWS (24/7 Cloud Execution)

> **Note**: MetaTrader 5 Python API requires a Windows environment. Deploy on an **AWS EC2 Windows Server** instance.

1. **Launch EC2 Instance**:
   - Go to **AWS Management Console** → **EC2** → **Launch Instance**.
   - **AMI**: Microsoft Windows Server 2022 Base (or 2025).
   - **Instance Type**: `t3.medium` or `t3.large` (2 vCPUs, 4GB+ RAM recommended).
   - **Storage**: 30+ GB gp3.
   - **Key Pair**: Download `.pem` key for Remote Desktop (RDP) login.

2. **Connect via RDP**:
   - In AWS EC2 Console, select your instance → **Connect** → **RDP Client**.
   - Get administrator password using your `.pem` key, then connect via **Remote Desktop Connection**.

3. **Install Software on the Windows EC2 Server**:
   - Download & install [MetaTrader 5](https://www.metatrader5.com/). Log in to your broker account and leave the MT5 terminal running.
   - Download & install [Python 3.11](https://www.python.org/downloads/) (check `"Add python.exe to PATH"` during installation).
   - Install [Git for Windows](https://git-scm.com/download/win).

4. **Clone Repository & Setup Environment**:
   - Open PowerShell on the EC2 server:
   ```powershell
   git clone https://github.com/Isuru128/EquantEdge.git
   cd EquantEdge
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Train Model & Run 24/7 Live Trader**:
   ```powershell
   # Generate latest training dataset
   python -m src.ml_dataset --symbols GBPUSD,XAUUSD,EURUSD --tf 1 --candles 50000

   # Train the XGBoost model
   python -m src.train_xgb --threshold 0.55

   # Start live automated execution loop
   python -m src.run_live
   ```

6. **(Optional) Run as a Persistent Windows Background Service**:
   - Use Windows Task Scheduler or NSSM (Non-Sucking Service Manager) to automatically start `python -m src.run_live` upon server boot.
