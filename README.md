# EquantEdge

High-frequency quantitative trading and execution engine that connects to MetaTrader 5, detects **15-Minute Fair Value Gaps (FVG)** aligned with the 15-Minute trend, executes on **1-Minute Market Structure Shifts (MSS)** with 1:2.0 Risk-to-Reward ratio, enforces strict session execution windows (excluding 14:00 - 20:00 NY time), and features real-time breakeven trailing.

---

## 📁 Project Structure

```
EquantEdge/
├── assets/                 # Brand UI assets (app icon .ico & .png)
├── data/                   # Historical datasets & schema (gitignored)
├── models/                 # Serialized ML model artifacts & metadata
├── src/
│   ├── connect.py          # MT5 terminal connection & chunked historical candle loader
│   ├── dashboard.py        # Multi-asset Tkinter desktop trading terminal UI
│   ├── db.py               # Trade logging & Supabase analytics integration
│   ├── execution.py        # Order placement, multi-mode filling retry, & SL/TP modification
│   ├── backtest.py         # Historical multi-timeframe backtesting engine & performance analytics
│   ├── ml_dataset.py       # Multi-asset dataset builder & breakeven-aware triple-barrier labeling
│   ├── ml_features.py      # Technical indicator engine & ML feature extractor
│   ├── patterns.py         # 15M FVG & 1M Market Structure Shift (MSS) detection logic
│   ├── risk.py             # Position sizing & risk manager
│   ├── run_live.py         # Multi-symbol real-time live trading execution engine
│   ├── strategy.py         # Strategy definition & multi-timeframe signal generator
│   └── train_xgb.py        # XGBoost classifier trainer with CV & threshold evaluator
├── tests/
│   ├── test_ml_pipeline.py # Unit tests for ML feature extraction & model inference
│   └── test_strategy.py    # Unit tests for 15M FVG & 1M MSS buy/sell detection
├── EquantEdge.bat          # 1-click Windows desktop batch launcher
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation & setup guide
```

---

## 🚀 Quick Start (Local)

1. **Prerequisites**: [MetaTrader 5 Desktop](https://www.metatrader5.com/) logged into your broker account & Python 3.10+.
2. **Setup**:
   ```bash
   git clone https://github.com/Isuru128/EquantEdge.git
   cd EquantEdge
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Backtest on Historical MT5 Candles**:
   ```bash
   # Backtest EURUSD, GBPUSD, AUDUSD across 50,000 1-minute historical candles:
   python -m src.backtest --symbols EURUSD,GBPUSD,AUDUSD --candles 50000

   # Or run backtest on a single symbol:
   python -m src.backtest --symbol EURUSD --candles 50000
   ```

4. **Generate Dataset & Train ML Confidence Model (Optional)**:
   ```bash
   python -m src.ml_dataset --symbols EURUSD,GBPUSD,AUDUSD --candles 50000
   python -m src.train_xgb --threshold 0.50
   ```

5. **Launch Live Automated Execution Engine**:
   ```bash
   # Real-time automated trading on MT5:
   python -m src.run_live

   # Or dry-run simulation mode:
   python -m src.run_live --dry-run
   ```

6. **Launch Visual Desktop Terminal**:
   ```bash
   # Live MT5 Terminal UI:
   python -m src.dashboard --live

   # Offline Demo UI:
   python -m src.dashboard
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

5. **Backtest, Train & Run 24/7 Live Trader**:
   ```powershell
   # 1. Backtest strategy
   python -m src.backtest --symbols EURUSD,GBPUSD,AUDUSD --candles 50000

   # 2. Start live automated execution loop
   python -m src.run_live
   ```

6. **(Optional) Run as a Persistent Windows Background Service**:
   - Use Windows Task Scheduler or NSSM (Non-Sucking Service Manager) to automatically start `python -m src.run_live` upon server boot.
