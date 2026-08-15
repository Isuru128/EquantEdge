# EquantPilot

Automated forex trading tool that connects to a MetaTrader 5 account,
detects price-action candlestick patterns, and executes trades based on
a defined strategy.

## ⚠️ Status
Early development. **Not yet backtested or live-trading ready.**
Always test on a demo account first.

## Project structure

```
EquantPilot/
├── src/
│   ├── connect.py          # MT5 terminal connection + data pull
│   ├── patterns.py         # Candlestick pattern detection logic
│   ├── strategy.py         # Signal generation from patterns
│   ├── risk.py             # Position sizing, stop-loss/take-profit rules
│   ├── execution.py        # Order placement/management via MT5 API
│   └── run_live.py         # Main live/demo trading loop
├── tests/                  # Unit tests for pattern + strategy logic
├── notebooks/              # Exploratory analysis (pattern testing, etc.)
├── logs/                   # Trade + error logs (gitignored)
├── data/                   # Cached historical candle data (gitignored)
├── requirements.txt
├── config.example.yaml     # Copy to config.yaml and fill in your settings
└── README.md
```

## Setup

1. Install [MetaTrader 5](https://www.metatrader5.com/) and open a **demo account**.
2. Install Python 3.10+.
3. Clone this repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd EquantPilot
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
4. Copy `config.example.yaml` to `config.yaml` and fill in your symbol,
   timeframe, and risk settings (do NOT commit `config.yaml` if it ever
   contains credentials).
5. Make sure the MT5 terminal is open and logged into your **demo** account.
6. Run the connection test:
   ```bash
   python src/connect.py
   ```

## Roadmap

- [x] MT5 connection + historical data pull
- [ ] Candlestick pattern detection (pin bar, engulfing, inside bar, etc.)
- [ ] Signal generation from patterns
- [ ] Risk management (position sizing, SL/TP, max daily loss)
- [ ] Backtesting harness
- [ ] Live/demo trading loop with logging + alerts
- [ ] Paper trading validation period
- [ ] Live deployment (small size)

## Safety notes

- Never commit account credentials, API keys, or `config.yaml` with secrets.
- Always test changes on a demo account before touching a live account.
- Set hard risk limits (max position size, max daily loss) before any live run.
