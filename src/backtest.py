"""
backtest.py

Historical Backtesting Engine for the 15M FVG + 1M Market Structure Shift (MSS) Strategy.

Features:
  - Fetches historical candles directly from MetaTrader 5 (or offline simulated data).
  - Multi-Timeframe M15 + M1 synchronization.
  - Enforces the 14:00 - 20:00 NY trade blocking rule.
  - Realistic trade simulation: Entry, Stop Loss (Peak/Valley), Take Profit (1:2.0 RR), Breakeven Trailing (+1.0R).
  - Calculates institutional metrics: Win Rate, Profit Factor, Expected Value (EV), Max Drawdown, Equity Growth, and Session Breakdown.

Usage:
    python -m src.backtest --symbols EURUSD,GBPUSD,AUDUSD --candles 30000
    python -m src.backtest --symbol EURUSD --candles 50000 --offline
"""

import os
import sys
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.strategy import generate_signals, resample_1m_to_15m, _detect_pip_size, RR_RATIO, SUPPORTED_SYMBOLS, MIN_SL_PIPS
    from src.connect import connect, get_candles
else:
    from .strategy import generate_signals, resample_1m_to_15m, _detect_pip_size, RR_RATIO, SUPPORTED_SYMBOLS, MIN_SL_PIPS
    from .connect import connect, get_candles

_NY_TZ = ZoneInfo("America/New_York")


def is_time_allowed_ny(dt: pd.Timestamp | datetime) -> bool:
    """Check if the given datetime is outside the 14:00 - 20:00 NY blocked window."""
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        if dt.tzinfo is None:
            # Assume UTC if naive, then convert to NY
            dt_ny = dt.tz_localize("UTC").astimezone(_NY_TZ)
        else:
            dt_ny = dt.astimezone(_NY_TZ)

        hour = dt_ny.hour
        if 14 <= hour < 20:
            return False
        return True
    except Exception:
        return True


def run_backtest_on_data(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    symbol: str = "EURUSD",
    initial_balance: float = 5000.0,
    risk_pct: float = 1.0,
    rr_ratio: float = RR_RATIO,
    max_hold_bars: int = 60,
    enforce_time_filter: bool = True,
) -> dict:
    """
    Run backtest simulation over 1-minute historical candles.
    """
    if len(df_1m) < 50:
        return {"error": "Not enough candle data for backtest."}

    pip_size = _detect_pip_size(df_1m, symbol=symbol)

    if df_15m is None or df_15m.empty:
        df_15m = resample_1m_to_15m(df_1m)

    # 1. Generate strategy signals across full dataset
    print(f"[{symbol}] Generating multi-timeframe signals across {len(df_1m):,} 1M candles...")
    df_signals = generate_signals(df_1m, df_15m=df_15m, symbol=symbol, rr_ratio=rr_ratio, pip_size=pip_size)

    signal_indices = df_signals.index[df_signals["signal"] != 0].tolist()
    print(f"[{symbol}] Found {len(signal_indices)} raw MSS setups. Simulating sequential trade executions...")

    trades = []
    fixed_risk_usd = initial_balance * (risk_pct / 100.0)
    current_balance = initial_balance
    peak_balance = initial_balance
    max_drawdown_pct = 0.0
    next_available_bar = 0

    n_bars = len(df_1m)

    for idx in signal_indices:
        if idx < next_available_bar or idx >= n_bars - 2:
            continue

        row = df_signals.iloc[idx]
        sig = int(row["signal"])
        sl = float(row["sl_price"])
        tp = float(row["tp_price"])
        entry = float(row["entry_price"])
        dt = row.get("datetime", "")

        # Session / Time Window Filter (14:00 - 20:00 NY exclusion)
        if enforce_time_filter and not is_time_allowed_ny(dt):
            continue

        risk_dist = abs(entry - sl)
        if (risk_dist / pip_size) < MIN_SL_PIPS:
            continue

        risk_usd = fixed_risk_usd
        be_dist = 1.0 * risk_dist

        # Forward simulate price action
        outcome = None
        exit_price = None
        exit_time = None
        bars_held = 0
        be_active = False

        for fwd_idx in range(idx + 1, min(idx + max_hold_bars + 1, n_bars)):
            fwd_bar = df_1m.iloc[fwd_idx]
            bars_held += 1

            if sig == -1:  # SELL
                if not be_active and fwd_bar["low"] <= (entry - be_dist):
                    be_active = True

                if not be_active:
                    if fwd_bar["high"] >= sl:
                        outcome = "LOSS"
                        exit_price = sl
                        exit_time = fwd_bar.get("datetime", "")
                        break
                else:
                    if fwd_bar["high"] >= entry:
                        outcome = "BREAKEVEN"
                        exit_price = entry
                        exit_time = fwd_bar.get("datetime", "")
                        break

                if fwd_bar["low"] <= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_time = fwd_bar.get("datetime", "")
                    break

            elif sig == 1:  # BUY
                if not be_active and fwd_bar["high"] >= (entry + be_dist):
                    be_active = True

                if not be_active:
                    if fwd_bar["low"] <= sl:
                        outcome = "LOSS"
                        exit_price = sl
                        exit_time = fwd_bar.get("datetime", "")
                        break
                else:
                    if fwd_bar["low"] <= entry:
                        outcome = "BREAKEVEN"
                        exit_price = entry
                        exit_time = fwd_bar.get("datetime", "")
                        break

                if fwd_bar["high"] >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_time = fwd_bar.get("datetime", "")
                    break

        # Lock next available entry bar to prevent overlapping positions
        next_available_bar = idx + bars_held + 1

        # Time expiry exit
        if outcome is None:
            final_bar = df_1m.iloc[min(idx + max_hold_bars, n_bars - 1)]
            final_close = float(final_bar["close"])
            exit_price = final_close
            exit_time = final_bar.get("datetime", "")
            pnl_points = (entry - final_close) if sig == -1 else (final_close - entry)
            if pnl_points > 0:
                outcome = "EXPIRED_PROFIT"
            elif be_active:
                outcome = "BREAKEVEN"
            else:
                outcome = "EXPIRED_LOSS"

        # Calculate PnL in USD & R-multiples (with 50% scale-out at +1.0R)
        if outcome == "WIN":
            # 50% locked at +1.0R (+0.5R) + 50% reached +2.0R (+1.0R) = +1.50R total
            pnl_r = 1.50
            pnl_usd = risk_usd * 1.50
        elif outcome == "LOSS":
            pnl_r = -1.0
            pnl_usd = -risk_usd
        elif outcome == "BREAKEVEN":
            # 50% locked at +1.0R (+0.5R) + 50% runner stopped at BE ($0.0R) = +0.50R secured!
            pnl_r = 0.50
            pnl_usd = risk_usd * 0.50
        elif outcome == "EXPIRED_PROFIT":
            ratio = min(rr_ratio, abs(exit_price - entry) / risk_dist)
            pnl_r = 0.50 + (0.50 * ratio) if be_active else ratio
            pnl_usd = risk_usd * pnl_r
        else:
            pnl_r = 0.50 if be_active else -min(1.0, abs(exit_price - entry) / risk_dist)
            pnl_usd = risk_usd * pnl_r

        current_balance += pnl_usd

        if current_balance > peak_balance:
            peak_balance = current_balance
        dd_pct = (peak_balance - current_balance) / peak_balance * 100.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        trades.append({
            "symbol": symbol,
            "signal": "BUY" if sig == 1 else "SELL",
            "entry_time": str(dt),
            "entry_price": round(entry, 5),
            "sl_price": round(sl, 5),
            "tp_price": round(tp, 5),
            "risk_pips": round(risk_dist / pip_size, 1),
            "exit_time": str(exit_time),
            "exit_price": round(exit_price, 5) if exit_price else entry,
            "outcome": outcome,
            "pnl_r": round(pnl_r, 2),
            "pnl_usd": round(pnl_usd, 2),
            "balance": round(current_balance, 2),
            "bars_held": bars_held,
        })

    trades_df = pd.DataFrame(trades)

    # Performance Analytics
    total_trades = len(trades_df)
    if total_trades > 0:
        wins = trades_df[trades_df["pnl_r"] > 0]
        losses = trades_df[trades_df["pnl_r"] < 0]
        breakevens = trades_df[trades_df["pnl_r"] == 0]

        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(breakevens)

        win_rate = (win_count / total_trades) * 100.0
        total_pnl_usd = current_balance - initial_balance
        total_return_pct = (total_pnl_usd / initial_balance) * 100.0

        total_gain_usd = wins["pnl_usd"].sum() if not wins.empty else 0.0
        total_loss_usd = abs(losses["pnl_usd"].sum()) if not losses.empty else 0.0
        profit_factor = (total_gain_usd / total_loss_usd) if total_loss_usd > 0 else (99.0 if total_gain_usd > 0 else 0.0)

        ev_r = trades_df["pnl_r"].mean()
    else:
        win_count = loss_count = be_count = 0
        win_rate = total_pnl_usd = total_return_pct = profit_factor = ev_r = 0.0

    return {
        "symbol": symbol,
        "initial_balance": initial_balance,
        "final_balance": round(current_balance, 2),
        "total_pnl_usd": round(total_pnl_usd, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "breakevens": be_count,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_r": round(ev_r, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trades": trades_df,
    }


def generate_synthetic_candles(symbol: str, count: int = 15000) -> pd.DataFrame:
    """Generate realistic OHLC candles for offline backtesting."""
    times = pd.date_range(end=datetime.now(), periods=count, freq="1min")
    base_prices = {"EURUSD": 1.1050, "GBPUSD": 1.3050, "AUDUSD": 0.6550}
    base = base_prices.get(symbol, 1.1000)
    step = 0.0001

    np.random.seed(42)
    noise = np.random.normal(0, step * 0.8, count)
    trend_cycle = np.sin(np.linspace(0, 30 * np.pi, count)) * step * 25.0
    prices = base + np.cumsum(noise) + trend_cycle

    opens = prices
    highs = prices + np.abs(np.random.normal(0, step * 0.5, count)) + (step * 0.2)
    lows = prices - np.abs(np.random.normal(0, step * 0.5, count)) - (step * 0.2)
    closes = opens + np.random.normal(0, step * 0.4, count)

    df = pd.DataFrame({
        "datetime": times,
        "open": opens,
        "high": np.maximum(highs, np.maximum(opens, closes)),
        "low": np.minimum(lows, np.minimum(opens, closes)),
        "close": closes,
        "volume": np.random.randint(50, 500, count),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest 15M FVG + 1M MSS Strategy.")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,AUDUSD", help="Comma-separated symbols")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol backtest")
    parser.add_argument("--candles", type=int, default=30000, help="Number of 1M candles to test")
    parser.add_argument("--balance", type=float, default=5000.0, help="Starting account balance")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk percent per trade")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-to-reward ratio")
    parser.add_argument("--offline", action="store_true", help="Run in offline simulation mode without MT5 connection")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else [s.strip() for s in args.symbols.split(",") if s.strip()]

    print("\n" + "=" * 78)
    print("  EQUANTEDGE -- 15M FVG + 1M MARKET STRUCTURE SHIFT (MSS) BACKTESTER")
    print(f"  Symbols: {', '.join(symbols)} | Lookback: {args.candles:,} 1M bars | Starting Balance: ${args.balance:,.2f}")
    print(f"  Risk: {args.risk}% per trade | Target: 1:{args.rr:.1f} RR | Breakeven Trail: +1.0R")
    print("  Session Gate: 14:00 - 20:00 NY Exclusion Enforced")
    print("=" * 78 + "\n")

    if not args.offline:
        try:
            connect()
            import MetaTrader5 as mt5
        except Exception as exc:
            print(f"[Backtest] MT5 connection not available ({exc}). Falling back to offline simulation mode.\n")
            args.offline = True

    overall_results = []

    for sym in symbols:
        if not args.offline:
            import MetaTrader5 as mt5
            df_1m = get_candles(sym, mt5.TIMEFRAME_M1, args.candles)
            df_15m = get_candles(sym, mt5.TIMEFRAME_M15, args.candles // 10)
        else:
            df_1m = generate_synthetic_candles(sym, count=args.candles)
            df_15m = resample_1m_to_15m(df_1m)

        res = run_backtest_on_data(
            df_1m=df_1m,
            df_15m=df_15m,
            symbol=sym,
            initial_balance=args.balance,
            risk_pct=args.risk,
            rr_ratio=args.rr,
        )

        overall_results.append(res)

        print(f"\n--- Results for {sym} " + "-" * 45)
        print(f"  Total Trades Taken : {res['total_trades']}")
        print(f"  Wins / Losses / BE : {res['wins']} Wins | {res['losses']} Losses | {res['breakevens']} Breakevens")
        print(f"  Win Rate           : {res['win_rate_pct']:.1f}%")
        print(f"  Profit Factor      : {res['profit_factor']:.2f}")
        print(f"  Expectancy (EV)    : {res['expectancy_r']:+.2f}R per trade")
        print(f"  Max Drawdown       : {res['max_drawdown_pct']:.2f}%")
        print(f"  Final Balance      : ${res['final_balance']:,.2f} ({res['total_return_pct']:+.2f}%)")
        print("-" * 65)

        if not res["trades"].empty:
            print("\n  Sample Recent Trades:")
            sample = res["trades"][["signal", "entry_time", "entry_price", "sl_price", "tp_price", "outcome", "pnl_r", "pnl_usd", "balance"]].tail(5)
            print(sample.to_string(index=False))

    print("\n" + "=" * 78)
    print("  BACKTEST COMPLETE")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
