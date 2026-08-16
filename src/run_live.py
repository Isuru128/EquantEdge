"""
run_live.py

Main dry-run loop for the Inside Bar Breakout strategy.

What this script does:
  1. Connects to a running MT5 terminal (demo account).
  2. Every M15 bar close, fetches the latest candles.
  3. Runs the strategy to check for a signal on the last closed bar.
  4. Prints any signal with entry levels (mother bar high/low).
  5. Does NOT place real orders — this is a signal-monitor / dry-run.

To place real orders, wire up execution.py (not yet implemented).

Usage:
    python src/run_live.py
"""

import time
import MetaTrader5 as mt5
from datetime import datetime

from .connect import connect, get_candles
from .strategy import get_latest_signal
from .execution import place_order, close_position, DRY_RUN

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL      = "EURUSD"
TIMEFRAME   = mt5.TIMEFRAME_M15
NUM_CANDLES = 200          # needs to be > EMA period (50) + buffer
POLL_SECS   = 60           # check every 60 s; M15 bars close every 900 s
RISK_PCT    = 1.0          # % of account balance to risk per trade
SL_PIPS     = 15           # stop-loss distance in pips from entry
TP_PIPS     = 30           # take-profit distance in pips (2:1 R:R)
# ─────────────────────────────────────────────────────────────────────────────


def print_signal(sig: dict) -> None:
    """Pretty-print a signal dict to the console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sig["signal"] == 0:
        print(f"[{ts}]  No signal  |  Trend: {sig.get('trend','?')}  "
              f"|  EMA-50: {sig.get('ema_50','?')}  "
              f"|  Inside bar: {sig.get('inside_bar', False)}")
        return

    direction = "🟢 BUY " if sig["signal"] == 1 else "🔴 SELL"
    entry_key = "mother_high" if sig["signal"] == 1 else "mother_low"
    entry_lvl = sig.get(entry_key)

    print(
        f"\n{'='*60}\n"
        f"  {direction} SIGNAL DETECTED\n"
        f"  Time        : {sig['datetime']}\n"
        f"  Close       : {sig['close']}\n"
        f"  EMA-50      : {sig['ema_50']}\n"
        f"  Trend       : {sig['trend']}\n"
        f"  Entry level : {entry_lvl}  ({entry_key})\n"
        f"  Mother high : {sig['mother_high']}\n"
        f"  Mother low  : {sig['mother_low']}\n"
        f"{'='*60}\n"
    )


def main() -> None:
    print("EquantEdge — Inside Bar Breakout Strategy")
    print(f"Mode: {'DRY RUN (no real orders)' if DRY_RUN else '⚠️  LIVE TRADING'}")
    print(f"Symbol: {SYMBOL}  |  Timeframe: M15  |  Poll: every {POLL_SECS}s")
    print("Press Ctrl+C to stop.\n")

    account = connect()
    open_ticket = None   # track if we have a position open

    last_seen_bar = None

    try:
        while True:
            df  = get_candles(SYMBOL, TIMEFRAME, NUM_CANDLES)
            sig = get_latest_signal(df)

            bar_time = sig.get("datetime")
            if bar_time != last_seen_bar:
                last_seen_bar = bar_time
                print_signal(sig)

                # ── Act on signal ──────────────────────────────────────────
                if sig["signal"] != 0 and open_ticket is None:
                    ticket = place_order(
                        symbol      = SYMBOL,
                        signal      = sig["signal"],
                        entry_price = sig["mother_high"] if sig["signal"] == 1 else sig["mother_low"],
                        sl_pips     = SL_PIPS,
                        tp_pips     = TP_PIPS,
                        account_balance = account.balance,
                        risk_pct    = RISK_PCT,
                    )
                    if ticket:
                        open_ticket = ticket
                        print(f"  ↳ Order placed — ticket #{ticket}")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()
