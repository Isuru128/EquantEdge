"""
mt5_connect.py

Step 1 of the automated trading tool: connect to a running MetaTrader 5
terminal and pull OHLC (candlestick) data for price-action analysis.

Requirements:
    pip install MetaTrader5 pandas

Prerequisites:
    - MetaTrader 5 desktop terminal installed and running on this machine
      (Windows, or Windows VPS; Wine on Linux/Mac is unsupported here).
    - Logged into your account (demo recommended to start) inside the
      MT5 terminal itself, OR pass login credentials below.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15   # e.g. M1, M5, M15, M30, H1, H4, D1
NUM_CANDLES = 500               # how many bars of history to pull

# Optional: only needed if you want to log in from the script instead of
# being already logged in via the terminal UI. Leave as None to skip.
LOGIN = None          # e.g. 12345678
PASSWORD = None        # e.g. "your_password"
SERVER = None          # e.g. "YourBroker-Demo"
 
def connect(silent: bool = False):
    """Initialize connection to the local MT5 terminal."""
    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed, error: {mt5.last_error()}")

    if LOGIN and PASSWORD and SERVER:
        authorized = mt5.login(LOGIN, password=PASSWORD, server=SERVER)
        if not authorized:
            mt5.shutdown()
            raise RuntimeError(f"login() failed, error: {mt5.last_error()}")

    account_info = mt5.account_info()
    if account_info is None:
        mt5.shutdown()
        raise RuntimeError("Could not fetch account info. Is a terminal running and logged in?")

    if not silent:
        print("Connected successfully.")
        print(f"  Account:  {account_info.login}")
        print(f"  Server:   {account_info.server}")
        print(f"  Balance:  {account_info.balance} {account_info.currency}")
        print(f"  Leverage: 1:{account_info.leverage}")
    return account_info


def get_candles(symbol: str, timeframe, count: int) -> pd.DataFrame:
    """Pull the most recent `count` candles for `symbol` as a DataFrame with chunked fallback."""
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    # If single request failed (e.g. requested count > MT5 single buffer limit ~50k-65k)
    if rates is None or len(rates) == 0:
        chunk_size = 40000
        collected_rates = []
        offset = 0
        while offset < count:
            batch_count = min(chunk_size, count - offset)
            batch = mt5.copy_rates_from_pos(symbol, timeframe, offset, batch_count)
            if batch is None or len(batch) == 0:
                break
            collected_rates.append(batch)
            offset += len(batch)
            if len(batch) < batch_count:
                break  # Reached oldest available bar in broker history

        if collected_rates:
            import numpy as np
            rates = np.concatenate(collected_rates[::-1])  # chronological order
        else:
            # Final fallback: try fetching whatever max bars broker has
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50000)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No data returned for {symbol}. Check the symbol is visible "
            f"in MT5's Market Watch window (right-click > Show All)."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={
        "time": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume",
    }, inplace=True)
    df.drop_duplicates(subset=["datetime"], inplace=True)
    df.sort_values(by="datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]



def basic_candle_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a couple of simple, transparent price-action tags as a starting
    point. Replace/extend this with your actual pattern logic.
    """
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["bullish"] = df["close"] > df["open"]
    df["body_pct"] = (body / range_.replace(0, pd.NA)).astype(float)
    df["is_doji"] = df["body_pct"] < 0.1
    df["is_pin_bar"] = (
        (lower_wick > body * 2) | (upper_wick > body * 2)
    ) & (df["body_pct"] < 0.35)
    return df


if __name__ == "__main__":
    connect()

    df = get_candles(SYMBOL, TIMEFRAME, NUM_CANDLES)
    df = basic_candle_tags(df)

    print(f"\nLast 10 candles for {SYMBOL}:")
    print(df.tail(10).to_string(index=False))

    out_path = f"{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} candles to {out_path}")

    mt5.shutdown()
