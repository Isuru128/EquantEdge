"""
strategy.py

Inside Bar Breakout Strategy with EMA-50 trend filter (M15 timeframe).

Signal logic:
  - Detect an inside bar (via patterns.is_inside_bar)
  - Use EMA-50 to determine trend direction:
      * Price above EMA-50  → only look for BUY breakouts
      * Price below EMA-50  → only look for SELL breakouts
  - Entry trigger:
      * BUY  signal when price breaks ABOVE the mother bar's high
      * SELL signal when price breaks BELOW the mother bar's low
  - The signal is set on the inside bar candle; the caller uses the
    mother bar's high/low as the pending entry level.

Signal column values:
   1  = buy breakout pending (break above mother bar high)
  -1  = sell breakout pending (break below mother bar low)
   0  = no trade
"""

import pandas as pd
from .patterns import is_inside_bar


EMA_PERIOD = 50   # trend filter period


def add_ema(df: pd.DataFrame, period: int = EMA_PERIOD) -> pd.DataFrame:
    """Append an EMA column to the dataframe."""
    df = df.copy()
    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Inside Bar Breakout strategy on a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame from connect.get_candles(), must have columns:
        datetime, open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame with added columns:
        ema_50         – EMA-50 value on each bar
        inside_bar     – True where an inside bar is detected
        mother_high    – mother bar's high (entry level for long)
        mother_low     – mother bar's low  (entry level for short)
        signal         – 1 (buy), -1 (sell), or 0 (hold)
        signal_label   – human-readable label for the signal
    """
    df = add_ema(df)

    df["inside_bar"]  = is_inside_bar(df)
    df["mother_high"] = df["high"].shift(1)   # prior candle's high
    df["mother_low"]  = df["low"].shift(1)    # prior candle's low

    ema_col = f"ema_{EMA_PERIOD}"

    # Trend direction: 1 = bullish (close above EMA), -1 = bearish
    df["trend"] = (df["close"] > df[ema_col]).map({True: 1, False: -1})

    # Build signal: only trade inside bars aligned with the trend
    conditions_buy  = df["inside_bar"] & (df["trend"] == 1)
    conditions_sell = df["inside_bar"] & (df["trend"] == -1)

    df["signal"] = 0
    df.loc[conditions_buy,  "signal"] =  1
    df.loc[conditions_sell, "signal"] = -1

    # Human-readable label
    label_map = {1: "BUY breakout (above mother high)", -1: "SELL breakout (below mother low)", 0: ""}
    df["signal_label"] = df["signal"].map(label_map)

    return df


def get_latest_signal(df: pd.DataFrame) -> dict:
    """
    Return the signal on the most recently closed candle (second to last row,
    as the last bar is typically still forming).

    Returns a dict with signal metadata, or {'signal': 0} if no signal.
    """
    if len(df) < EMA_PERIOD + 2:
        return {"signal": 0, "reason": "Not enough bars to compute EMA"}

    df = generate_signals(df)
    # Use the last fully closed bar (exclude the still-forming last candle)
    bar = df.iloc[-2]

    return {
        "datetime":    bar["datetime"],
        "signal":      int(bar["signal"]),
        "label":       bar["signal_label"],
        "close":       bar["close"],
        "ema_50":      round(bar[f"ema_{EMA_PERIOD}"], 5),
        "trend":       "UP" if bar["trend"] == 1 else "DOWN",
        "inside_bar":  bool(bar["inside_bar"]),
        "mother_high": bar.get("mother_high"),
        "mother_low":  bar.get("mother_low"),
    }
