"""
patterns.py

Candlestick pattern detection logic.

Each function takes a DataFrame of OHLC candles (from connect.py) and
returns a boolean Series flagging which rows match the pattern.
"""

import pandas as pd


def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Inside Bar: the current candle's high AND low are both contained
    within the prior candle's (mother bar's) high and low.

    Rules:
      - current high  < prior high
      - current low   > prior low
      - Mother bar should have a meaningful range (not a doji)

    Returns a boolean Series aligned with df.index.
    The first row is always False (no prior candle to compare).
    """
    prev_high = df["high"].shift(1)
    prev_low  = df["low"].shift(1)

    mother_range = prev_high - prev_low
    min_mother_range = df["close"].mean() * 0.0005  # at least 0.5 pips equivalent

    inside = (
        (df["high"] < prev_high) &
        (df["low"]  > prev_low)  &
        (mother_range > min_mother_range)
    )
    return inside.fillna(False)


def is_pin_bar(df: pd.DataFrame) -> pd.Series:
    """
    Pin Bar (aka hammer / shooting star):
      - The wick on one side is > 2x the body size
      - The body is in the outer 1/3 of the total candle range
      - body_pct < 35% of the total range

    Returns a boolean Series.
    """
    body       = (df["close"] - df["open"]).abs()
    range_     = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    body_pct = body / range_.replace(0, pd.NA)

    pin = (
        ((lower_wick > body * 2) | (upper_wick > body * 2)) &
        (body_pct < 0.35)
    )
    return pin.fillna(False)


def is_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Engulfing candle:
      - Current candle's body fully engulfs the prior candle's body
      - Opposite colour to the prior candle

    Returns a boolean Series.
    """
    curr_open  = df["open"]
    curr_close = df["close"]
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    curr_body_high = df[["open", "close"]].max(axis=1)
    curr_body_low  = df[["open", "close"]].min(axis=1)
    prev_body_high = df[["open", "close"]].max(axis=1).shift(1)
    prev_body_low  = df[["open", "close"]].min(axis=1).shift(1)

    bullish_engulf = (
        (curr_close > curr_open) &       # current is bullish
        (prev_close < prev_open) &       # prior was bearish
        (curr_body_high > prev_body_high) &
        (curr_body_low  < prev_body_low)
    )

    bearish_engulf = (
        (curr_close < curr_open) &       # current is bearish
        (prev_close > prev_open) &       # prior was bullish
        (curr_body_high > prev_body_high) &
        (curr_body_low  < prev_body_low)
    )

    return (bullish_engulf | bearish_engulf).fillna(False)
