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


def is_liquidity_sweep_sell(df: pd.DataFrame, swing_window: int | None = 10) -> pd.Series:
    """
    Bearish Liquidity Sweep & Reversal Pattern:
      1. Candle 1 (t-1) is Bullish (prev_close > prev_open)
      2. Candle 2 (t) is Bearish (curr_close < curr_open)
      3. Candle 2 sweeps Candle 1 High (curr_high > prev_high) -> Liquidity Taken
      4. Candle 2 closes below Candle 1 Open (curr_close < prev_open) -> Full breakdown confirmation
      5. (Optional) Candle 1 High is a prominent swing high over `swing_window` bars.
    """
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high  = df["high"].shift(1)

    curr_open  = df["open"]
    curr_close = df["close"]
    curr_high  = df["high"]

    cond_prev_bullish = prev_close > prev_open
    cond_curr_bearish = curr_close < curr_open
    cond_sweep_high   = curr_high > prev_high
    cond_close_break  = curr_close < prev_open

    sweep_sell = (
        cond_prev_bullish &
        cond_curr_bearish &
        cond_sweep_high &
        cond_close_break
    )

    if swing_window and swing_window > 1:
        # Check that Candle 1 high was the highest high of prior swing window
        prior_high_max = df["high"].shift(1).rolling(window=swing_window, min_periods=2).max()
        cond_swing = prev_high >= (prior_high_max - 1e-6)
        sweep_sell = sweep_sell & cond_swing

    return sweep_sell.fillna(False)


def is_liquidity_sweep_buy(df: pd.DataFrame, swing_window: int | None = 10) -> pd.Series:
    """
    Bullish Liquidity Sweep & Reversal Pattern (Mirror):
      1. Candle 1 (t-1) is Bearish (prev_close < prev_open)
      2. Candle 2 (t) is Bullish (curr_close > curr_open)
      3. Candle 2 sweeps Candle 1 Low (curr_low < prev_low) -> Liquidity Taken
      4. Candle 2 closes above Candle 1 Open (curr_close > prev_open) -> Full breakout confirmation
      5. (Optional) Candle 1 Low is a prominent swing low over `swing_window` bars.
    """
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_low   = df["low"].shift(1)

    curr_open  = df["open"]
    curr_close = df["close"]
    curr_low   = df["low"]

    cond_prev_bearish = prev_close < prev_open
    cond_curr_bullish = curr_close > curr_open
    cond_sweep_low    = curr_low < prev_low
    cond_close_break  = curr_close > prev_open

    sweep_buy = (
        cond_prev_bearish &
        cond_curr_bullish &
        cond_sweep_low &
        cond_close_break
    )

    if swing_window and swing_window > 1:
        # Check that Candle 1 low was the lowest low of prior swing window
        prior_low_min = df["low"].shift(1).rolling(window=swing_window, min_periods=2).min()
        cond_swing = prev_low <= (prior_low_min + 1e-6)
        sweep_buy = sweep_buy & cond_swing

    return sweep_buy.fillna(False)


