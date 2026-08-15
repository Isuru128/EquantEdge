"""
patterns.py

Candlestick pattern detection logic.

This is a placeholder — fill in with the exact price-action rules your
strategy trades (pin bar, engulfing, inside bar, etc.). Each function
should take a DataFrame of OHLC candles and return a boolean Series
flagging which rows match the pattern.
"""

import pandas as pd


def is_pin_bar(df: pd.DataFrame) -> pd.Series:
    """
    TODO: define your exact pin bar rule, e.g.:
    - wick on one side > 2x the body
    - body in the outer third of the candle range
    - closes in the direction of the reversal
    """
    raise NotImplementedError("Define your pin bar rule here.")


def is_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    TODO: define your exact engulfing rule, e.g.:
    - current candle's body fully engulfs the prior candle's body
    - opposite color to the prior candle
    """
    raise NotImplementedError("Define your engulfing rule here.")


def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    TODO: define your exact inside bar rule, e.g.:
    - current high < prior high AND current low > prior low
    """
    raise NotImplementedError("Define your inside bar rule here.")
