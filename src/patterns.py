"""
patterns.py

Candlestick and price action pattern detection utilities, including:
  - 15-Minute Fair Value Gap (FVG) detection
  - 1-Minute Market Structure Shift (MSS) identification
  - Classical candlestick patterns (Inside Bar, Pin Bar, Engulfing)
"""

import pandas as pd
import numpy as np


def find_15m_fvgs(df_15m: pd.DataFrame, max_lookback_bars: int = 96) -> list[dict]:
    """
    Detect Fair Value Gaps (FVG) on the 15-minute timeframe.
    
    Rules:
      - Max age: cannot exceed `max_lookback_bars` (96 15M candles = 24 hours).
      - Bearish FVG (Sell Imbalance):
          Candle 1 (t-2) Low > Candle 3 (t) High
          Gap zone = [Candle 3 High, Candle 1 Low]
          Midpoint (CE / Consequent Encroachment) = (Candle 3 High + Candle 1 Low) / 2
      - Bullish FVG (Buy Imbalance):
          Candle 1 (t-2) High < Candle 3 (t) Low
          Gap zone = [Candle 1 High, Candle 3 Low]
          Midpoint = (Candle 1 High + Candle 3 Low) / 2
    
    Returns a list of FVG dicts sorted from newest to oldest.
    """
    if len(df_15m) < 3:
        return []

    n = len(df_15m)
    start_idx = max(2, n - max_lookback_bars)
    fvgs = []

    for i in range(start_idx, n):
        c1 = df_15m.iloc[i - 2]  # First candle
        c2 = df_15m.iloc[i - 1]  # Middle impulse candle
        c3 = df_15m.iloc[i]      # Third candle

        # Bearish FVG
        if c1["low"] > c3["high"]:
            gap_top = float(c1["low"])
            gap_bottom = float(c3["high"])
            gap_size = gap_top - gap_bottom
            if gap_size > 0:
                fvgs.append({
                    "type": "BEARISH",
                    "direction": -1,
                    "top": gap_top,
                    "bottom": gap_bottom,
                    "midpoint": round((gap_top + gap_bottom) / 2, 5),
                    "size": round(gap_size, 5),
                    "formed_idx": i,
                    "datetime": str(c3["datetime"]) if "datetime" in c3 else str(i),
                    "bars_ago": n - 1 - i,
                })

        # Bullish FVG
        elif c1["high"] < c3["low"]:
            gap_top = float(c3["low"])
            gap_bottom = float(c1["high"])
            gap_size = gap_top - gap_bottom
            if gap_size > 0:
                fvgs.append({
                    "type": "BULLISH",
                    "direction": 1,
                    "top": gap_top,
                    "bottom": gap_bottom,
                    "midpoint": round((gap_top + gap_bottom) / 2, 5),
                    "size": round(gap_size, 5),
                    "formed_idx": i,
                    "datetime": str(c3["datetime"]) if "datetime" in c3 else str(i),
                    "bars_ago": n - 1 - i,
                })

    # Return newest first
    fvgs.reverse()
    return fvgs


def get_nearest_active_fvg(
    current_price: float,
    fvgs: list[dict],
    side: int | None = None,
) -> dict | None:
    """
    Find the nearest active 15M Fair Value Gap to the current price.
    - side: 1 (Bullish), -1 (Bearish), or None (either).
    """
    matching_fvgs = [f for f in fvgs if side is None or f["direction"] == side]
    if not matching_fvgs:
        return None

    def distance_to_fvg(fvg):
        if fvg["bottom"] <= current_price <= fvg["top"]:
            return 0.0
        elif current_price < fvg["bottom"]:
            return fvg["bottom"] - current_price
        else:
            return current_price - fvg["top"]

    matching_fvgs.sort(key=distance_to_fvg)
    return matching_fvgs[0]


def is_price_in_fvg(price: float, fvg: dict, tolerance_pips: float = 0.0, pip_size: float = 0.0001) -> bool:
    """
    Check if the given price is inside (or tapping) the FVG zone.
    """
    tol = tolerance_pips * pip_size
    return (fvg["bottom"] - tol) <= price <= (fvg["top"] + tol)


def detect_market_structure_shift(
    df_1m: pd.DataFrame,
    fvg: dict,
    trend_side: int,
    lookback_bars: int = 40,
    pip_size: float = 0.0001,
) -> dict | None:
    """
    Detect 1-Minute Market Structure Shift (MSS) inside/tapping the 15M FVG.

    Sell Scenario (trend_side == -1):
      1. Price has entered/tapped the 15M Bearish FVG zone within the last `lookback_bars`.
      2. Price established a bullish leg with a swing high peak (at index `peak_idx`).
      3. Identify the swing low preceding that peak (the origin of the last bullish leg).
      4. The latest closed candle (iloc[-2] or iloc[-1]) closes strictly BELOW that swing low.
      5. Signal:
         - entry_price = Swing Low Price (the broken level)
         - sl_price = Peak High Price (the swing high)
         - tp_price = entry_price - (2.0 * (sl_price - entry_price))

    Buy Scenario (trend_side == 1):
      1. Price has entered/tapped the 15M Bullish FVG zone within the last `lookback_bars`.
      2. Price established a bearish leg with a swing low valley (at index `valley_idx`).
      3. Identify the swing high preceding that valley (the origin of the last bearish leg).
      4. The latest closed candle closes strictly ABOVE that swing high.
      5. Signal:
         - entry_price = Swing High Price (the broken level)
         - sl_price = Valley Low Price (the swing low)
         - tp_price = entry_price + (2.0 * (entry_price - sl_price))
    """
    if len(df_1m) < 10:
        return None

    # Focus on the recent window of bars
    window_df = df_1m.iloc[-lookback_bars:].copy().reset_index(drop=True)
    n = len(window_df)
    if n < 6:
        return None

    # Last closed bar is at index n-2 (n-1 is current actively forming bar)
    curr_bar = window_df.iloc[-2]

    # Check if price has interacted with the 15M FVG in this window
    fvg_bottom = fvg["bottom"]
    fvg_top = fvg["top"]

    touched_fvg = (
        (window_df["high"] >= fvg_bottom) & (window_df["low"] <= fvg_top)
    ).any()

    if not touched_fvg:
        return None

    if trend_side == -1:  # SELL SCENARIO
        fvg_bars = window_df[window_df["high"] >= fvg_bottom]
        if fvg_bars.empty:
            return None

        # Highest bar in the window (the peak / liquidity sweep)
        peak_idx = int(window_df["high"].idxmax())
        peak_high = float(window_df.loc[peak_idx, "high"])

        # The peak must be before current closed bar (n-2) to allow reversal
        if peak_idx >= n - 2 or peak_idx < 1:
            return None

        # Base of the bullish leg inside/entering the FVG leading up to the peak
        fvg_entry_idx = int(fvg_bars.index.min())
        start_leg_idx = max(0, min(fvg_entry_idx, peak_idx - 1))
        pre_peak_df = window_df.iloc[start_leg_idx: peak_idx]
        if pre_peak_df.empty:
            pre_peak_df = window_df.iloc[max(0, peak_idx - 3): peak_idx]
        if pre_peak_df.empty:
            return None

        swing_low_idx = int(pre_peak_df["low"].idxmin())
        swing_low_price = float(pre_peak_df.loc[swing_low_idx, "low"])

        # Condition for Market Structure Shift:
        # Current closed candle (n-2) closes BELOW the swing low of the last bullish leg
        # AND price held above the swing low after the peak
        post_peak_closes = window_df["close"].iloc[peak_idx: -2]
        prior_held = post_peak_closes.max() >= (swing_low_price - (0.5 * pip_size)) if not post_peak_closes.empty else True

        curr_close = float(curr_bar["close"])
        is_break = (curr_close < swing_low_price) and prior_held

        if is_break:
            risk = peak_high - swing_low_price
            if risk <= (0.5 * pip_size):  # Minimum viable risk floor
                return None

            return {
                "signal": -1,
                "type": "SELL_MSS",
                "entry_price": swing_low_price,
                "trigger_close": curr_close,
                "sl_price": peak_high,
                "tp_price": swing_low_price - (2.0 * risk),
                "risk_distance": risk,
                "sl_pips": risk / pip_size,
                "tp_pips": (2.0 * risk) / pip_size,
                "peak_high": peak_high,
                "swing_low": swing_low_price,
                "fvg_top": fvg_top,
                "fvg_bottom": fvg_bottom,
                "mss_time": str(curr_bar.get("datetime", "")),
            }

    elif trend_side == 1:  # BUY SCENARIO (Mirror)
        fvg_bars = window_df[window_df["low"] <= fvg_top]
        if fvg_bars.empty:
            return None

        # Lowest valley in the window
        valley_idx = int(window_df["low"].idxmin())
        valley_low = float(window_df.loc[valley_idx, "low"])

        if valley_idx >= n - 2 or valley_idx < 1:
            return None

        # Base of the bearish leg inside/entering the FVG leading down to the valley
        fvg_entry_idx = int(fvg_bars.index.min())
        start_leg_idx = max(0, min(fvg_entry_idx, valley_idx - 1))
        pre_valley_df = window_df.iloc[start_leg_idx: valley_idx]
        if pre_valley_df.empty:
            pre_valley_df = window_df.iloc[max(0, valley_idx - 3): valley_idx]
        if pre_valley_df.empty:
            return None

        swing_high_idx = int(pre_valley_df["high"].idxmax())
        swing_high_price = float(pre_valley_df.loc[swing_high_idx, "high"])

        # Condition for Market Structure Shift:
        # Current closed candle closes ABOVE the swing high of the last bearish leg
        post_valley_closes = window_df["close"].iloc[valley_idx: -2]
        prior_held = post_valley_closes.min() <= (swing_high_price + (0.5 * pip_size)) if not post_valley_closes.empty else True

        curr_close = float(curr_bar["close"])
        is_break = (curr_close > swing_high_price) and prior_held

        if is_break:
            risk = swing_high_price - valley_low
            if risk <= (0.5 * pip_size):
                return None

            return {
                "signal": 1,
                "type": "BUY_MSS",
                "entry_price": swing_high_price,
                "trigger_close": curr_close,
                "sl_price": valley_low,
                "tp_price": swing_high_price + (2.0 * risk),
                "risk_distance": risk,
                "sl_pips": risk / pip_size,
                "tp_pips": (2.0 * risk) / pip_size,
                "valley_low": valley_low,
                "swing_high": swing_high_price,
                "fvg_top": fvg_top,
                "fvg_bottom": fvg_bottom,
                "mss_time": str(curr_bar.get("datetime", "")),
            }

    return None


# ── Classical Candlestick Primitives ──────────────────────────────────────────

def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    mother_range = prev_high - prev_low
    min_mother_range = df["close"].mean() * 0.0005

    inside = (
        (df["high"] < prev_high) &
        (df["low"] > prev_low) &
        (mother_range > min_mother_range)
    )
    return inside.fillna(False)


def is_pin_bar(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    body_pct = body / range_.replace(0, pd.NA)

    pin = (
        ((lower_wick > body * 2) | (upper_wick > body * 2)) &
        (body_pct < 0.35)
    )
    return pin.fillna(False)


def is_engulfing(df: pd.DataFrame) -> pd.Series:
    curr_open = df["open"]
    curr_close = df["close"]
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    curr_body_high = df[["open", "close"]].max(axis=1)
    curr_body_low = df[["open", "close"]].min(axis=1)
    prev_body_high = df[["open", "close"]].max(axis=1).shift(1)
    prev_body_low = df[["open", "close"]].min(axis=1).shift(1)

    bullish_engulf = (
        (curr_close > curr_open) &
        (prev_close < prev_open) &
        (curr_body_high > prev_body_high) &
        (curr_body_low < prev_body_low)
    )
    bearish_engulf = (
        (curr_close < curr_open) &
        (prev_close > prev_open) &
        (curr_body_high > prev_body_high) &
        (curr_body_low < prev_body_low)
    )
    return (bullish_engulf | bearish_engulf).fillna(False)
