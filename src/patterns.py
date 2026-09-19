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


def find_swing_pivots(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Identify fractal Swing High and Swing Low pivots in a DataFrame.
    
    Returns
    -------
    (swing_highs, swing_lows) where each element is {'idx': i, 'price': val, 'time': datetime}
    """
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df["datetime"].astype(str).to_numpy() if "datetime" in df.columns else None
    n = len(df)
    
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, n - right_bars):
        # Swing High
        is_sh = True
        for j in range(1, left_bars + 1):
            if highs[i] <= highs[i - j]:
                is_sh = False
                break
        if is_sh:
            for j in range(1, right_bars + 1):
                if highs[i] < highs[i + j]:
                    is_sh = False
                    break
        if is_sh:
            swing_highs.append({
                "idx": i,
                "price": float(highs[i]),
                "datetime": times[i] if times is not None else str(i),
            })
            
        # Swing Low
        is_sl = True
        for j in range(1, left_bars + 1):
            if lows[i] >= lows[i - j]:
                is_sl = False
                break
        if is_sl:
            for j in range(1, right_bars + 1):
                if lows[i] > lows[i + j]:
                    is_sl = False
                    break
        if is_sl:
            swing_lows.append({
                "idx": i,
                "price": float(lows[i]),
                "datetime": times[i] if times is not None else str(i),
            })
            
    return swing_highs, swing_lows


def detect_market_structure_shift(
    df_1m: pd.DataFrame,
    fvg: dict,
    trend_side: int,
    lookback_bars: int = 60,
    pip_size: float = 0.0001,
    min_sl_pips: float = 4.0,
    require_retest: bool = True,
    require_aggressive_displacement: bool = True,
) -> dict | None:
    """
    Detect 1-Minute Market Structure Shift (MSS) with Retest Confirmation inside/tapping the 15M FVG.

    Aggressive MSS Requirements (Non-Choppy):
      1. Rapid Displacement: From peak/valley to MSS breakout must occur in <= 8 bars.
      2. High Candle Conviction: Breakout candle has strong body-to-range ratio (>= 0.40).
      3. Decisive Displacement Depth: Breakout candle closes cleanly beyond the broken level (>= 0.3 pips).
      4. Directional Leg Dominance: >= 55% of candles in the displacement leg are directional.
    """
    if len(df_1m) < 12:
        return None

    # Focus on a generous structural window
    window_df = df_1m.iloc[-lookback_bars:].copy().reset_index(drop=True)
    n = len(window_df)
    if n < 8:
        return None

    # Current closed evaluation bar is index n-2 (n-1 is currently forming)
    curr_bar = window_df.iloc[-2]
    curr_high = float(curr_bar["high"])
    curr_low = float(curr_bar["low"])
    curr_close = float(curr_bar["close"])

    fvg_bottom = fvg["bottom"]
    fvg_top = fvg["top"]

    # Ensure price interacted with the FVG
    touched_fvg = (
        (window_df["high"] >= fvg_bottom) & (window_df["low"] <= fvg_top)
    ).any()
    if not touched_fvg:
        return None

    tolerance = 0.5 * pip_size

    if trend_side == -1:  # SELL SCENARIO
        # 1. Peak High in the window
        peak_idx = int(window_df["high"].idxmax())
        peak_high = float(window_df.loc[peak_idx, "high"])

        if peak_idx >= n - 2 or peak_idx < 2:
            return None

        # 2. Find structural swing low before the peak
        pre_peak_df = window_df.iloc[:peak_idx + 1]
        _, swing_lows = find_swing_pivots(pre_peak_df, left_bars=1, right_bars=1)

        if swing_lows:
            # Pick the most significant/recent structural swing low prior to the peak
            swing_low_price = float(swing_lows[-1]["price"])
            swing_low_idx = int(swing_lows[-1]["idx"])
        else:
            start_leg = max(0, peak_idx - 20)
            swing_low_idx = int(window_df["low"].iloc[start_leg:peak_idx].idxmin())
            swing_low_price = float(window_df.loc[swing_low_idx, "low"])

        # 3. Check for Market Structure Shift Breakout (Candle closing below structural swing low)
        post_peak_df = window_df.iloc[peak_idx: -1]  # from peak up to closed bars
        breakout_bars = post_peak_df[post_peak_df["close"] < swing_low_price]
        if breakout_bars.empty:
            return None

        break_idx = int(breakout_bars.index[0])
        break_bar = window_df.iloc[break_idx]

        # ── Aggressive MSS Displacement Filter (Reject Choppy/Drifting Legs) ─
        if require_aggressive_displacement:
            bars_to_break = break_idx - peak_idx
            if bars_to_break > 8:  # Took too long to break (choppy grind)
                return None

            break_body = float(break_bar["open"] - break_bar["close"])
            break_range = float(break_bar["high"] - break_bar["low"])
            body_ratio = break_body / max(break_range, 1e-5)
            if body_ratio < 0.35:  # Weak breakout body (doji/wick heavy)
                return None

            # Check displacement leg directional dominance
            disp_leg = window_df.iloc[peak_idx: break_idx + 1]
            bearish_bars = (disp_leg["close"] < disp_leg["open"]).sum()
            if (bearish_bars / max(1, len(disp_leg))) < 0.50:
                return None

        # Ensure peak high was not breached after the peak
        if window_df["high"].iloc[peak_idx + 1: -1].max() > peak_high:
            return None

        # 4. Check Retest condition
        if require_retest:
            is_retesting = (curr_high >= (swing_low_price - tolerance))
            if not is_retesting:
                return None
        else:
            is_retesting = (curr_close < swing_low_price)

        risk = peak_high - swing_low_price
        if risk < (min_sl_pips * pip_size):
            return None

        return {
            "signal": -1,
            "type": "SELL_MSS_RETEST" if require_retest else "SELL_MSS",
            "entry_price": swing_low_price,
            "trigger_close": curr_close,
            "sl_price": peak_high,
            "tp_price": round(swing_low_price - (2.0 * risk), 5),
            "risk_distance": round(risk, 5),
            "sl_pips": round(risk / pip_size, 1),
            "tp_pips": round((2.0 * risk) / pip_size, 1),
            "peak_high": peak_high,
            "swing_low": swing_low_price,
            "fvg_top": fvg_top,
            "fvg_bottom": fvg_bottom,
            "mss_time": str(curr_bar.get("datetime", "")),
            "retested": True,
        }

    elif trend_side == 1:  # BUY SCENARIO (Mirror)
        # 1. Valley Low in the window
        valley_idx = int(window_df["low"].idxmin())
        valley_low = float(window_df.loc[valley_idx, "low"])

        if valley_idx >= n - 2 or valley_idx < 2:
            return None

        # 2. Find structural swing high before the valley
        pre_valley_df = window_df.iloc[:valley_idx + 1]
        swing_highs, _ = find_swing_pivots(pre_valley_df, left_bars=1, right_bars=1)

        if swing_highs:
            swing_high_price = float(swing_highs[-1]["price"])
            swing_high_idx = int(swing_highs[-1]["idx"])
        else:
            start_leg = max(0, valley_idx - 20)
            swing_high_idx = int(window_df["high"].iloc[start_leg:valley_idx].idxmax())
            swing_high_price = float(window_df.loc[swing_high_idx, "high"])

        # 3. Check for Market Structure Shift Breakout (Candle closing above structural swing high)
        post_valley_df = window_df.iloc[valley_idx: -1]
        breakout_bars = post_valley_df[post_valley_df["close"] > swing_high_price]
        if breakout_bars.empty:
            return None

        break_idx = int(breakout_bars.index[0])
        break_bar = window_df.iloc[break_idx]

        # ── Aggressive MSS Displacement Filter (Reject Choppy/Drifting Legs) ─
        if require_aggressive_displacement:
            bars_to_break = break_idx - valley_idx
            if bars_to_break > 8:  # Took too long to break (choppy grind)
                return None

            break_body = float(break_bar["close"] - break_bar["open"])
            break_range = float(break_bar["high"] - break_bar["low"])
            body_ratio = break_body / max(break_range, 1e-5)
            if body_ratio < 0.35:  # Weak breakout body (doji/wick heavy)
                return None

            # Check displacement leg directional dominance
            disp_leg = window_df.iloc[valley_idx: break_idx + 1]
            bullish_bars = (disp_leg["close"] > disp_leg["open"]).sum()
            if (bullish_bars / max(1, len(disp_leg))) < 0.50:
                return None

        # Ensure valley low was not breached after the valley
        if window_df["low"].iloc[valley_idx + 1: -1].min() < valley_low:
            return None

        # 4. Check Retest condition
        if require_retest:
            is_retesting = (curr_low <= (swing_high_price + tolerance))
            if not is_retesting:
                return None
        else:
            is_retesting = (curr_close > swing_high_price)

        risk = swing_high_price - valley_low
        if risk < (min_sl_pips * pip_size):
            return None

        return {
            "signal": 1,
            "type": "BUY_MSS_RETEST" if require_retest else "BUY_MSS",
            "entry_price": swing_high_price,
            "trigger_close": curr_close,
            "sl_price": valley_low,
            "tp_price": round(swing_high_price + (2.0 * risk), 5),
            "risk_distance": round(risk, 5),
            "sl_pips": round(risk / pip_size, 1),
            "tp_pips": round((2.0 * risk) / pip_size, 1),
            "valley_low": valley_low,
            "swing_high": swing_high_price,
            "fvg_top": fvg_top,
            "fvg_bottom": fvg_bottom,
            "mss_time": str(curr_bar.get("datetime", "")),
            "retested": True,
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
