"""
ml_features.py

Feature engineering module for the Liquidity Sweep XGBoost ML model.
Extracts candlestick anatomy, volatility, momentum, and context indicators.
"""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "signal_side",          # 1 for BUY, -1 for SELL
    "sweep_depth_pips",     # Distance swept past prior candle's high/low
    "rejection_ratio",      # Ratio of close breakout beyond prior open to candle range
    "wick_body_ratio",      # Rejection wick vs body size
    "body_to_range",        # Body ratio of confirmation candle
    "prev_body_to_range",   # Body ratio of candle 1
    "distance_to_ma_atr",   # Distance to M3 MA-50 normalized by ATR
    "ma_slope_5",           # Rate of change of MA over 5 bars normalized by ATR
    "is_swing_peak",        # 1 if Candle 1 was a 15-bar local high/low (prominent swing level)
    "swing_prominence_pips",# Height/depth of Candle 1 above/below rolling median
    "rsi_14",               # RSI (14)
    "rsi_divergence",       # 1 if RSI divergence is present (exhaustion), else 0
    "htf_trend",            # 1 if close > HTF EMA-200, -1 if close < HTF EMA-200
    "htf_dist_atr",         # Distance to HTF EMA-200 normalized by ATR
    "atr_14",               # ATR (14)
    "volume_ratio",         # Volume vs 20-period average volume
    "hour",                 # Hour of day (0-23)
    "minute",               # Minute of hour (0-59)
    "session_code",         # 0=Asia, 1=London, 2=NY, 3=Other
]


def compute_technical_indicators(df: pd.DataFrame, ma_period: int = 50, htf_period: int = 200) -> pd.DataFrame:
    """
    Compute core technical indicators (ATR, RSI, M3 MA, HTF EMA, Rolling Volume) on DataFrame.
    """
    df = df.copy()

    # 1. Base Moving Average
    ma_col = f"ma_{ma_period}"
    if ma_col not in df:
        df[ma_col] = df["close"].ewm(span=ma_period, adjust=False).mean()

    # 2. Higher Timeframe EMA (e.g. 200 bars for HTF trend bias)
    htf_col = f"ema_{htf_period}"
    if htf_col not in df:
        df[htf_col] = df["close"].ewm(span=htf_period, adjust=False).mean()

    # 3. Average True Range (ATR-14)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # 4. Relative Strength Index (RSI-14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
    loss = ((-delta.where(delta < 0, 0.0))).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    # 5. Normalized MA Slope over 5 bars
    ma_diff_5 = df[ma_col] - df[ma_col].shift(5)
    df["ma_slope_5"] = (ma_diff_5 / (5 * df["atr_14"] + 1e-6)).fillna(0.0)

    # 6. Rolling Volume Ratio
    vol_sma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = (df["volume"] / (vol_sma_20 + 1e-6)).fillna(1.0)

    # 7. Candlestick Anatomy
    df["candle_range"] = (df["high"] - df["low"]).replace(0, 1e-6)
    df["body_size"] = (df["close"] - df["open"]).abs()
    
    # Upper and lower wicks
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"] - body_top
    df["lower_wick"] = body_bottom - df["low"]

    return df


def _get_session_code(dt: pd.Timestamp) -> int:
    """Classify UTC hour into approximate sessions: Asia=0, London=1, NY=2, Other=3."""
    hour = dt.hour
    if 0 <= hour < 7:
        return 0  # Asia
    elif 7 <= hour < 12:
        return 1  # London
    elif 12 <= hour < 20:
        return 2  # New York
    else:
        return 3  # Other


def extract_features_for_signal(
    df: pd.DataFrame,
    idx: int,
    signal_side: int,
    pip_size: float = 0.01,
    ma_period: int = 50,
    htf_period: int = 200,
) -> dict:
    """
    Extract a single dictionary of engineered ML features for a confirmed setup at index `idx`.
    """
    ma_col = f"ma_{ma_period}"
    htf_col = f"ema_{htf_period}"
    if "rsi_14" not in df or "atr_14" not in df or ma_col not in df or htf_col not in df:
        df = compute_technical_indicators(df, ma_period=ma_period, htf_period=htf_period)


    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]

    dt = pd.to_datetime(curr["datetime"]) if "datetime" in curr else pd.Timestamp.now()

    # Candlestick anatomy
    c_range = max(1e-6, curr["high"] - curr["low"])
    p_range = max(1e-6, prev["high"] - prev["low"])
    c_body = abs(curr["close"] - curr["open"])
    p_body = abs(prev["close"] - prev["open"])

    body_to_range = c_body / c_range
    prev_body_to_range = p_body / p_range

    atr = float(curr["atr_14"]) if not pd.isna(curr["atr_14"]) else 0.5
    ma_val = float(curr[ma_col]) if not pd.isna(curr[ma_col]) else float(curr["close"])
    htf_val = float(curr[htf_col]) if not pd.isna(curr[htf_col]) else float(curr["close"])
    
    dist_to_ma_atr = abs(curr["close"] - ma_val) / (atr + 1e-6)
    htf_dist_atr = (curr["close"] - htf_val) / (atr + 1e-6)
    htf_trend = 1 if curr["close"] >= htf_val else -1

    # 15-bar local swing prominence
    lookback = min(idx - 1, 15)
    if lookback > 2:
        rolling_highs = df["high"].iloc[idx - 1 - lookback : idx - 1]
        rolling_lows  = df["low"].iloc[idx - 1 - lookback : idx - 1]
        max_prior_high = rolling_highs.max()
        min_prior_low  = rolling_lows.min()
        median_prior_high = rolling_highs.median()
        median_prior_low  = rolling_lows.median()
    else:
        max_prior_high = prev["high"]
        min_prior_low  = prev["low"]
        median_prior_high = prev["high"]
        median_prior_low  = prev["low"]

    if signal_side == -1:  # SELL
        sweep_depth = (curr["high"] - prev["high"]) / pip_size
        rejection_ratio = (prev["open"] - curr["close"]) / c_range
        c_upper_wick = curr["high"] - max(curr["open"], curr["close"])
        wick_body_ratio = c_upper_wick / (c_body + 1e-6)
        is_swing_peak = 1 if prev["high"] >= (max_prior_high - 1e-6) else 0
        swing_prominence_pips = (prev["high"] - median_prior_high) / pip_size
        
        # Bearish RSI divergence (RSI < 70 and declining or lower than prior peak)
        prior_rsi_max = df["rsi_14"].iloc[max(0, idx - 10) : idx - 1].max() if idx > 5 else 50.0
        rsi_divergence = 1 if (curr["high"] > prev["high"] and curr.get("rsi_14", 50) < prior_rsi_max) else 0
    else:                  # BUY
        sweep_depth = (prev["low"] - curr["low"]) / pip_size
        rejection_ratio = (curr["close"] - prev["open"]) / c_range
        c_lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        wick_body_ratio = c_lower_wick / (c_body + 1e-6)
        is_swing_peak = 1 if prev["low"] <= (min_prior_low + 1e-6) else 0
        swing_prominence_pips = (median_prior_low - prev["low"]) / pip_size
        
        # Bullish RSI divergence
        prior_rsi_min = df["rsi_14"].iloc[max(0, idx - 10) : idx - 1].min() if idx > 5 else 50.0
        rsi_divergence = 1 if (curr["low"] < prev["low"] and curr.get("rsi_14", 50) > prior_rsi_min) else 0

    features = {
        "signal_side":           int(signal_side),
        "sweep_depth_pips":      float(round(sweep_depth, 2)),
        "rejection_ratio":       float(round(rejection_ratio, 4)),
        "wick_body_ratio":       float(round(wick_body_ratio, 4)),
        "body_to_range":         float(round(body_to_range, 4)),
        "prev_body_to_range":    float(round(prev_body_to_range, 4)),
        "distance_to_ma_atr":    float(round(dist_to_ma_atr, 4)),
        "ma_slope_5":            float(round(curr.get("ma_slope_5", 0.0), 4)),
        "is_swing_peak":         int(is_swing_peak),
        "swing_prominence_pips": float(round(swing_prominence_pips, 2)),
        "rsi_14":                float(round(curr.get("rsi_14", 50.0), 2)),
        "rsi_divergence":        int(rsi_divergence),
        "htf_trend":             int(htf_trend),
        "htf_dist_atr":          float(round(htf_dist_atr, 4)),
        "atr_14":                float(round(atr, 4)),
        "volume_ratio":          float(round(curr.get("volume_ratio", 1.0), 3)),
        "hour":                  int(dt.hour),
        "minute":                int(dt.minute),
        "session_code":          int(_get_session_code(dt)),
    }
    return features

