"""
ml_features.py

Feature engineering module for 15M FVG + 1M MSS Strategy ML Meta-Labeling.
Extracts market microstructure, displacement velocity, FVG depth, candlestick anatomy, and session context.
"""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "signal_side",          # 1 for BUY, -1 for SELL
    "body_to_range",        # Ratio of confirmation candle body to total range
    "break_displacement",   # Distance the MSS candle closed beyond the broken swing level
    "rejection_wick_ratio", # Ratio of rejection wick to candle range
    "risk_pips",            # Risk distance (entry to SL) in pips
    "dist_to_15m_ema50",    # Distance to 15M EMA-50 normalized by ATR
    "rsi_14",               # 1M RSI (14)
    "atr_14",               # 1M Average True Range (14)
    "volume_ratio",         # Volume vs 20-period moving average
    "hour",                 # Hour of day (0-23)
    "minute",               # Minute of hour (0-59)
    "session_code",         # 0=Asia, 1=London, 2=NY, 3=Other
]


def compute_technical_indicators(
    df: pd.DataFrame,
    fast_ma: int = 50,
    slow_ma: int = 200,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Compute technical indicators on DataFrame.
    """
    df = df.copy()

    # Moving Averages
    fast_col = f"ema_{fast_ma}"
    slow_col = f"ema_{slow_ma}"

    if fast_col not in df:
        df[fast_col] = df["close"].ewm(span=fast_ma, adjust=False).mean()
    if slow_col not in df:
        df[slow_col] = df["close"].ewm(span=slow_ma, adjust=False).mean()

    # Average True Range (ATR-14)
    if "atr_14" not in df:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0005)

    # Relative Strength Index (RSI-14)
    if "rsi_14" not in df:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = ((-delta.where(delta < 0, 0.0))).rolling(window=rsi_period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Rolling Volume Ratio
    if "volume" in df:
        vol_sma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = (df["volume"] / (vol_sma_20 + 1e-6)).fillna(1.0)
    else:
        df["volume_ratio"] = 1.0

    return df


def _get_session_code(dt: pd.Timestamp) -> int:
    """Classify UTC hour into sessions: Asia=0, London=1, NY=2, Other=3."""
    hour = dt.hour
    if 0 <= hour < 7:
        return 0
    elif 7 <= hour < 12:
        return 1
    elif 12 <= hour < 20:
        return 2
    else:
        return 3


def extract_features_for_signal(
    df: pd.DataFrame,
    idx: int,
    signal_side: int,
    pip_size: float = 0.0001,
    htf_trend_dir: int = 1,
) -> dict:
    """
    Extract a dictionary of ML features for a trade setup triggered at index `idx`.
    """
    if "rsi_14" not in df or "atr_14" not in df:
        df = compute_technical_indicators(df)

    curr = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else curr

    dt = pd.to_datetime(curr["datetime"]) if "datetime" in curr and not pd.isna(curr["datetime"]) else pd.Timestamp.now()

    c_range = max(1e-6, curr["high"] - curr["low"])
    c_body = abs(curr["close"] - curr["open"])

    body_top = max(curr["open"], curr["close"])
    body_bottom = min(curr["open"], curr["close"])
    upper_wick = curr["high"] - body_top
    lower_wick = body_bottom - curr["low"]

    rejection_wick = upper_wick if signal_side == -1 else lower_wick
    break_disp = abs(curr["close"] - prev["close"]) / pip_size
    atr = float(curr.get("atr_14", 0.0005))
    ema_val = float(curr.get("ema_50", curr["close"]))
    dist_to_ema = (curr["close"] - ema_val) / (atr + 1e-6)

    features = {
        "signal_side":          int(signal_side),
        "htf_trend_dir":        int(htf_trend_dir),
        "body_to_range":        float(round(c_body / c_range, 4)),
        "break_displacement":   float(round(break_disp, 2)),
        "rejection_wick_ratio": float(round(rejection_wick / c_range, 4)),
        "risk_pips":            float(round(c_range / pip_size, 1)),
        "dist_to_15m_ema50":    float(round(dist_to_ema, 4)),
        "rsi_14":               float(round(curr.get("rsi_14", 50.0), 2)),
        "atr_14":               float(round(atr, 5)),
        "volume_ratio":         float(round(curr.get("volume_ratio", 1.0), 3)),
        "hour":                 int(dt.hour),
        "minute":               int(dt.minute),
        "session_code":         int(_get_session_code(dt)),
    }
    return features
