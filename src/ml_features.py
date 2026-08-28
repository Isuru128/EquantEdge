"""
ml_features.py

Feature engineering module for 15M FVG + 1M MSS Strategy ML Meta-Labeling.
Extracts institutional SMC features: Liquidity Sweeps, Killzone timing, FVG penetration depth,
displacement momentum, volume surge, candlestick anatomy, and session microstructure.
"""

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

_NY_TZ = ZoneInfo("America/New_York")

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
    "disp_volume_surge",    # Volume surge on breakout candle
    "is_liquidity_sweep",   # 1 if setup swept prior local/session liquidity, 0 otherwise
    "is_killzone",          # 1 if in London Open (02-05 NY) or NY AM (07-10 NY), 0 otherwise
    "fvg_penetration_pct",  # Percentage depth penetration into 15M FVG
    "hour",                 # Hour of day (0-23 UTC)
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


def _is_killzone_time(dt: pd.Timestamp) -> int:
    """Return 1 if within London Open (02-05 NY) or NY AM (07-10 NY) Killzones."""
    try:
        if dt.tzinfo is None:
            dt_ny = dt.tz_localize("UTC").astimezone(_NY_TZ)
        else:
            dt_ny = dt.astimezone(_NY_TZ)
        hour = dt_ny.hour
        # London Open: 02:00 - 05:00 NY | NY AM: 07:00 - 10:00 NY
        if (2 <= hour < 5) or (7 <= hour < 10):
            return 1
        return 0
    except Exception:
        return 0


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
    fvg: dict | None = None,
    peak_high: float | None = None,
    valley_low: float | None = None,
) -> dict:
    """
    Extract a comprehensive dictionary of ML features for a trade setup triggered at index `idx`.
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

    # 1. Liquidity Sweep Detection
    start_lookback = max(0, idx - 25)
    lookback_window = df.iloc[start_lookback:idx]
    is_sweep = 0
    if not lookback_window.empty:
        if signal_side == -1:  # SELL (Check if peak swept previous high)
            high_level = peak_high if peak_high is not None else curr["high"]
            prior_max_high = lookback_window["high"].iloc[:-3].max() if len(lookback_window) > 3 else lookback_window["high"].max()
            if high_level >= prior_max_high:
                is_sweep = 1
        elif signal_side == 1:  # BUY (Check if valley swept previous low)
            low_level = valley_low if valley_low is not None else curr["low"]
            prior_min_low = lookback_window["low"].iloc[:-3].min() if len(lookback_window) > 3 else lookback_window["low"].min()
            if low_level <= prior_min_low:
                is_sweep = 1

    # 2. FVG Penetration Depth %
    fvg_depth_pct = 0.50
    if fvg and "top" in fvg and "bottom" in fvg:
        fvg_size = max(1e-5, fvg["top"] - fvg["bottom"])
        if signal_side == -1 and peak_high is not None:
            fvg_depth_pct = min(1.0, max(0.0, (peak_high - fvg["bottom"]) / fvg_size))
        elif signal_side == 1 and valley_low is not None:
            fvg_depth_pct = min(1.0, max(0.0, (fvg["top"] - valley_low) / fvg_size))

    # 3. Volume Surge
    vol_ratio = float(curr.get("volume_ratio", 1.0))
    disp_surge = float(prev.get("volume_ratio", vol_ratio))

    features = {
        "signal_side":          int(signal_side),
        "body_to_range":        float(round(c_body / c_range, 4)),
        "break_displacement":   float(round(break_disp, 2)),
        "rejection_wick_ratio": float(round(rejection_wick / c_range, 4)),
        "risk_pips":            float(round(c_range / pip_size, 1)),
        "dist_to_15m_ema50":    float(round(dist_to_ema, 4)),
        "rsi_14":               float(round(curr.get("rsi_14", 50.0), 2)),
        "atr_14":               float(round(atr, 5)),
        "volume_ratio":         float(round(vol_ratio, 3)),
        "disp_volume_surge":    float(round(disp_surge, 3)),
        "is_liquidity_sweep":   int(is_sweep),
        "is_killzone":          int(_is_killzone_time(dt)),
        "fvg_penetration_pct":  float(round(fvg_depth_pct, 4)),
        "hour":                 int(dt.hour),
        "minute":               int(dt.minute),
        "session_code":         int(_get_session_code(dt)),
    }
    return features
