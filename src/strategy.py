"""
strategy.py

Liquidity Sweep Reversal Strategy with Moving Average Dynamic Target & Filter.

Strategy Rules:
---------------
1. SELL ENTRY (Bearish Liquidity Sweep):
   - Candle 1 (t-1): Bullish candle (close > open).
   - Candle 2 (t): Bearish candle (close < open).
   - Liquidity Taken: Bearish candle's High > Bullish candle's High (swept the top).
   - Close Confirmation: Bearish candle closes strictly below Bullish candle's Open.
   - Trend / MA Filter: Moving Average is BELOW the entry price (MA < close).
   - Stop Loss: 1 pip above the Bear candle's High (for broker spread buffer).
   - Target / Take Profit: 1:2 Risk-to-Reward ratio OR on-time Moving Average price (close when tapping MA).

2. BUY ENTRY (Bullish Liquidity Sweep - Mirror):
   - Candle 1 (t-1): Bearish candle (close < open).
   - Candle 2 (t): Bullish candle (close > open).
   - Liquidity Taken: Bullish candle's Low < Bearish candle's Low (swept the bottom).
   - Close Confirmation: Bullish candle closes strictly above Bearish candle's Open.
   - Trend / MA Filter: Moving Average is ABOVE the entry price (MA > close).
   - Stop Loss: 1 pip below the Bull candle's Low (for broker spread buffer).
   - Target / Take Profit: 1:2 Risk-to-Reward ratio OR on-time Moving Average price (close when tapping MA).

Signal Column Values:
    1 = BUY signal confirmed
   -1 = SELL signal confirmed
    0 = No signal
"""

import os
import sys
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.patterns import is_liquidity_sweep_sell, is_liquidity_sweep_buy
else:
    from .patterns import is_liquidity_sweep_sell, is_liquidity_sweep_buy


# ── Strategy Parameters ────────────────────────────────────────────────────────
MA_PERIOD       = 50       # Fast dynamic target Moving Average
MA_TYPE         = "EMA"    # 'EMA' or 'SMA'
HTF_EMA_PERIOD  = 200      # Higher Timeframe Trend EMA
SWING_WINDOW    = 10       # Lookback window for structural swing prominence
RR_RATIO        = 1.5      # High-win-rate 1:1.5 Risk-to-Reward ratio
PIP_BUFFER      = 1.0      # Minimum pip buffer floor
ATR_BUFFER_MULT = 0.35     # Dynamic ATR stop loss buffer multiplier
DEFAULT_PIP     = 0.01     # Default pip size (0.01 for XAUUSD/JPY, 0.0001 for Forex)

ML_MODEL_PATH           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgb_sweep_model.pkl")
ML_CONFIDENCE_THRESHOLD = 0.55  # Minimum model win probability required to take trade
USE_ML_FILTER           = True  # Auto-filter trades when ML model artifact is present
# ─────────────────────────────────────────────────────────────────────────────


_ml_model_cache = None
_ml_model_loaded = False


def get_ml_model():
    """Lazily load and cache the trained XGBoost model artifact."""
    global _ml_model_cache, _ml_model_loaded
    if _ml_model_loaded:
        return _ml_model_cache

    _ml_model_loaded = True
    if os.path.exists(ML_MODEL_PATH):
        try:
            import joblib
            _ml_model_cache = joblib.load(ML_MODEL_PATH)
            print(f"[Strategy] Loaded XGBoost ML Filter model from {ML_MODEL_PATH}")
        except Exception as exc:
            print(f"[Strategy] Warning: Could not load ML model: {exc}")
            _ml_model_cache = None
    else:
        _ml_model_cache = None
    return _ml_model_cache


def add_ma(df: pd.DataFrame, period: int = MA_PERIOD, ma_type: str = MA_TYPE) -> pd.DataFrame:
    """
    Append a Moving Average column (EMA or SMA) to the dataframe.
    """
    df = df.copy()
    col_name = f"ma_{period}"
    if ma_type.upper() == "EMA":
        df[col_name] = df["close"].ewm(span=period, adjust=False).mean()
    else:
        df[col_name] = df["close"].rolling(window=period).mean()
    return df


def _detect_pip_size(
    df: pd.DataFrame,
    symbol: str | None = None,
    default_pip: float = DEFAULT_PIP,
) -> float:
    """
    Auto-detect pip/point size for US30, Gold, Forex majors (EURUSD, GBPUSD), etc.
    """
    if symbol:
        sym = str(symbol).upper()
        if any(k in sym for k in ("US30", "DJI", "WS30")):
            return 1.0
        elif "XAU" in sym or "XAG" in sym:
            return 0.01
        elif "JPY" in sym:
            return 0.01
        elif any(k in sym for k in ("EUR", "GBP", "AUD", "NZD", "USD")):
            return 0.0001

    if "close" not in df or df.empty:
        return default_pip
    avg_price = df["close"].dropna().iloc[-1] if len(df["close"].dropna()) > 0 else 0
    if avg_price > 10000:  # e.g., US30 / Dow Jones (~35,000-45,000) -> 1.0 point
        return 1.0
    elif avg_price > 500:  # e.g., Gold XAUUSD (~2000-4500) -> 0.01
        return 0.01
    elif avg_price > 50:  # e.g., JPY pairs (~100-160) -> 0.01
        return 0.01
    else:  # e.g., EURUSD (~1.08), GBPUSD (~1.27) -> 0.0001
        return 0.0001


def generate_signals(
    df: pd.DataFrame,
    ma_period: int = MA_PERIOD,
    htf_period: int = HTF_EMA_PERIOD,
    swing_window: int = SWING_WINDOW,
    ma_type: str = MA_TYPE,
    rr_ratio: float = RR_RATIO,
    pip_buffer: float = PIP_BUFFER,
    atr_buffer_mult: float = ATR_BUFFER_MULT,
    pip_size: float | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Run the Liquidity Sweep Reversal strategy with swing prominence and dynamic ATR SL buffer.
    """
    df = add_ma(df, period=ma_period, ma_type=ma_type)
    htf_col = f"ema_{htf_period}"
    if htf_col not in df:
        df[htf_col] = df["close"].ewm(span=htf_period, adjust=False).mean()

    ma_col = f"ma_{ma_period}"

    if pip_size is None:
        pip_size = _detect_pip_size(df, symbol=symbol)

    # Compute ATR-14 if not already present
    if "atr_14" not in df:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=1).mean().fillna(pip_size * 5)

    # Dynamic ATR buffer with floor
    buffer_amount = np.maximum(pip_buffer * pip_size, atr_buffer_mult * df["atr_14"])

    # 1. Pattern Detection with Swing Prominence
    df["sweep_sell"] = is_liquidity_sweep_sell(df, swing_window=swing_window)
    df["sweep_buy"]  = is_liquidity_sweep_buy(df, swing_window=swing_window)

    # 2. Moving Average Conditions:
    # - For SELL: MA-50 must be BELOW entry price (close > MA-50)
    # - For BUY:  MA-50 must be ABOVE entry price (close < MA-50)
    ma_valid_sell = df["close"] > df[ma_col]
    ma_valid_buy  = df["close"] < df[ma_col]

    conditions_sell = df["sweep_sell"] & ma_valid_sell
    conditions_buy  = df["sweep_buy"] & ma_valid_buy

    df["signal"] = 0
    df.loc[conditions_buy,  "signal"] =  1
    df.loc[conditions_sell, "signal"] = -1

    # Initialize SL / TP columns
    df["entry_price"]   = df["close"]
    df["target_ma"]     = df[ma_col]
    df["htf_trend"]     = (df["close"] > df[htf_col]).map({True: "UP", False: "DOWN"})
    df["sl_price"]      = np.nan
    df["tp_price"]      = np.nan
    df["risk_distance"] = np.nan
    df["sl_pips"]       = np.nan
    df["tp_pips"]       = np.nan

    # ── Sell Signal Calculations ───────────────────────────────────────────────
    sell_mask = df["signal"] == -1
    if sell_mask.any():
        sl_sell = df.loc[sell_mask, "high"] + buffer_amount[sell_mask]
        risk_sell = sl_sell - df.loc[sell_mask, "close"]
        tp_sell = df.loc[sell_mask, "close"] - (risk_sell * rr_ratio)

        df.loc[sell_mask, "sl_price"]      = sl_sell
        df.loc[sell_mask, "tp_price"]      = tp_sell
        df.loc[sell_mask, "risk_distance"] = risk_sell
        df.loc[sell_mask, "sl_pips"]       = risk_sell / pip_size
        df.loc[sell_mask, "tp_pips"]       = (risk_sell * rr_ratio) / pip_size

    # ── Buy Signal Calculations ────────────────────────────────────────────────
    buy_mask = df["signal"] == 1
    if buy_mask.any():
        sl_buy = df.loc[buy_mask, "low"] - buffer_amount[buy_mask]
        risk_buy = df.loc[buy_mask, "close"] - sl_buy
        tp_buy = df.loc[buy_mask, "close"] + (risk_buy * rr_ratio)

        df.loc[buy_mask, "sl_price"]      = sl_buy
        df.loc[buy_mask, "tp_price"]      = tp_buy
        df.loc[buy_mask, "risk_distance"] = risk_buy
        df.loc[buy_mask, "sl_pips"]       = risk_buy / pip_size
        df.loc[buy_mask, "tp_pips"]       = (risk_buy * rr_ratio) / pip_size

    # Human-readable labels (Fixed 1:1.5 RR)
    label_map = {
        1: f"BUY Liquidity Sweep (1:{rr_ratio:.1f} RR)",
        -1: f"SELL Liquidity Sweep (1:{rr_ratio:.1f} RR)",
        0: "",
    }
    df["signal_label"] = df["signal"].map(label_map)

    return df




def get_latest_signal(
    df: pd.DataFrame,
    ma_period: int = MA_PERIOD,
    ma_type: str = MA_TYPE,
    rr_ratio: float = RR_RATIO,
    pip_buffer: float = PIP_BUFFER,
    pip_size: float | None = None,
    use_ml: bool = USE_ML_FILTER,
    confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Return the signal evaluated on the most recently closed candle (second to last row,
    as the last bar is actively forming in real time), with optional XGBoost confidence filter.
    """
    if len(df) < ma_period + 2:
        return {"signal": 0, "reason": "Not enough bars to compute Moving Average"}

    df = generate_signals(
        df,
        ma_period=ma_period,
        ma_type=ma_type,
        rr_ratio=rr_ratio,
        pip_buffer=pip_buffer,
        pip_size=pip_size,
    )
    
    # Use the last fully closed candle (iloc[-2])
    bar = df.iloc[-2]
    prev_bar = df.iloc[-3]
    ma_col = f"ma_{ma_period}"

    sig = int(bar["signal"])
    ma_val = float(bar[ma_col]) if not pd.isna(bar[ma_col]) else None

    result = {
        "datetime":      bar["datetime"],
        "signal":        sig,
        "label":         bar["signal_label"],
        "close":         float(bar["close"]),
        "open":          float(bar["open"]),
        "high":          float(bar["high"]),
        "low":           float(bar["low"]),
        "prev_open":     float(prev_bar["open"]),
        "prev_high":     float(prev_bar["high"]),
        "prev_low":      float(prev_bar["low"]),
        "prev_close":    float(prev_bar["close"]),
        "ma":            round(ma_val, 5) if ma_val is not None else None,
        f"ema_{ma_period}": round(ma_val, 5) if ma_val is not None else None,
        "entry_price":   float(bar["entry_price"]),
        "sl_price":      round(float(bar["sl_price"]), 5) if not pd.isna(bar["sl_price"]) else None,
        "tp_price":      round(float(bar["tp_price"]), 5) if not pd.isna(bar["tp_price"]) else None,
        "target_ma":     round(ma_val, 5) if ma_val is not None else None,
        "sl_pips":       round(float(bar["sl_pips"]), 2) if not pd.isna(bar["sl_pips"]) else None,
        "tp_pips":       round(float(bar["tp_pips"]), 2) if not pd.isna(bar["tp_pips"]) else None,
        "risk_distance": round(float(bar["risk_distance"]), 5) if not pd.isna(bar["risk_distance"]) else None,
        "sweep_sell":    bool(bar["sweep_sell"]),
        "sweep_buy":     bool(bar["sweep_buy"]),
        "xgb_prob":      None,
        "ml_filtered":   False,
    }

    # ── XGBoost Meta-Labeling Inference & Filter ──────────────────────────────
    if sig != 0 and use_ml:
        model = get_ml_model()
        if model is not None:
            try:
                if __package__ is None or __package__ == "":
                    from src.ml_features import extract_features_for_signal, FEATURE_COLUMNS
                else:
                    from .ml_features import extract_features_for_signal, FEATURE_COLUMNS

                feats = extract_features_for_signal(
                    df=df,
                    idx=len(df) - 2,
                    signal_side=sig,
                    pip_size=pip_size or _detect_pip_size(df),
                    ma_period=ma_period,
                )
                feats_df = pd.DataFrame([[feats[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                prob = float(model.predict_proba(feats_df)[0][1])
                result["xgb_prob"] = round(prob, 4)

                if prob < confidence_threshold:
                    result["ml_filtered"] = True
                    result["signal"] = 0
                    result["label"] = f"FILTERED by ML (Win Prob: {prob*100:.1f}% < {confidence_threshold*100:.0f}%)"
                else:
                    result["ml_filtered"] = False
                    result["label"] += f" | [ML Conf: {prob*100:.1f}%]"
            except Exception as exc:
                print(f"[Strategy] Warning during ML inference: {exc}")

    return result

