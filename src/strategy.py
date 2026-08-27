"""
strategy.py

15-Minute Fair Value Gap (FVG) + 1-Minute Market Structure Shift (MSS) Trading Strategy.

Strategy Blueprint:
-------------------
1. 15-Minute Fair Value Gaps (FVG):
   - Bearish FVG (for Sell): 3-bar sequence on M15 where Candle 1 Low > Candle 3 High.
   - Bullish FVG (for Buy): 3-bar sequence on M15 where Candle 1 High < Candle 3 Low.
   - Max Lookback: Up to 96 15M candles (24 hours).

2. 1-Minute Market Structure Shift (MSS) Trigger:
   - SELL SCENARIO:
     - Price enters/touches an active 15M Bearish FVG zone.
     - On M1, price prints a bullish leg establishing a swing peak (high of last bullish leg).
     - A 1-minute candle closes strictly BELOW the lowest price of the last bullish leg (Market Structure Shift).
     - Order Entry: Limit order / level at the broken swing low (shift price).
     - Stop Loss: Swing high of the last bullish leg (peak high).
     - Take Profit: Exact 1:2.0 Risk-to-Reward ratio (2.0 * Risk).
     - Risk: 1.0% account balance per trade.

   - BUY SCENARIO (Mirror):
     - Price enters/touches an active 15M Bullish FVG zone.
     - On M1, price prints a bearish leg establishing a swing valley (low of last bearish leg).
     - A 1-minute candle closes strictly ABOVE the highest price of the last bearish leg (Market Structure Shift).
     - Order Entry: Limit order / level at the broken swing high (shift price).
     - Stop Loss: Swing low of the last bearish leg (valley low).
     - Take Profit: Exact 1:2.0 Risk-to-Reward ratio (2.0 * Risk).
     - Risk: 1.0% account balance per trade.

3. Target Assets: EURUSD, GBPUSD, AUDUSD.
4. Trading Session Restriction: No trades executed between 14:00 - 20:00 NY time (UTC-4 / UTC-5).
"""

import os
import sys
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.patterns import find_15m_fvgs, get_nearest_active_fvg, detect_market_structure_shift
else:
    from .patterns import find_15m_fvgs, get_nearest_active_fvg, detect_market_structure_shift


# ── Strategy Parameters ────────────────────────────────────────────────────────
STRATEGY_NAME           = "15M FVG + 1M Market Structure Shift (MSS)"
SUPPORTED_SYMBOLS       = ["EURUSD", "GBPUSD", "AUDUSD"]
MAX_FVG_LOOKBACK_BARS   = 96        # Max 96 15M candles (24 hours lookback for FVG)
RR_RATIO                = 2.0       # Institutional 1:2.0 Risk-to-Reward ratio
RISK_PERCENT            = 1.0       # 1.0% risk per trade
DEFAULT_PIP             = 0.0001    # 0.0001 for EURUSD, GBPUSD, AUDUSD

ML_MODEL_PATH           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgb_strategy_model.pkl")
ML_CONFIDENCE_THRESHOLD = 0.50
USE_ML_FILTER           = True
# ─────────────────────────────────────────────────────────────────────────────


_ml_model_cache = None
_ml_model_loaded = False


def get_ml_model():
    """Lazily load and cache the trained ML filter artifact."""
    global _ml_model_cache, _ml_model_loaded
    if _ml_model_loaded:
        return _ml_model_cache

    _ml_model_loaded = True
    if os.path.exists(ML_MODEL_PATH):
        try:
            import joblib
            _ml_model_cache = joblib.load(ML_MODEL_PATH)
            print(f"[Strategy] Loaded ML Filter model from {ML_MODEL_PATH}")
        except Exception as exc:
            print(f"[Strategy] Warning: Could not load ML model: {exc}")
            _ml_model_cache = None
    else:
        _ml_model_cache = None
    return _ml_model_cache


def _detect_pip_size(
    df: pd.DataFrame | None = None,
    symbol: str | None = None,
    default_pip: float = DEFAULT_PIP,
) -> float:
    """Auto-detect pip size for FX pairs (EURUSD, GBPUSD, AUDUSD = 0.0001)."""
    if symbol:
        sym = str(symbol).upper()
        if any(k in sym for k in ("US30", "DJI", "WS30")):
            return 1.0
        elif "XAU" in sym or "XAG" in sym or "JPY" in sym:
            return 0.01
        elif any(k in sym for k in ("EUR", "GBP", "AUD", "NZD", "USD", "CHF", "CAD")):
            return 0.0001

    if df is not None and "close" in df and not df.empty:
        avg_price = df["close"].dropna().iloc[-1] if len(df["close"].dropna()) > 0 else 0
        if avg_price > 10000:
            return 1.0
        elif avg_price > 50:
            return 0.01
        else:
            return 0.0001

    return default_pip


def resample_1m_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLC DataFrame into 15-minute candles.
    """
    if df_1m.empty:
        return pd.DataFrame()

    df = df_1m.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg_dict["volume"] = "sum"

    df_15m = df.resample("15min").agg(agg_dict).dropna().reset_index()
    return df_15m


def generate_signals(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    symbol: str | None = None,
    rr_ratio: float = RR_RATIO,
    pip_size: float | None = None,
    max_fvg_lookback: int = MAX_FVG_LOOKBACK_BARS,
) -> pd.DataFrame:
    """
    Scan historical DataFrame and generate 15M FVG + 1M MSS signals across all candles.
    """
    if pip_size is None:
        pip_size = _detect_pip_size(df_1m, symbol=symbol)

    if df_15m is None or df_15m.empty:
        df_15m = resample_1m_to_15m(df_1m)

    df_out = df_1m.copy()
    df_out["signal"] = 0
    df_out["entry_price"] = np.nan
    df_out["sl_price"] = np.nan
    df_out["tp_price"] = np.nan
    df_out["risk_distance"] = np.nan
    df_out["sl_pips"] = np.nan
    df_out["tp_pips"] = np.nan
    df_out["signal_label"] = ""
    df_out["fvg_top"] = np.nan
    df_out["fvg_bottom"] = np.nan

    n_1m = len(df_1m)
    if len(df_15m) < 3 or n_1m < 15:
        return df_out

    # 1. Precompute all 15M FVGs
    all_fvgs = find_15m_fvgs(df_15m, max_lookback_bars=len(df_15m))
    if not all_fvgs:
        return df_out

    # Build FVG lookup list with timestamp or index for fast evaluation
    # For each 1M bar, check if price interacts with any active 15M FVG (within 96 M15 bars = 1440 M1 bars)
    # Group FVGs into Bearish & Bullish
    bearish_fvgs = [f for f in all_fvgs if f["direction"] == -1]
    # Extract NumPy arrays for ultra-fast vectorized scanning
    highs = df_1m["high"].to_numpy(dtype=float)
    lows = df_1m["low"].to_numpy(dtype=float)
    closes = df_1m["close"].to_numpy(dtype=float)
    datetimes = df_1m["datetime"].astype(str).values if "datetime" in df_1m.columns else None

    bearish_fvgs = [f for f in all_fvgs if f["direction"] == -1]
    bullish_fvgs = [f for f in all_fvgs if f["direction"] == 1]

    lookback = 25
    cooldown_until = 0

    signals = np.zeros(n_1m, dtype=int)
    entry_prices = np.full(n_1m, np.nan)
    sl_prices = np.full(n_1m, np.nan)
    tp_prices = np.full(n_1m, np.nan)
    risk_distances = np.full(n_1m, np.nan)
    sl_pips_arr = np.full(n_1m, np.nan)
    tp_pips_arr = np.full(n_1m, np.nan)
    fvg_tops = np.full(n_1m, np.nan)
    fvg_bots = np.full(n_1m, np.nan)
    labels = [""] * n_1m

    for i in range(lookback, n_1m):
        if i < cooldown_until:
            continue

        bar_time = datetimes[i] if datetimes is not None else None
        w_h = highs[i - lookback: i + 1]
        w_l = lows[i - lookback: i + 1]
        w_c = closes[i - lookback: i + 1]
        w_h_max = np.max(w_h)
        w_l_min = np.min(w_l)

        # ── Check Bearish FVGs (Sell Setup) ──────────────────────────────────
        for fvg in bearish_fvgs:
            if bar_time and fvg.get("datetime") and fvg["datetime"] >= bar_time:
                continue

            if w_h_max >= fvg["bottom"] and w_l_min <= fvg["top"]:
                p_idx = int(np.argmax(w_h))
                if 1 <= p_idx <= len(w_h) - 3:
                    sw_l = float(np.min(w_l[max(0, p_idx - 4): p_idx]))
                    peak_h = float(w_h[p_idx])
                    curr_c = float(w_c[-2])
                    prior_held = float(np.max(w_c[p_idx: -2])) >= (sw_l - (0.5 * pip_size))
                    
                    if curr_c < sw_l and prior_held:
                        risk = peak_h - sw_l
                        if risk > (0.5 * pip_size):
                            signals[i] = -1
                            entry_prices[i] = sw_l
                            sl_prices[i] = peak_h
                            tp_prices[i] = sw_l - (rr_ratio * risk)
                            risk_distances[i] = risk
                            sl_pips_arr[i] = risk / pip_size
                            tp_pips_arr[i] = (rr_ratio * risk) / pip_size
                            fvg_tops[i] = fvg["top"]
                            fvg_bots[i] = fvg["bottom"]
                            labels[i] = f"SELL MSS (15M FVG + 1M Shift | 1:{rr_ratio:.1f} RR)"
                            cooldown_until = i + 15
                            break

        if signals[i] != 0:
            continue

        # ── Check Bullish FVGs (Buy Setup) ───────────────────────────────────
        for fvg in bullish_fvgs:
            if bar_time and fvg.get("datetime") and fvg["datetime"] >= bar_time:
                continue

            if w_l_min <= fvg["top"] and w_h_max >= fvg["bottom"]:
                v_idx = int(np.argmin(w_l))
                if 1 <= v_idx <= len(w_l) - 3:
                    sw_h = float(np.max(w_h[max(0, v_idx - 4): v_idx]))
                    valley_l = float(w_l[v_idx])
                    curr_c = float(w_c[-2])
                    prior_held = float(np.min(w_c[v_idx: -2])) <= (sw_h + (0.5 * pip_size))

                    if curr_c > sw_h and prior_held:
                        risk = sw_h - valley_l
                        if risk > (0.5 * pip_size):
                            signals[i] = 1
                            entry_prices[i] = sw_h
                            sl_prices[i] = valley_l
                            tp_prices[i] = sw_h + (rr_ratio * risk)
                            risk_distances[i] = risk
                            sl_pips_arr[i] = risk / pip_size
                            tp_pips_arr[i] = (rr_ratio * risk) / pip_size
                            fvg_tops[i] = fvg["top"]
                            fvg_bots[i] = fvg["bottom"]
                            labels[i] = f"BUY MSS (15M FVG + 1M Shift | 1:{rr_ratio:.1f} RR)"
                            cooldown_until = i + 15
                            break

    df_out["signal"] = signals
    df_out["entry_price"] = entry_prices
    df_out["sl_price"] = sl_prices
    df_out["tp_price"] = tp_prices
    df_out["risk_distance"] = risk_distances
    df_out["sl_pips"] = sl_pips_arr
    df_out["tp_pips"] = tp_pips_arr
    df_out["fvg_top"] = fvg_tops
    df_out["fvg_bottom"] = fvg_bots
    df_out["signal_label"] = labels

    return df_out


def get_latest_signal(
    df: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    symbol: str | None = None,
    rr_ratio: float = RR_RATIO,
    pip_size: float | None = None,
    use_ml: bool = USE_ML_FILTER,
    confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Evaluate the most recently closed 1-minute candle for a confirmed 15M FVG + 1M MSS setup.
    """
    if len(df) < 15:
        return {"signal": 0, "reason": "Not enough 1M candles (need >= 15)"}

    if pip_size is None:
        pip_size = _detect_pip_size(df, symbol=symbol)

    if df_15m is None or df_15m.empty:
        df_15m = resample_1m_to_15m(df)

    fvgs = find_15m_fvgs(df_15m, max_lookback_bars=MAX_FVG_LOOKBACK_BARS)
    current_price = float(df.iloc[-2]["close"])
    nearest_fvg = get_nearest_active_fvg(current_price, fvgs)

    bar = df.iloc[-2]
    prev_bar = df.iloc[-3] if len(df) >= 3 else bar

    result = {
        "datetime": bar["datetime"] if "datetime" in bar else None,
        "signal": 0,
        "label": "",
        "close": float(bar["close"]),
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "prev_open": float(prev_bar["open"]),
        "prev_high": float(prev_bar["high"]),
        "prev_low": float(prev_bar["low"]),
        "prev_close": float(prev_bar["close"]),
        "fvg_active": nearest_fvg is not None,
        "fvg_top": nearest_fvg["top"] if nearest_fvg else None,
        "fvg_bottom": nearest_fvg["bottom"] if nearest_fvg else None,
        "fvg_type": nearest_fvg["type"] if nearest_fvg else None,
        "entry_price": float(bar["close"]),
        "sl_price": None,
        "tp_price": None,
        "risk_distance": None,
        "sl_pips": None,
        "tp_pips": None,
        "xgb_prob": None,
        "ml_filtered": False,
    }

    if not fvgs:
        return result

    # Check both Bearish and Bullish active FVGs
    for fvg in fvgs[:10]:
        side = fvg["direction"]
        mss = detect_market_structure_shift(
            df_1m=df,
            fvg=fvg,
            trend_side=side,
            pip_size=pip_size,
        )

        if mss is not None:
            sig = mss["signal"]
            result["signal"] = sig
            result["entry_price"] = round(mss["entry_price"], 5)
            result["sl_price"] = round(mss["sl_price"], 5)
            result["tp_price"] = round(mss["tp_price"], 5)
            result["risk_distance"] = round(mss["risk_distance"], 5)
            result["sl_pips"] = round(mss["sl_pips"], 1)
            result["tp_pips"] = round(mss["tp_pips"], 1)
            result["fvg_top"] = fvg["top"]
            result["fvg_bottom"] = fvg["bottom"]
            result["fvg_type"] = fvg["type"]

            dir_name = "BUY" if sig == 1 else "SELL"
            result["label"] = f"{dir_name} MSS (15M FVG + 1M Shift | 1:{rr_ratio:.1f} RR)"

            if use_ml:
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
                            pip_size=pip_size,
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
            break

    return result
