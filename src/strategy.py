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


from zoneinfo import ZoneInfo

# ── Strategy Parameters ────────────────────────────────────────────────────────
STRATEGY_NAME           = "15M FVG + 1M Market Structure Shift (MSS)"
SUPPORTED_SYMBOLS       = ["EURUSD", "GBPUSD", "AUDUSD"]
MAX_FVG_LOOKBACK_BARS   = 96        # Max 96 15M candles (24 hours lookback for FVG)
RR_RATIO                = 2.0       # Institutional 1:2.0 Risk-to-Reward ratio
RISK_PERCENT            = 1.0       # 1.0% risk per trade
MIN_SL_PIPS             = 4.0       # Minimum measured Stop Loss filter (trades with SL < 4.0 pips are rejected)
DEFAULT_PIP             = 0.0001    # 0.0001 for EURUSD, GBPUSD, AUDUSD

# EURUSD-Specific Stricter Rules
EURUSD_MIN_CONFIDENCE   = 0.58      # Stricter ML threshold for EURUSD (>= 58%)
EURUSD_MIN_SL_PIPS      = 5.0       # Minimum 5.0 pips SL for EURUSD

ML_MODEL_PATH           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgb_strategy_model.pkl")
ML_CONFIDENCE_THRESHOLD = 0.55
USE_ML_FILTER           = True
# ─────────────────────────────────────────────────────────────────────────────

_NY_TZ = ZoneInfo("America/New_York")


def is_prime_killzone(dt) -> bool:
    """
    Validate that trade occurs during London Open (02:00 - 05:00 NY)
    or New York AM (07:00 - 11:30 NY) Killzones to eliminate choppy drift.
    Applies to all high-liquidity forex majors (EURUSD, GBPUSD, AUDUSD).
    """
    if dt is None:
        return True
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt_ny = dt.tz_localize("UTC").astimezone(_NY_TZ)
        elif hasattr(dt, "astimezone"):
            dt_ny = dt.astimezone(_NY_TZ)
        else:
            return True
        mins = dt_ny.hour * 60 + dt_ny.minute
        # London Open: 02:00 (120m) to 05:00 (300m)
        # NY AM: 07:00 (420m) to 11:30 (690m)
        if (120 <= mins <= 300) or (420 <= mins <= 690):
            return True
        return False
    except Exception:
        return True


is_eurusd_prime_killzone = is_prime_killzone


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
    min_sl_pips: float = MIN_SL_PIPS,
) -> pd.DataFrame:
    """
    Scan historical DataFrame and generate pure 15M FVG + 1M MSS signals across all candles.
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

    datetimes = df_1m["datetime"].astype(str).values if "datetime" in df_1m.columns else None

    lookback = 15
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

        # Check active FVGs matching the current bar timestamp
        active_fvgs = [
            f for f in all_fvgs
            if not (bar_time and f.get("datetime") and f["datetime"] >= bar_time)
        ]
        if not active_fvgs:
            continue

        sub_df = df_1m.iloc[: i + 1]

        for fvg in active_fvgs[:6]:
            side = fvg["direction"]

            mss = detect_market_structure_shift(
                df_1m=sub_df,
                fvg=fvg,
                trend_side=side,
                lookback_bars=60,
                pip_size=pip_size,
                min_sl_pips=min_sl_pips,
                require_retest=True,
            )

            if mss is not None:
                sig = mss["signal"]
                signals[i] = sig
                entry_prices[i] = mss["entry_price"]
                sl_prices[i] = mss["sl_price"]
                tp_prices[i] = mss["tp_price"]
                risk_distances[i] = mss["risk_distance"]
                sl_pips_arr[i] = mss["sl_pips"]
                tp_pips_arr[i] = mss["tp_pips"]
                fvg_tops[i] = fvg["top"]
                fvg_bots[i] = fvg["bottom"]
                dir_name = "BUY" if sig == 1 else "SELL"
                labels[i] = f"{dir_name} MSS RETEST (15M FVG + 1M Pivot | 1:{rr_ratio:.1f} RR)"
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
    min_sl_pips: float = MIN_SL_PIPS,
    use_ml: bool = False,
    confidence_threshold: float = 0.50,
) -> dict:
    """
    Evaluate the most recently closed 1-minute candle for a confirmed basic 15M FVG + 1M MSS setup.
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

    # Check active FVGs
    for fvg in fvgs[:10]:
        side = fvg["direction"]
        mss = detect_market_structure_shift(
            df_1m=df,
            fvg=fvg,
            trend_side=side,
            lookback_bars=60,
            pip_size=pip_size,
            min_sl_pips=min_sl_pips,
            require_retest=True,
        )

        if mss is not None:
            sig = mss["signal"]
            sl_pips_val = round(mss["sl_pips"], 1)

            if sl_pips_val < min_sl_pips:
                continue

            result["signal"] = sig
            result["entry_price"] = round(mss["entry_price"], 5)
            result["sl_price"] = round(mss["sl_price"], 5)
            result["tp_price"] = round(mss["tp_price"], 5)
            result["risk_distance"] = round(mss["risk_distance"], 5)
            result["sl_pips"] = sl_pips_val
            result["tp_pips"] = round(mss["tp_pips"], 1)
            result["fvg_top"] = fvg["top"]
            result["fvg_bottom"] = fvg["bottom"]
            result["fvg_type"] = fvg["type"]

            dir_name = "BUY" if sig == 1 else "SELL"
            result["label"] = f"{dir_name} MSS RETEST (15M FVG + 1M Pivot | 1:{rr_ratio:.1f} RR)"
            break

    return result
