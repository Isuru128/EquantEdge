"""
ml_dataset.py

Dataset builder for Liquidity Sweep ML Meta-Labeling.
Scans historical candle data, detects all candidate setups, extracts features,
and applies Triple-Barrier labeling (1 if 1:2 TP hit before SL, 0 if SL hit first).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.strategy import generate_signals, MA_PERIOD, RR_RATIO, PIP_BUFFER
    from src.ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
    from src.connect import connect, get_candles
else:
    from .strategy import generate_signals, MA_PERIOD, RR_RATIO, PIP_BUFFER
    from .ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
    from .connect import connect, get_candles


def label_signals_triple_barrier(
    df: pd.DataFrame,
    symbol: str | None = None,
    max_hold_bars: int = 25,
    rr_ratio: float = RR_RATIO,
    pip_size: float | None = None,
) -> pd.DataFrame:

    """
    Simulate trade outcomes using Breakeven-Aware Triple-Barrier Method:
      - Upper Barrier: 1:1.5 Take Profit price (Outcome = 1, WIN)
      - Breakeven Trail: If trade achieves +1.0R floating profit, Stop Loss is moved to Entry.
      - Lower Barrier: Stop Loss price (Outcome = 0, LOSS only if SL hit before reaching +1.0R)
      - Vertical Barrier: `max_hold_bars` (Evaluates floating PnL at expiry)
    """
    if pip_size is None:
        if __package__ is None or __package__ == "":
            from src.strategy import _detect_pip_size
        else:
            from .strategy import _detect_pip_size
        pip_size = _detect_pip_size(df, symbol=symbol)

    # 1. Generate strategy signals and compute indicators
    df = compute_technical_indicators(df, ma_period=MA_PERIOD)
    df = generate_signals(df, ma_period=MA_PERIOD, rr_ratio=rr_ratio, pip_buffer=PIP_BUFFER, pip_size=pip_size)

    signal_indices = df.index[df["signal"] != 0].tolist()
    records = []

    n_bars = len(df)

    for idx in signal_indices:
        if idx >= n_bars - 2:
            continue  # Not enough future bars to evaluate outcome

        row = df.iloc[idx]
        sig = int(row["signal"])
        sl = float(row["sl_price"])
        tp = float(row["tp_price"])
        entry = float(row["entry_price"])
        risk_dist = abs(entry - sl)
        be_dist = 1.0 * risk_dist  # +1.0R threshold to move SL to Breakeven

        # Extract features at signal trigger
        feats = extract_features_for_signal(
            df=df,
            idx=idx,
            signal_side=sig,
            pip_size=pip_size,
            ma_period=MA_PERIOD,
        )

        # Forward simulate price path with Breakeven Trailing
        target = None
        bars_held = 0
        be_active = False

        for fwd_idx in range(idx + 1, min(idx + max_hold_bars + 1, n_bars)):
            fwd_bar = df.iloc[fwd_idx]
            bars_held += 1

            if sig == -1:  # SELL
                # Check if +1.0R Breakeven Trigger reached
                if not be_active and fwd_bar["low"] <= (entry - be_dist):
                    be_active = True

                # Check SL hit
                if not be_active:
                    if fwd_bar["high"] >= sl:
                        target = 0
                        break
                else:
                    # Breakeven active: SL is at entry
                    if fwd_bar["high"] >= entry:
                        target = 1  # Breakeven Scratch / Protected Win
                        break

                # Check 1:1.5 TP hit
                if fwd_bar["low"] <= tp:
                    target = 1
                    break

            elif sig == 1:  # BUY
                # Check if +1.0R Breakeven Trigger reached
                if not be_active and fwd_bar["high"] >= (entry + be_dist):
                    be_active = True

                # Check SL hit
                if not be_active:
                    if fwd_bar["low"] <= sl:
                        target = 0
                        break
                else:
                    # Breakeven active: SL is at entry
                    if fwd_bar["low"] <= entry:
                        target = 1  # Breakeven Scratch / Protected Win
                        break

                # Check 1:1.5 TP hit
                if fwd_bar["high"] >= tp:
                    target = 1
                    break

        # If time barrier expired without hitting either
        if target is None:
            final_close = df.iloc[min(idx + max_hold_bars, n_bars - 1)]["close"]
            pnl = (entry - final_close) if sig == -1 else (final_close - entry)
            target = 1 if (pnl >= 0 or be_active) else 0

        feats["symbol"]      = symbol or "UNKNOWN"
        feats["datetime"]    = str(row["datetime"])
        feats["entry_price"] = entry
        feats["sl_price"]    = sl
        feats["tp_price"]    = tp
        feats["bars_held"]   = bars_held
        feats["target"]      = int(target)

        records.append(feats)

    dataset_df = pd.DataFrame(records)
    return dataset_df


def build_dataset_from_mt5(
    symbols: list[str] | str = ("GBPUSD", "XAUUSD", "EURUSD"),
    timeframe: int = 1,  # M1 (1-minute)
    num_candles: int = 50000,
    output_path: str = "data/sweep_ml_dataset.csv",
) -> pd.DataFrame:
    """
    Fetch candles directly from active MT5 connection for one or multiple symbols (e.g. GBPUSD, XAUUSD, EURUSD)
    and build a pooled multi-asset ML dataset on M1 (or configured timeframe).
    """
    import MetaTrader5 as mt5
    tf_map = {
        1: mt5.TIMEFRAME_M1,
        3: mt5.TIMEFRAME_M3,
        5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)


    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    print(f"[Dataset] Connecting to MT5. Processing symbols: {symbols} on M{timeframe} ({num_candles} candles each)...")
    connect()

    all_frames = []
    for sym in symbols:
        try:
            sym_info = mt5.symbol_info(sym)
            actual_sym = sym
            if sym_info is None:
                if "US30" in sym:
                    for alt in ("US30Cash", "DJI", "WS30", "DJ30", "WallStreet30"):
                        if mt5.symbol_info(alt) is not None:
                            actual_sym = alt
                            break
            
            df_candles = get_candles(actual_sym, tf, num_candles)
            if df_candles.empty:
                print(f"[Dataset] Warning: No candles returned for {sym} ({actual_sym}). Skipping.")
                continue

            print(f"[Dataset] Fetched {len(df_candles)} candles for {sym} ({actual_sym}). Labeling setups...")
            sym_dataset = label_signals_triple_barrier(df_candles, symbol=sym)
            if not sym_dataset.empty:
                all_frames.append(sym_dataset)
                wins = (sym_dataset["target"] == 1).sum()
                print(f"  -> {sym}: {len(sym_dataset)} setups ({wins} Wins, {len(sym_dataset)-wins} Losses)")

        except Exception as exc:
            print(f"[Dataset] Error processing {sym}: {exc}")

    if not all_frames:
        raise RuntimeError("No data could be generated for any requested symbols.")

    dataset = pd.concat(all_frames, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)

    win_count = (dataset["target"] == 1).sum() if not dataset.empty else 0
    total_count = len(dataset)
    win_rate = (win_count / total_count * 100) if total_count > 0 else 0

    print(f"\n[Dataset] Total Pooled Dataset: {total_count} labeled setups across {len(symbols)} symbols.")
    print(f"[Dataset] Base Multi-Asset Win Rate: {win_rate:.1f}% ({win_count} Wins, {total_count - win_count} Losses).")
    print(f"[Dataset] Saved pooled dataset to {output_path}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML dataset for Liquidity Sweep strategy.")
    parser.add_argument("--symbols", type=str, default="GBPUSD,XAUUSD,EURUSD", help="Comma-separated symbols (e.g. GBPUSD,XAUUSD,EURUSD)")
    parser.add_argument("--tf", type=int, default=1, help="Timeframe in minutes (1, 3, 5, 15)")
    parser.add_argument("--candles", type=int, default=50000, help="Number of historical candles to fetch per symbol")
    parser.add_argument("--out", type=str, default="data/sweep_ml_dataset.csv", help="Output CSV path")
    args = parser.parse_args()


    build_dataset_from_mt5(
        symbols=args.symbols,
        timeframe=args.tf,
        num_candles=args.candles,
        output_path=args.out,
    )



