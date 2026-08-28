"""
ml_dataset.py

Dataset builder for 15M FVG + 1M MSS Strategy ML Meta-Labeling.
Scans historical candle data across EURUSD, GBPUSD, AUDUSD, extracts multi-timeframe features,
and applies Triple-Barrier labeling (1:2.0 RR).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.strategy import generate_signals, _detect_pip_size, resample_1m_to_15m, RR_RATIO, SUPPORTED_SYMBOLS
    from src.ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
    from src.connect import connect, get_candles
else:
    from .strategy import generate_signals, _detect_pip_size, resample_1m_to_15m, RR_RATIO, SUPPORTED_SYMBOLS
    from .ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
    from .connect import connect, get_candles


def label_signals_triple_barrier(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    symbol: str | None = None,
    max_hold_bars: int = 45,
    rr_ratio: float = RR_RATIO,
    pip_size: float | None = None,
) -> pd.DataFrame:
    """
    Simulate trade outcomes using Breakeven-Aware Triple-Barrier Method:
      - Upper Barrier: 1:2 Take Profit price (Outcome = 1, WIN)
      - Breakeven Trail: If trade achieves +1.0R floating profit, Stop Loss is moved to Entry.
      - Lower Barrier: Stop Loss price (Outcome = 0, LOSS)
      - Vertical Barrier: `max_hold_bars` (Evaluates floating PnL at expiry)
    """
    if pip_size is None:
        pip_size = _detect_pip_size(df_1m, symbol=symbol)

    if df_15m is None or df_15m.empty:
        df_15m = resample_1m_to_15m(df_1m)

    df_signals = generate_signals(df_1m, df_15m=df_15m, symbol=symbol, rr_ratio=rr_ratio, pip_size=pip_size)
    signal_indices = df_signals.index[df_signals["signal"] != 0].tolist()
    records = []
    n_bars = len(df_1m)

    for idx in signal_indices:
        if idx >= n_bars - 2:
            continue

        row = df_signals.iloc[idx]
        sig = int(row["signal"])
        sl = float(row["sl_price"]) if not pd.isna(row["sl_price"]) else None
        tp = float(row["tp_price"]) if not pd.isna(row["tp_price"]) else None
        entry = float(row["entry_price"])

        if sl is None or tp is None:
            continue

        risk_dist = abs(entry - sl)
        be_dist = 1.0 * risk_dist
        fvg_top_val = float(row["fvg_top"]) if not pd.isna(row.get("fvg_top")) else None
        fvg_bot_val = float(row["fvg_bottom"]) if not pd.isna(row.get("fvg_bottom")) else None
        fvg_info = {"top": fvg_top_val, "bottom": fvg_bot_val} if fvg_top_val and fvg_bot_val else None

        # Extract features
        feats = extract_features_for_signal(
            df=df_1m,
            idx=idx,
            signal_side=sig,
            pip_size=pip_size,
            fvg=fvg_info,
            peak_high=sl if sig == -1 else None,
            valley_low=sl if sig == 1 else None,
        )

        # Forward simulate price path
        target = None
        bars_held = 0
        be_active = False

        for fwd_idx in range(idx + 1, min(idx + max_hold_bars + 1, n_bars)):
            fwd_bar = df_1m.iloc[fwd_idx]
            bars_held += 1

            if sig == -1:  # SELL
                if not be_active and fwd_bar["low"] <= (entry - be_dist):
                    be_active = True

                if not be_active:
                    if fwd_bar["high"] >= sl:
                        target = 0
                        break
                else:
                    if fwd_bar["high"] >= entry:
                        target = 1  # Breakeven protected
                        break

                if fwd_bar["low"] <= tp:
                    target = 1
                    break

            elif sig == 1:  # BUY
                if not be_active and fwd_bar["high"] >= (entry + be_dist):
                    be_active = True

                if not be_active:
                    if fwd_bar["low"] <= sl:
                        target = 0
                        break
                else:
                    if fwd_bar["low"] <= entry:
                        target = 1  # Breakeven protected
                        break

                if fwd_bar["high"] >= tp:
                    target = 1
                    break

        if target is None:
            final_close = df_1m.iloc[min(idx + max_hold_bars, n_bars - 1)]["close"]
            pnl = (entry - final_close) if sig == -1 else (final_close - entry)
            target = 1 if (pnl >= 0 or be_active) else 0

        feats["symbol"] = symbol or "UNKNOWN"
        feats["datetime"] = str(row["datetime"]) if "datetime" in row else ""
        feats["entry_price"] = entry
        feats["sl_price"] = sl
        feats["tp_price"] = tp
        feats["bars_held"] = bars_held
        feats["target"] = int(target)

        records.append(feats)

    dataset_df = pd.DataFrame(records)
    return dataset_df


def build_dataset_from_mt5(
    symbols: list[str] | str = ("EURUSD", "GBPUSD", "AUDUSD"),
    num_candles: int = 50000,
    rr_ratio: float = 2.0,
    output_path: str = "data/strategy_ml_dataset.csv",
) -> pd.DataFrame:
    """
    Fetch 1M & 15M candles from active MT5 connection for EURUSD, GBPUSD, AUDUSD
    and build a pooled dataset.
    """
    import MetaTrader5 as mt5

    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    print(f"[Dataset] Connecting to MT5 for symbols: {symbols} ({num_candles} 1M candles each, 1:{rr_ratio:.1f} RR)...")
    connect()

    all_frames = []
    for sym in symbols:
        try:
            df_1m = get_candles(sym, mt5.TIMEFRAME_M1, num_candles)
            df_15m = get_candles(sym, mt5.TIMEFRAME_M15, num_candles // 10)

            if df_1m.empty:
                print(f"[Dataset] Warning: No candles returned for {sym}. Skipping.")
                continue

            print(f"[Dataset] Fetched {len(df_1m)} 1M candles for {sym}. Labeling setups...")
            sym_dataset = label_signals_triple_barrier(df_1m, df_15m=df_15m, symbol=sym, rr_ratio=rr_ratio)
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
    print(f"[Dataset] Base Win Rate: {win_rate:.1f}% ({win_count} Wins, {total_count - win_count} Losses).")
    print(f"[Dataset] Saved pooled dataset to {output_path}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML dataset for 15M FVG + 1M MSS strategy.")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,AUDUSD", help="Comma-separated symbols")
    parser.add_argument("--candles", type=int, default=50000, help="Historical 1M candles count")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-to-reward ratio")
    parser.add_argument("--out", type=str, default="data/strategy_ml_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    build_dataset_from_mt5(
        symbols=args.symbols,
        num_candles=args.candles,
        rr_ratio=args.rr,
        output_path=args.out,
    )
