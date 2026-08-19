"""
test_ml_pipeline.py

Unit tests for the XGBoost ML feature engineering, dataset builder, and inference filter.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
from src.ml_dataset import label_signals_triple_barrier
from src.strategy import generate_signals, get_latest_signal


def test_feature_extraction():
    """Verify that feature extraction generates all required features without NaNs."""
    dates = pd.date_range("2026-08-19 10:00", periods=60, freq="1min")
    data = {
        "datetime": dates,
        "open":  [4300.0] * 58 + [4340.0, 4346.0],
        "high":  [4305.0] * 58 + [4346.0, 4348.0],
        "low":   [4295.0] * 58 + [4339.0, 4338.0],
        "close": [4300.0] * 58 + [4345.0, 4338.0],
        "volume": [100.0] * 60,
    }
    df = pd.DataFrame(data)
    df_ind = compute_technical_indicators(df, ma_period=10)

    feats = extract_features_for_signal(
        df=df_ind,
        idx=59,
        signal_side=-1,
        pip_size=0.01,
        ma_period=10,
    )

    for col in FEATURE_COLUMNS:
        assert col in feats, f"Missing feature: {col}"
        assert not pd.isna(feats[col]), f"NaN found in feature {col}"

    assert feats["signal_side"] == -1
    assert feats["sweep_depth_pips"] == 200.0  # (4348 - 4346) / 0.01 = 200.0 pips
    print("[PASS] Feature extraction test passed!")



def test_triple_barrier_labeling():
    """Verify that triple barrier forward simulation assigns 1 to wins and 0 to losses."""
    dates = pd.date_range("2026-08-19 10:00", periods=70, freq="1min")
    
    # Sell signal at bar 59: Entry=4338, SL=4348.01, TP=4317.98
    # Following bars drop to 4310 (hits TP before SL) -> Target = 1
    prices = [4300.0] * 58 + [4345.0, 4338.0] + [4330.0, 4325.0, 4315.0, 4310.0] + [4310.0] * 6
    highs =  [4305.0] * 58 + [4346.0, 4348.0] + [4335.0, 4328.0, 4318.0, 4312.0] + [4312.0] * 6
    lows =   [4295.0] * 58 + [4339.0, 4338.0] + [4328.0, 4320.0, 4312.0, 4305.0] + [4305.0] * 6
    opens =  [4300.0] * 58 + [4340.0, 4346.0] + [4338.0, 4330.0, 4325.0, 4315.0] + [4310.0] * 6

    df = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": [100.0] * 70,
    })

    dataset = label_signals_triple_barrier(df, max_hold_bars=10, pip_size=0.01)
    assert not dataset.empty, "Dataset should contain labeled records"
    assert dataset.iloc[0]["target"] == 1, f"Expected target 1 (TP hit), got {dataset.iloc[0]['target']}"
def test_model_train_and_inference():
    """Verify that an XGBoost model can be trained, saved, and used in get_latest_signal inference."""
    import xgboost as xgb
    import joblib
    from src.train_xgb import train_model
    from src.strategy import get_latest_signal

    # Generate synthetic training dataset
    np.random.seed(42)
    n_samples = 60
    synth_data = {
        col: np.random.randn(n_samples) for col in FEATURE_COLUMNS
    }
    synth_data["target"] = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    synth_df = pd.DataFrame(synth_data)

    test_csv = "data/test_synth_dataset.csv"
    test_model_path = "models/xgb_sweep_model.pkl"
    test_meta_path = "models/xgb_metadata.json"

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    synth_df.to_csv(test_csv, index=False)

    meta = train_model(
        dataset_path=test_csv,
        model_output_path=test_model_path,
        meta_output_path=test_meta_path,
        confidence_threshold=0.50,
    )

    assert os.path.exists(test_model_path), "Model artifact must exist"
    assert os.path.exists(test_meta_path), "Metadata JSON must exist"
    assert meta["total_samples"] == n_samples

    # Test inference inside get_latest_signal
    # Note: get_latest_signal evaluates iloc[-2] (the last closed bar), with iloc[-1] being forming.
    dates = pd.date_range("2026-08-19 10:00", periods=61, freq="1min")
    data = {
        "datetime": dates,
        "open":  [4300.0] * 58 + [4340.0, 4346.0, 4338.0],
        "high":  [4305.0] * 58 + [4346.0, 4348.0, 4339.0],
        "low":   [4295.0] * 58 + [4339.0, 4338.0, 4335.0],
        "close": [4300.0] * 58 + [4345.0, 4338.0, 4336.0],
        "volume": [100.0] * 61,
    }
    df = pd.DataFrame(data)

    sig = get_latest_signal(df, ma_period=10, use_ml=True, confidence_threshold=0.50)
    assert "xgb_prob" in sig, "Signal dict must contain xgb_prob"
    assert sig["xgb_prob"] is not None, "xgb_prob should be computed"
    print(f"[PASS] Model training & live inference test passed! (Evaluated Win Prob: {sig['xgb_prob']*100:.1f}%)")



if __name__ == "__main__":
    test_feature_extraction()
    test_triple_barrier_labeling()
    test_model_train_and_inference()
    print("\nALL ML PIPELINE UNIT TESTS PASSED!")

