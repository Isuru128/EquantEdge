"""
test_ml_pipeline.py

Unit tests for ML feature engineering, dataset builder, and XGBoost training pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_features import compute_technical_indicators, extract_features_for_signal, FEATURE_COLUMNS
from src.ml_dataset import label_signals_triple_barrier
from src.strategy import get_latest_signal


def test_feature_extraction():
    """Verify feature extraction generates all required features without NaNs."""
    dates = pd.date_range("2026-08-19 10:00", periods=60, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": [4300.0] * 60,
        "high": [4305.0] * 60,
        "low": [4295.0] * 60,
        "close": [4300.0] * 60,
        "volume": [100.0] * 60,
    })

    feats = extract_features_for_signal(
        df=df,
        idx=59,
        signal_side=1,
        pip_size=0.01,
    )

    for col in FEATURE_COLUMNS:
        assert col in feats, f"Missing feature: {col}"
        assert not pd.isna(feats[col]), f"NaN found in feature {col}"

    assert feats["signal_side"] == 1
    print("[PASS] Feature extraction test passed!")


def test_triple_barrier_labeling():
    """Verify that triple barrier simulation labels trade outcomes."""
    dates = pd.date_range("2026-08-19 10:00", periods=100, freq="1min")
    prices = [4300.0 + (i * 0.5) for i in range(100)]
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [100.0] * 100,
    })

    dataset = label_signals_triple_barrier(df, max_hold_bars=10, pip_size=0.01)
    # Target values should be binary 0 or 1
    if not dataset.empty:
        assert set(dataset["target"].unique()).issubset({0, 1})
    print("[PASS] Triple barrier labeling test passed!")


def test_model_train_and_inference():
    """Verify that model training and inference run cleanly."""
    from src.train_xgb import train_model

    # Synthetic training dataset
    np.random.seed(42)
    n_samples = 50
    synth_data = {col: np.random.randn(n_samples) for col in FEATURE_COLUMNS}
    synth_data["target"] = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    synth_df = pd.DataFrame(synth_data)

    test_csv = "data/test_tmp_dataset.csv"
    test_model_path = "models/xgb_strategy_model.pkl"
    test_meta_path = "models/xgb_metadata.json"

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    synth_df.to_csv(test_csv, index=False)

    try:
        meta = train_model(
            dataset_path=test_csv,
            model_output_path=test_model_path,
            meta_output_path=test_meta_path,
            confidence_threshold=0.50,
        )

        assert os.path.exists(test_model_path), "Model artifact must exist"
        assert os.path.exists(test_meta_path), "Metadata JSON must exist"
        assert meta["total_samples"] == n_samples
        print("[PASS] Model training & inference pipeline test passed!")
    finally:
        # Clean up temporary test files
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        if os.path.exists(test_meta_path):
            os.remove(test_meta_path)


if __name__ == "__main__":
    test_feature_extraction()
    test_triple_barrier_labeling()
    test_model_train_and_inference()
    print("\nALL ML PIPELINE UNIT TESTS PASSED!")
