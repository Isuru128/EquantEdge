"""
train_xgb.py

XGBoost Meta-Labeling Model Training & Validation for Liquidity Sweep Strategy.
Trains a gradient-boosted decision tree classifier to predict the probability
of a setup reaching its 1:2 Take Profit before hitting Stop Loss.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.ml_features import FEATURE_COLUMNS
    from src.ml_dataset import build_dataset_from_mt5
else:
    from .ml_features import FEATURE_COLUMNS
    from .ml_dataset import build_dataset_from_mt5


def train_model(
    dataset_path: str = "data/sweep_ml_dataset.csv",
    model_output_path: str = "models/xgb_sweep_model.pkl",
    meta_output_path: str = "models/xgb_metadata.json",
    confidence_threshold: float = 0.50,
) -> dict:

    """
    Train XGBoost model on labeled sweep dataset and export serialized artifacts.
    """
    import joblib
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    if not os.path.exists(dataset_path):
        print(f"[Train] Dataset not found at {dataset_path}. Fetching from MT5...")
        build_dataset_from_mt5(num_candles=10000, output_path=dataset_path)

    df = pd.read_csv(dataset_path)
    if len(df) < 30:
        raise ValueError(f"Dataset has only {len(df)} rows. Need at least 30 samples to train.")

    X = df[FEATURE_COLUMNS].fillna(0.0)
    y = df["target"].astype(int)

    base_win_rate = float(y.mean() * 100)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = float(n_neg / max(1, n_pos))

    print(f"[Train] Dataset shape: {X.shape} | Base Win Rate (Unfiltered): {base_win_rate:.2f}% ({n_pos} Wins, {n_neg} Losses)")
    print(f"[Train] Applied Class Balance Weight (scale_pos_weight): {pos_weight:.2f}")

    # Time-series cross validation (5 splits)
    tscv = TimeSeriesSplit(n_splits=min(5, len(df) // 15))
    cv_aucs = []
    
    # Store out-of-fold predictions
    oof_probs = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))

        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=train_weight,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X_train, y_train)

        test_probs = clf.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = test_probs

        try:
            auc = roc_auc_score(y_test, test_probs)
            cv_aucs.append(auc)
        except Exception:
            pass

    avg_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.5
    print(f"[Train] CV Out-Of-Fold Mean AUC: {avg_auc:.3f}")

    # Threshold Sweep Analysis on Out-of-Fold predictions
    valid_oof_mask = ~np.isnan(oof_probs)
    oof_y = y[valid_oof_mask]
    oof_p = oof_probs[valid_oof_mask]

    print("\n" + "="*70)
    print(f"{'THRESHOLD':<12} | {'TRADES TAKEN':<14} | {'WIN RATE':<12} | {'EXPECTANCY (2.0 RR)':<20}")
    print("-" * 70)
    for thresh in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        sel = oof_p >= thresh
        if sel.sum() > 0:
            wr = float(oof_y[sel].mean() * 100)
            ev = (wr / 100.0 * 2.0) - ((100.0 - wr) / 100.0 * 1.0)
            print(f"p >= {thresh:.2f}    | {sel.sum():<5} ({sel.mean()*100:4.1f}%)   | {wr:5.1f}%      | {ev:+.2f}R / trade")
        else:
            print(f"p >= {thresh:.2f}    | 0 trades      | N/A         | N/A")
    print("="*70 + "\n")



    # Train final production model on full dataset
    final_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    final_model.fit(X, y)

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, [float(round(v, 4)) for v in final_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("[Train] Top Feature Importances:")
    for feat, imp in list(sorted_importances.items())[:8]:
        print(f"  - {feat:<24}: {imp*100:.2f}%")



    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_model, model_output_path)
    print(f"\n[Train] Saved model artifact to {model_output_path}")

    # Filtered win rate at specified threshold
    sel_thresh = oof_p >= confidence_threshold
    avg_filtered_wr = float(oof_y[sel_thresh].mean() * 100) if sel_thresh.sum() > 0 else base_win_rate

    # Save metadata
    metadata = {
        "features": FEATURE_COLUMNS,
        "base_win_rate_pct": round(base_win_rate, 2),
        "cv_filtered_win_rate_pct": round(avg_filtered_wr, 2),
        "cv_auc": round(avg_auc, 3),
        "confidence_threshold": confidence_threshold,
        "feature_importances": sorted_importances,
        "total_samples": int(len(df)),
    }
    with open(meta_output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Train] Saved model metadata to {meta_output_path}")


    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for Liquidity Sweep strategy.")
    parser.add_argument("--dataset", type=str, default="data/sweep_ml_dataset.csv", help="Path to labeled dataset CSV")
    parser.add_argument("--out", type=str, default="models/xgb_sweep_model.pkl", help="Path to output model file")
    parser.add_argument("--meta", type=str, default="models/xgb_metadata.json", help="Path to output metadata JSON")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for trade execution")
    args = parser.parse_args()


    train_model(
        dataset_path=args.dataset,
        model_output_path=args.out,
        meta_output_path=args.meta,
        confidence_threshold=args.threshold,
    )
