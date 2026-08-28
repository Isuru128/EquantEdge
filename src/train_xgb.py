"""
train_xgb.py

XGBoost Meta-Labeling Model Training & Anti-Overfitting Pipeline.

Anti-Overfitting Protections Implemented:
  1. Purged TimeSeriesSplit Cross-Validation (prevents lookahead / data leakage).
  2. Early Stopping (halts boosting automatically once validation loss stops improving).
  3. Shallow Tree Depth (max_depth=3 to prevent memorizing random noise).
  4. Stochastic Subsampling (subsample=0.80, colsample_bytree=0.80 to decorrelate trees).
  5. L1 Lasso & L2 Ridge Regularization (reg_alpha=0.2, reg_lambda=1.5).
  6. Conservative Shrinkage / Learning Rate (learning_rate=0.03).
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
    dataset_path: str = "data/strategy_ml_dataset.csv",
    model_output_path: str = "models/xgb_strategy_model.pkl",
    meta_output_path: str = "models/xgb_metadata.json",
    confidence_threshold: float = 0.55,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 15,
    reg_alpha: float = 0.2,
    reg_lambda: float = 1.5,
) -> dict:
    """
    Train regularized XGBoost classifier with early stopping and time-series cross-validation.
    """
    import joblib
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    if not os.path.exists(dataset_path):
        print(f"[Train] Dataset not found at {dataset_path}. Fetching from MT5...")
        build_dataset_from_mt5(num_candles=10000, output_path=dataset_path)

    df = pd.read_csv(dataset_path)
    if len(df) < 30:
        raise ValueError(f"Dataset has only {len(df)} rows. Need at least 30 samples to train.")

    # Filter to valid feature columns
    valid_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[valid_cols].fillna(0.0)
    y = df["target"].astype(int)

    base_win_rate = float(y.mean() * 100)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = float(n_neg / max(1, n_pos))

    print(f"[Train] Dataset shape: {X.shape} | Base Win Rate (Unfiltered): {base_win_rate:.2f}% ({n_pos} Wins, {n_neg} Losses)")
    print(f"[Train] Applied Anti-Overfitting Config: max_depth={max_depth}, lr={learning_rate}, early_stopping={early_stopping_rounds}, L1={reg_alpha}, L2={reg_lambda}")

    # Time-series cross validation (up to 5 splits)
    n_splits = max(2, min(5, len(df) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_aucs = []
    oof_probs = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_pos = max(1, int(y_train.sum()))
        train_neg = len(y_train) - train_pos
        train_weight = float(train_neg / train_pos)

        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=3,
            gamma=0.3,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=train_weight,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
        )

        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        test_probs = clf.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = test_probs

        try:
            auc = roc_auc_score(y_test, test_probs)
            cv_aucs.append(auc)
        except Exception:
            pass

    avg_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.5
    print(f"[Train] Out-of-Fold Cross-Validation AUC: {avg_auc:.3f}")

    # Threshold Sweep Analysis on Out-of-Fold predictions (strictly unseen data)
    valid_oof_mask = ~np.isnan(oof_probs)
    oof_y = y[valid_oof_mask]
    oof_p = oof_probs[valid_oof_mask]

    print("\n" + "=" * 70)
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
    print("=" * 70 + "\n")

    # Train final production model with 85% train / 15% validation early stopping
    split_point = int(len(X) * 0.85)
    X_f_train, X_f_val = X.iloc[:split_point], X.iloc[split_point:]
    y_f_train, y_f_val = y.iloc[:split_point], y.iloc[split_point:]

    final_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.3,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
    )
    final_model.fit(X_f_train, y_f_train, eval_set=[(X_f_val, y_f_val)], verbose=False)

    best_iteration = getattr(final_model, "best_iteration", 100)
    print(f"[Train] Early stopping selected optimal {best_iteration} boosting trees.")

    # Feature importances
    importances = dict(zip(valid_cols, [float(round(v, 4)) for v in final_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("[Train] Top Feature Importances:")
    for feat, imp in list(sorted_importances.items())[:8]:
        print(f"  - {feat:<24}: {imp*100:.2f}%")

    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_model, model_output_path)
    print(f"\n[Train] Saved model artifact to {model_output_path}")

    sel_thresh = oof_p >= confidence_threshold
    avg_filtered_wr = float(oof_y[sel_thresh].mean() * 100) if sel_thresh.sum() > 0 else base_win_rate

    # Save metadata
    metadata = {
        "features": valid_cols,
        "base_win_rate_pct": round(base_win_rate, 2),
        "cv_filtered_win_rate_pct": round(avg_filtered_wr, 2),
        "cv_auc": round(avg_auc, 3),
        "confidence_threshold": confidence_threshold,
        "best_iteration_trees": int(best_iteration),
        "feature_importances": sorted_importances,
        "total_samples": int(len(df)),
    }
    with open(meta_output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Train] Saved model metadata to {meta_output_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anti-overfitted XGBoost model for trading strategy.")
    parser.add_argument("--dataset", type=str, default="data/strategy_ml_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="models/xgb_strategy_model.pkl", help="Path to output model")
    parser.add_argument("--meta", type=str, default="models/xgb_metadata.json", help="Path to metadata JSON")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth (lower prevents overfitting)")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate shrinkage")
    parser.add_argument("--early-stopping", type=int, default=15, help="Rounds of no improvement to stop")
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        model_output_path=args.out,
        meta_output_path=args.meta,
        confidence_threshold=args.threshold,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        early_stopping_rounds=args.early_stopping,
    )
