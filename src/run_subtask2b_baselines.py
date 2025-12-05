from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.data_loader import load_all_data


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute per-fold MAE and MSE for valence and arousal dimensions.

    Here, 'valence' and 'arousal' refer to disposition_change_valence /
    disposition_change_arousal (2B target).
    """
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == 2

    diff = y_pred - y_true
    mae_val = np.mean(np.abs(diff[:, 0]))
    mae_ar = np.mean(np.abs(diff[:, 1]))
    mse_val = np.mean(diff[:, 0] ** 2)
    mse_ar = np.mean(diff[:, 1] ** 2)

    return {
        "MAE_valence": float(mae_val),
        "MAE_arousal": float(mae_ar),
        "MSE_valence": float(mse_val),
        "MSE_arousal": float(mse_ar),
    }


def _aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate a list of per-fold metric dicts into mean/std metrics.
    """
    if not fold_metrics:
        raise ValueError("No fold metrics to aggregate.")

    keys = fold_metrics[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std(ddof=1))
    return aggregated


def _prepare_subtask2b_user_dataframe() -> pd.DataFrame:
    """
    Build a user-level DataFrame for Subtask 2B disposition change.

    - Targets from data["subtask2b_user"]:
        disposition_change_valence, disposition_change_arousal
    - Features from data["subtask2b_detailed"], aggregated per user:
        num_texts_per_user, group,
        mean_valence_half1, mean_valence_half2,
        mean_arousal_half1, mean_arousal_half2
    """
    data = load_all_data()

    df_detailed = data["subtask2b_detailed"].copy()
    df_user = data["subtask2b_user"].copy()

    feature_cols = [
        "num_texts_per_user",
        "group",
        "mean_valence_half1",
        "mean_valence_half2",
        "mean_arousal_half1",
        "mean_arousal_half2",
    ]

    df_features = (
        df_detailed[["user_id"] + feature_cols]
        .drop_duplicates(subset=["user_id"])
        .reset_index(drop=True)
    )

    df = df_user.merge(
        df_features,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    df = df.dropna(
        subset=[
            "disposition_change_valence",
            "disposition_change_arousal",
        ]
        + feature_cols
    ).reset_index(drop=True)

    return df


def _run_cv_global_mean(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """
    Cross-validated metrics for a trivial global-mean predictor
    on disposition_change_valence/arousal.
    """
    y = df[
        ["disposition_change_valence", "disposition_change_arousal"]
    ].to_numpy(dtype=float)
    global_mean = y.mean(axis=0, keepdims=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for _, val_idx in kf.split(y):
        y_val = y[val_idx]
        y_pred = np.repeat(global_mean, repeats=len(val_idx), axis=0)
        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _run_cv_ridge_features(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """
    Cross-validated Ridge regression on simple numeric user-level features
    for disposition change.
    """
    feature_cols = [
        "num_texts_per_user",
        "group",
        "mean_valence_half1",
        "mean_valence_half2",
        "mean_arousal_half1",
        "mean_arousal_half2",
    ]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[
        ["disposition_change_valence", "disposition_change_arousal"]
    ].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _run_cv_random_forest_features(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Cross-validated RandomForestRegressor on simple numeric user-level features
    for disposition change.
    """
    feature_cols = [
        "num_texts_per_user",
        "group",
        "mean_valence_half1",
        "mean_valence_half2",
        "mean_arousal_half1",
        "mean_arousal_half2",
    ]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[
        ["disposition_change_valence", "disposition_change_arousal"]
    ].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def run_subtask2b_baselines(n_splits: int = 5) -> pd.DataFrame:
    """
    Run k-fold cross-validation for simple Subtask 2B baselines
    and return a unified metrics DataFrame.
    """
    df = _prepare_subtask2b_user_dataframe()

    results: List[Dict[str, float]] = []

    gm_metrics = _run_cv_global_mean(df, n_splits=n_splits)
    gm_metrics["model_name"] = "global_mean"
    results.append(gm_metrics)

    ridge_metrics = _run_cv_ridge_features(df, n_splits=n_splits)
    ridge_metrics["model_name"] = "ridge_features"
    results.append(ridge_metrics)

    rf_metrics = _run_cv_random_forest_features(df, n_splits=n_splits)
    rf_metrics["model_name"] = "random_forest_features"
    results.append(rf_metrics)

    df_results = pd.DataFrame(results)

    cols_order = [
        "model_name",
        "MAE_valence_mean",
        "MAE_valence_std",
        "MAE_arousal_mean",
        "MAE_arousal_std",
        "MSE_valence_mean",
        "MSE_valence_std",
        "MSE_arousal_mean",
        "MSE_arousal_std",
    ]
    cols_order = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[cols_order]

    return df_results


def main() -> None:
    df_results = run_subtask2b_baselines(n_splits=5)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "subtask2b_baseline_comparison.csv"

    df_results.to_csv(out_path, index=False)
    print(f"Saved Subtask 2B baseline comparison metrics to {out_path}")
    print(df_results)


if __name__ == "__main__":
    main()

