from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loader import load_all_data, print_data_summary


def print_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> None:
    """
    Compute and print MAE and MSE for a 2D target (valence, arousal).
    y_true and y_pred should be shape (n_samples, 2).
    """
    mae_valence = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_arousal = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mse_valence = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_arousal = mean_squared_error(y_true[:, 1], y_pred[:, 1])

    print(f"[{prefix}] MAE: valence={mae_valence:.4f}, arousal={mae_arousal:.4f}")
    print(f"[{prefix}] MSE: valence={mse_valence:.4f}, arousal={mse_arousal:.4f}")


def run_subtask1_baselines(df: pd.DataFrame, random_state: int = 42) -> None:
    """
    Run simple baselines for Subtask 1 (valence/arousal per entry):
    - global mean predictor
    - TF-IDF + Ridge regression
    """
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )

    y_train = train_df[["valence", "arousal"]].to_numpy()
    y_val = val_df[["valence", "arousal"]].to_numpy()

    # Global mean predictor
    mean_valence = y_train[:, 0].mean()
    mean_arousal = y_train[:, 1].mean()
    y_pred_mean = np.column_stack(
        [
            np.full_like(y_val[:, 0], fill_value=mean_valence, dtype=float),
            np.full_like(y_val[:, 1], fill_value=mean_arousal, dtype=float),
        ]
    )
    print_regression_metrics(y_val, y_pred_mean, prefix="Subtask 1 | Global mean")

    # TF-IDF + Ridge regression
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    X_val = vectorizer.transform(val_df["text"].astype(str))

    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred_reg = model.predict(X_val)
    print_regression_metrics(y_val, y_pred_reg, prefix="Subtask 1 | TF-IDF + Ridge")


def run_subtask2a_baseline(df: pd.DataFrame) -> None:
    """
    Run Δ=0 baseline for Subtask 2A (state change forecasting).
    Uses columns: state_change_valence, state_change_arousal.
    """
    mask = df["state_change_valence"].notna() & df["state_change_arousal"].notna()
    df_valid = df.loc[mask].copy()

    y_true = df_valid[["state_change_valence", "state_change_arousal"]].to_numpy()
    y_pred = np.zeros_like(y_true, dtype=float)

    mae_valence = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_arousal = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mse_valence = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_arousal = mean_squared_error(y_true[:, 1], y_pred[:, 1])

    print(f"[Subtask 2A | Δ=0 baseline] Δ-MAE: valence={mae_valence:.4f}, arousal={mae_arousal:.4f}")
    print(f"[Subtask 2A | Δ=0 baseline] Δ-MSE: valence={mse_valence:.4f}, arousal={mse_arousal:.4f}")
    print(f"[Subtask 2A] evaluated on {len(df_valid)} valid transitions")


def run_subtask2b_baseline(df_user: pd.DataFrame) -> None:
    """
    Run global mean baseline for Subtask 2B (dispositional change between halves).
    Uses user-level disposition_change_valence/arousal.
    """
    y_true = df_user[["disposition_change_valence", "disposition_change_arousal"]].to_numpy()

    mean_dc_valence = y_true[:, 0].mean()
    mean_dc_arousal = y_true[:, 1].mean()

    y_pred = np.column_stack(
        [
            np.full_like(y_true[:, 0], fill_value=mean_dc_valence, dtype=float),
            np.full_like(y_true[:, 1], fill_value=mean_dc_arousal, dtype=float),
        ]
    )

    print_regression_metrics(y_true, y_pred, prefix="Subtask 2B | Global mean")


def main() -> None:
    data = load_all_data()
    print_data_summary(data)

    print("\n=== Running Subtask 1 baselines ===")
    run_subtask1_baselines(data["subtask1"])

    print("\n=== Running Subtask 2A baseline ===")
    run_subtask2a_baseline(data["subtask2a"])

    print("\n=== Running Subtask 2B baseline ===")
    run_subtask2b_baseline(data["subtask2b_user"])


if __name__ == "__main__":
    main()

