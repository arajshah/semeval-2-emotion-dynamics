from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.sequence_models.subtask2a_sequence_dataset import load_subtask2a_with_embeddings


def compute_delta_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Δ-MAE, Δ-MSE, and direction accuracy for valence and arousal.

    y_true, y_pred: shape (N, 2) with columns [ΔV, ΔA]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae_val = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
    mae_ar = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))
    mse_val = float(np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2))
    mse_ar = float(np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2))

    def direction_acc(y_t: np.ndarray, y_p: np.ndarray) -> float:
        sign_true = np.sign(y_t)
        sign_pred = np.sign(y_p)
        return float((sign_true == sign_pred).mean())

    dir_val = direction_acc(y_true[:, 0], y_pred[:, 0])
    dir_ar = direction_acc(y_true[:, 1], y_pred[:, 1])

    return {
        "Delta_MAE_valence": mae_val,
        "Delta_MAE_arousal": mae_ar,
        "Delta_MSE_valence": mse_val,
        "Delta_MSE_arousal": mse_ar,
        "DirAcc_valence": dir_val,
        "DirAcc_arousal": dir_ar,
    }


def split_users_unseen(
    df: pd.DataFrame, random_state: int = 42, train_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the split logic in build_subtask2a_datasets:
      - shuffle unique users with RandomState(seed)
      - train = first 80%, val = last 20%
    """
    users = df["user_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(users)

    split_idx = int(len(users) * train_frac)
    train_users = set(users[:split_idx])
    val_users = set(users[split_idx:])

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()
    return train_df, val_df


def baseline_zero_change(n: int) -> np.ndarray:
    """Predict ΔV=0, ΔA=0 for all samples."""
    return np.zeros((n, 2), dtype=np.float32)


def baseline_mean_change(train_df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Predict constant Δ equal to the mean Δ computed on TRAIN ONLY.
    """
    mean_dv = float(train_df["state_change_valence"].mean())
    mean_da = float(train_df["state_change_arousal"].mean())
    return np.tile(np.array([mean_dv, mean_da], dtype=np.float32), (n, 1))


def baseline_momentum(val_df: pd.DataFrame) -> np.ndarray:
    """
    "Last-change" / momentum baseline:
      For each user sequence, predict Δ at time i as the true Δ at time i-1.
      For the first timepoint in a user's sequence, predict 0.

    This uses only information that would be available from the user's past
    (previous observed V/A scores imply previous Δ), and is a strong sanity baseline.
    """
    working = val_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    preds = np.zeros((len(working), 2), dtype=np.float32)

    # Group by user and shift the true deltas by 1
    for user_id, grp in working.groupby("user_id", sort=False):
        idx = grp.index.to_numpy()
        dv = grp["state_change_valence"].to_numpy(dtype=np.float32)
        da = grp["state_change_arousal"].to_numpy(dtype=np.float32)

        # pred at first point = 0, else previous true delta
        preds[idx, 0] = np.concatenate([np.array([0.0], dtype=np.float32), dv[:-1]])
        preds[idx, 1] = np.concatenate([np.array([0.0], dtype=np.float32), da[:-1]])

    return preds


def main() -> None:
    embeddings_path = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz")
    random_state = 42

    # Load Subtask 2A df aligned with embeddings (df contains user_id, text_id, timestamp, state_change_*).
    merged_df, _embeddings = load_subtask2a_with_embeddings(embeddings_path)

    # Ensure timestamp is datetime for sorting (load_subtask2a_with_embeddings already tries to enforce this)
    if not np.issubdtype(merged_df["timestamp"].dtype, np.datetime64):
        merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], errors="coerce")

    # Match train/val user split used by your sequence training
    train_df, val_df = split_users_unseen(merged_df, random_state=random_state, train_frac=0.8)

    # Sort val for stable evaluation (important for momentum baseline construction)
    val_df = val_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    y_true = val_df[["state_change_valence", "state_change_arousal"]].to_numpy(dtype=np.float32)
    n = len(val_df)

    rows = []

    # 1) Zero-change baseline
    y_pred_zero = baseline_zero_change(n)
    m_zero = compute_delta_metrics(y_true, y_pred_zero)
    rows.append({"model": "zero_change", **m_zero})

    # 2) Mean-change baseline (train mean)
    y_pred_mean = baseline_mean_change(train_df, n)
    m_mean = compute_delta_metrics(y_true, y_pred_mean)
    rows.append({"model": "mean_change_train", **m_mean})

    # 3) Momentum baseline (last-change per user)
    y_pred_mom = baseline_momentum(val_df)
    m_mom = compute_delta_metrics(y_true, y_pred_mom)
    rows.append({"model": "momentum_last_change", **m_mom})

    # Print results
    print(f"Train users: {train_df['user_id'].nunique()}, Val users: {val_df['user_id'].nunique()}")
    print(f"Val samples: {n}")
    for r in rows:
        print("\nBaseline:", r["model"])
        print(f"  Delta_MAE_valence: {r['Delta_MAE_valence']:.4f}")
        print(f"  Delta_MAE_arousal: {r['Delta_MAE_arousal']:.4f}")
        print(f"  Delta_MSE_valence: {r['Delta_MSE_valence']:.4f}")
        print(f"  Delta_MSE_arousal: {r['Delta_MSE_arousal']:.4f}")
        print(f"  DirAcc_valence:    {r['DirAcc_valence']:.4f}")
        print(f"  DirAcc_arousal:    {r['DirAcc_arousal']:.4f}")

    # Save comparison CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "subtask2a_baseline_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved Subtask 2A baseline comparison to: {out_path}")


if __name__ == "__main__":
    main()
