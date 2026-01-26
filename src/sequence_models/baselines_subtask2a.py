from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    from src.sequence_models.subtask2a_sequence_dataset import (
        load_subtask2a_with_embeddings,
    )

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


# =========================
# Linear(prev) baseline utils (Subtask 2A)
# =========================


@dataclass
class LinearPrevModel:
    coef_valence: np.ndarray
    coef_arousal: np.ndarray
    feature_names: Tuple[str, str, str]
    fit_rows: int
    n_users: int


def add_prev_delta_features(
    df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    ts_col: str = "timestamp",
    text_id_col: str = "text_id",
    dv_col: str = "state_change_valence",
    da_col: str = "state_change_arousal",
) -> pd.DataFrame:
    working = df.copy()
    working["_orig_idx"] = working.index
    working["_ts"] = pd.to_datetime(working[ts_col], errors="raise")
    working = working.sort_values([user_col, "_ts", text_id_col], kind="mergesort")

    eligible = working[dv_col].notna() & working[da_col].notna()
    eligible_df = working.loc[eligible, [user_col, dv_col, da_col]].copy()
    eligible_df["dv_prev"] = eligible_df.groupby(user_col, sort=False)[dv_col].shift(1)
    eligible_df["da_prev"] = eligible_df.groupby(user_col, sort=False)[da_col].shift(1)

    working["dv_prev"] = np.nan
    working["da_prev"] = np.nan
    working.loc[eligible_df.index, "dv_prev"] = eligible_df["dv_prev"]
    working.loc[eligible_df.index, "da_prev"] = eligible_df["da_prev"]
    working["dv_prev_missing"] = working["dv_prev"].isna()
    working["da_prev_missing"] = working["da_prev"].isna()

    out = working.set_index("_orig_idx").sort_index()
    out = out.drop(columns=["_ts"])
    out.index.name = df.index.name
    return out


def fit_linear_prev(
    train_df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    ts_col: str = "timestamp",
    text_id_col: str = "text_id",
    dv_col: str = "state_change_valence",
    da_col: str = "state_change_arousal",
) -> LinearPrevModel:
    feats = add_prev_delta_features(
        train_df,
        user_col=user_col,
        ts_col=ts_col,
        text_id_col=text_id_col,
        dv_col=dv_col,
        da_col=da_col,
    )
    fit_mask = feats["dv_prev"].notna() & feats["da_prev"].notna()
    if fit_mask.sum() == 0:
        raise ValueError("No eligible rows with prev deltas for fitting.")
    X = np.column_stack(
        [
            np.ones(fit_mask.sum(), dtype=float),
            feats.loc[fit_mask, "dv_prev"].to_numpy(dtype=float),
            feats.loc[fit_mask, "da_prev"].to_numpy(dtype=float),
        ]
    )
    yv = feats.loc[fit_mask, dv_col].to_numpy(dtype=float)
    ya = feats.loc[fit_mask, da_col].to_numpy(dtype=float)
    coef_v, *_ = np.linalg.lstsq(X, yv, rcond=None)
    coef_a, *_ = np.linalg.lstsq(X, ya, rcond=None)
    return LinearPrevModel(
        coef_valence=coef_v,
        coef_arousal=coef_a,
        feature_names=("intercept", "dv_prev", "da_prev"),
        fit_rows=int(fit_mask.sum()),
        n_users=int(feats[user_col].nunique()),
    )


def predict_linear_prev(
    df: pd.DataFrame,
    model: LinearPrevModel,
    *,
    user_col: str = "user_id",
    ts_col: str = "timestamp",
    text_id_col: str = "text_id",
    dv_col: str = "state_change_valence",
    da_col: str = "state_change_arousal",
    fill_value: float = 0.0,
    out_v_col: str = "baseline_delta_valence_pred",
    out_a_col: str = "baseline_delta_arousal_pred",
) -> pd.DataFrame:
    feats = add_prev_delta_features(
        df,
        user_col=user_col,
        ts_col=ts_col,
        text_id_col=text_id_col,
        dv_col=dv_col,
        da_col=da_col,
    )
    dv_prev = feats["dv_prev"].fillna(fill_value).to_numpy(dtype=float)
    da_prev = feats["da_prev"].fillna(fill_value).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(feats), dtype=float), dv_prev, da_prev])
    preds_v = X @ model.coef_valence
    preds_a = X @ model.coef_arousal
    out = feats.copy()
    out[out_v_col] = preds_v
    out[out_a_col] = preds_a
    return out


def select_latest_eligible_anchors(
    df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    ts_col: str = "timestamp",
    text_id_col: str = "text_id",
    dv_col: str = "state_change_valence",
    da_col: str = "state_change_arousal",
) -> pd.DataFrame:
    working = df.copy()
    working["_ts"] = pd.to_datetime(working[ts_col], errors="raise")
    eligible = working[dv_col].notna() & working[da_col].notna()
    eligible_df = working.loc[eligible].copy()
    eligible_df = eligible_df.sort_values(
        [user_col, "_ts", text_id_col], kind="mergesort"
    )
    anchors = eligible_df.groupby(user_col, sort=False).tail(1)
    anchors = df.loc[anchors.index].copy()
    return anchors


def make_phase0_user_preds(
    df: pd.DataFrame,
    model: LinearPrevModel,
    *,
    user_col: str = "user_id",
    ts_col: str = "timestamp",
    text_id_col: str = "text_id",
    dv_col: str = "state_change_valence",
    da_col: str = "state_change_arousal",
) -> pd.DataFrame:
    anchors = select_latest_eligible_anchors(
        df,
        user_col=user_col,
        ts_col=ts_col,
        text_id_col=text_id_col,
        dv_col=dv_col,
        da_col=da_col,
    )
    preds = predict_linear_prev(
        df,
        model,
        user_col=user_col,
        ts_col=ts_col,
        text_id_col=text_id_col,
        dv_col=dv_col,
        da_col=da_col,
        fill_value=0.0,
        out_v_col="delta_valence_pred",
        out_a_col="delta_arousal_pred",
    )
    pred_anchors = preds.loc[anchors.index].copy()
    anchor_ts = pd.to_datetime(pred_anchors[ts_col], errors="raise")
    out = pd.DataFrame(
        {
            "user_id": pred_anchors[user_col].astype("string"),
            "anchor_idx": pred_anchors.index.to_numpy(dtype="int64"),
            "anchor_text_id": pred_anchors[text_id_col].astype("string"),
            "anchor_timestamp": anchor_ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z").astype(
                "string"
            ),
            "delta_valence_true": pred_anchors[dv_col].to_numpy(dtype=float),
            "delta_arousal_true": pred_anchors[da_col].to_numpy(dtype=float),
            "delta_valence_pred": pred_anchors["delta_valence_pred"].to_numpy(dtype=float),
            "delta_arousal_pred": pred_anchors["delta_arousal_pred"].to_numpy(dtype=float),
        }
    )
    return out


def _self_test_linear_prev() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 5 + ["u2"] * 4,
            "timestamp": [
                "2025-01-01T00:00:00+0000",
                "2025-01-02T00:00:00+0000",
                "2025-01-03T00:00:00+0000",
                "2025-01-04T00:00:00+0000",
                "2025-01-05T00:00:00+0000",
                "2025-01-01T00:00:00+0000",
                "2025-01-02T00:00:00+0000",
                "2025-01-03T00:00:00+0000",
                "2025-01-04T00:00:00+0000",
            ],
            "text_id": [1, 2, 3, 4, 5, 10, 11, 12, 13],
            "state_change_valence": [0.1, 0.2, np.nan, 0.05, 0.1, 0.0, 0.1, 0.1, np.nan],
            "state_change_arousal": [0.0, 0.1, np.nan, -0.1, 0.2, 0.0, 0.0, 0.1, np.nan],
        }
    )
    train_df = df.iloc[:6].copy()
    val_df = df.iloc[6:].copy()
    model = fit_linear_prev(train_df)
    out = make_phase0_user_preds(val_df, model)
    required = {
        "user_id",
        "anchor_idx",
        "anchor_text_id",
        "anchor_timestamp",
        "delta_valence_true",
        "delta_arousal_true",
        "delta_valence_pred",
        "delta_arousal_pred",
    }
    missing = required - set(out.columns)
    if missing:
        raise AssertionError(f"Self-test missing columns: {missing}")
    if out["user_id"].duplicated().any():
        raise AssertionError("Self-test failed: duplicate user_id in anchors.")
    print("OK: linear(prev) self-test passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subtask 2A baselines.")
    parser.add_argument("--self_test", action="store_true", help="Run linear(prev) self-test.")
    args, _unknown = parser.parse_known_args()
    if args.self_test:
        _self_test_linear_prev()
    else:
        main()
