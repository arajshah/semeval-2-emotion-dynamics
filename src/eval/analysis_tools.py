from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.sequence_models.subtask2a_sequence_dataset import load_subtask2a_with_embeddings
from src.sequence_models.simple_sequence_model import SimpleSequenceRegressor


def compute_delta_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_arousal: np.ndarray | None = None,
    y_pred_arousal: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute delta metrics.

    Legacy mode: y_true/y_pred are shape (N, 2) with columns [ΔV, ΔA].
    Phase 0 mode: pass y_true/y_pred for valence, and y_true_arousal/y_pred_arousal
    for arousal to compute Pearson r and MAE.
    """
    if y_true_arousal is None and y_pred_arousal is None:
        mae_val = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        mae_ar = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
        mse_val = np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
        mse_ar = np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)

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

    if y_true_arousal is None or y_pred_arousal is None:
        raise ValueError("Both arousal arrays must be provided for Phase 0 delta metrics.")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_arousal = np.asarray(y_true_arousal, dtype=float)
    y_pred_arousal = np.asarray(y_pred_arousal, dtype=float)

    return {
        "r_delta_valence": safe_pearsonr(y_true, y_pred, label="delta_valence"),
        "r_delta_arousal": safe_pearsonr(y_true_arousal, y_pred_arousal, label="delta_arousal"),
        "mae_delta_valence": float(np.mean(np.abs(y_true - y_pred))),
        "mae_delta_arousal": float(np.mean(np.abs(y_true_arousal - y_pred_arousal))),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, MSE, and Pearson correlation for regression targets.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))

    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"mae": mae, "mse": mse, "pearson": pearson}


def safe_pearsonr(x: np.ndarray, y: np.ndarray, label: str = "") -> float:
    """
    Compute Pearson r, returning 0.0 (with warning) for constant arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        warnings.warn(f"Constant or tiny arrays for Pearson r ({label}); returning 0.0")
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def iter_slices(df: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    """
    Return evaluation slices as positional indices (0..len(df)-1), aligned with y_true/y_pred.
    This avoids mixing pandas index labels with numpy positional indexing.
    """
    n = len(df)
    all_pos = np.arange(n, dtype=int)

    slices: List[Tuple[str, np.ndarray]] = [("all", all_pos)]

    if "is_words" in df.columns:
        mask_words = df["is_words"].to_numpy(dtype=bool)
        slices.append(("words", np.flatnonzero(mask_words).astype(int)))
        slices.append(("essays", np.flatnonzero(~mask_words).astype(int)))

    return slices


def compute_subtask1_correlations(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute within/between/composite correlations for valence and arousal.
    """
    # Ensure df.index is positional (0..len-1) and aligned with y_true/y_pred
    df = df.reset_index(drop=True)
    
    def within_user_corr(values_true: np.ndarray, values_pred: np.ndarray) -> float:
        per_user = []
        for _, group in df.groupby("user_id", sort=False):
            idx = group.index.to_numpy()
            if len(idx) < 2:
                continue
            per_user.append(safe_pearsonr(values_true[idx], values_pred[idx], label="within_user"))
        if not per_user:
            warnings.warn("No users with >=2 points for within-user correlation; returning 0.0")
            return 0.0
        return float(np.mean(per_user))

    def between_user_corr(values_true: np.ndarray, values_pred: np.ndarray) -> float:
        grouped = df.groupby("user_id", sort=False)
        true_means = grouped.apply(lambda g: float(np.mean(values_true[g.index])))
        pred_means = grouped.apply(lambda g: float(np.mean(values_pred[g.index])))
        return safe_pearsonr(true_means.to_numpy(), pred_means.to_numpy(), label="between_user")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    within_v = within_user_corr(y_true[:, 0], y_pred[:, 0])
    between_v = between_user_corr(y_true[:, 0], y_pred[:, 0])
    within_a = within_user_corr(y_true[:, 1], y_pred[:, 1])
    between_a = between_user_corr(y_true[:, 1], y_pred[:, 1])

    return {
        "r_within_valence": within_v,
        "r_between_valence": between_v,
        "r_composite_valence": float(np.mean([within_v, between_v])),
        "r_within_arousal": within_a,
        "r_between_arousal": between_a,
        "r_composite_arousal": float(np.mean([within_a, between_a])),
    }


def compute_subtask1_slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Compute Subtask 1 metrics across slices (all/words/essays).
    """
    results: List[Dict[str, float]] = []
    for slice_name, pos in iter_slices(df):
        if len(pos) == 0:
            continue
        slice_df = df.iloc[pos].reset_index(drop=True)
        slice_true = y_true[pos]
        slice_pred = y_pred[pos]
        corr = compute_subtask1_correlations(slice_df, slice_true, slice_pred)
        diagnostics_val = compute_regression_metrics(slice_true[:, 0], slice_pred[:, 0])
        diagnostics_aro = compute_regression_metrics(slice_true[:, 1], slice_pred[:, 1])
        results.append(
            {
                "slice": slice_name,
                **corr,
                "mae_valence": diagnostics_val["mae"],
                "mse_valence": diagnostics_val["mse"],
                "mae_arousal": diagnostics_aro["mae"],
                "mse_arousal": diagnostics_aro["mse"],
            }
        )
    return results

def compute_subtask1_metrics_from_preds(
    df: pd.DataFrame,
    preds_df: pd.DataFrame,
    val_idx: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Compute Subtask 1 metrics for frozen val indices using per-row predictions.
    """
    if "idx" not in preds_df.columns:
        raise ValueError("Predictions DataFrame must include an 'idx' column.")

    preds_df = preds_df.copy()
    preds_df = preds_df.sort_values("idx").reset_index(drop=True)

    val_idx = np.asarray(val_idx, dtype=int)
    expected_idx = np.sort(val_idx)
    pred_idx = preds_df["idx"].to_numpy(dtype=int)
    if not np.array_equal(pred_idx, expected_idx):
        raise ValueError("Prediction indices do not match the expected validation indices.")

    val_df = df.iloc[expected_idx].reset_index(drop=True)
    y_true = val_df[["valence", "arousal"]].to_numpy(dtype=float)
    y_pred = preds_df[["valence_pred", "arousal_pred"]].to_numpy(dtype=float)
    return compute_subtask1_slice_metrics(val_df, y_true, y_pred)


def make_unseen_user_splits(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create group-aware train/val splits for unseen-user evaluation.
    """
    groups = df["user_id"].to_numpy()
    indices = np.arange(len(df))
    n_users = len(np.unique(groups))

    if n_splits >= 2 and n_users >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(indices, groups=groups))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        splits = list(splitter.split(indices, groups=groups))

    return [(train_idx, val_idx) for train_idx, val_idx in splits]


def make_seen_user_time_split(
    df: pd.DataFrame, val_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a time-aware split within each user for seen-user evaluation.
    """
    time_cols = ["timestamp", "time", "date", "datetime"]
    step_cols = ["entry_index", "t", "step", "idx"]
    time_series = None

    for col in time_cols:
        if col in df.columns:
            time_series = pd.to_datetime(df[col], errors="coerce")
            break

    if time_series is None:
        for col in step_cols:
            if col in df.columns:
                time_series = pd.to_numeric(df[col], errors="coerce")
                break

    if time_series is not None:
        working = df.copy()
        working["_sort_time"] = time_series
    else:
        working = df

    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, group in working.groupby("user_id", sort=False):
        if "_sort_time" in working.columns:
            group = group.sort_values("_sort_time", kind="stable")

        idx = group.index.to_numpy()
        n = len(idx)
        if n <= 1:
            train_indices.extend(idx)
            continue

        n_train = int(np.floor(n * (1 - val_frac)))
        n_train = max(1, min(n - 1, n_train))

        train_indices.extend(idx[:n_train])
        val_indices.extend(idx[n_train:])

    return np.array(train_indices), np.array(val_indices)


def evaluate_subtask1(
    df: pd.DataFrame,
    pred_valence: np.ndarray,
    pred_arousal: np.ndarray,
    regime: str,
    slice_name: str,
    idx: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for a given Subtask 1 subset.
    """
    subset = df.iloc[idx]
    y_valence = subset["valence"].to_numpy(dtype=float)
    y_arousal = subset["arousal"].to_numpy(dtype=float)

    metrics_val = compute_regression_metrics(y_valence, pred_valence[idx])
    metrics_aro = compute_regression_metrics(y_arousal, pred_arousal[idx])

    return {
        "regime": regime,
        "slice": slice_name,
        "n": int(len(idx)),
        "valence_mae": metrics_val["mae"],
        "valence_mse": metrics_val["mse"],
        "valence_pearson": metrics_val["pearson"],
        "arousal_mae": metrics_aro["mae"],
        "arousal_mse": metrics_aro["mse"],
        "arousal_pearson": metrics_aro["pearson"],
    }


def build_subtask2a_eval_sequences(
    seq_len: int = 5,
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build evaluation sequences for Subtask 2A based on precomputed embeddings.

    Returns:
        sequences:   (N, seq_len, D) float32
        y_true:      (N, 2) float32 [ΔV, ΔA]
        user_ids:    (N,) user_id array
        timestamps:  (N,) timestamp array
        lengths:     (N,) int64 actual sequence lengths (<= seq_len)
        embedding_dim: int
    """
    merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)

    merged = merged.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    seq_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    user_list: List = []
    time_list: List = []
    len_list: List[int] = []
    embedding_dim = embeddings.shape[1]

    for user_id, group in merged.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        emb_indices = group["emb_index"].to_numpy()
        dval = group["state_change_valence"].to_numpy()
        dar = group["state_change_arousal"].to_numpy()
        tstamp = group["timestamp"].to_numpy()

        for idx in range(len(group)):
            if np.isnan(dval[idx]) or np.isnan(dar[idx]):
                continue

            start = max(0, idx - (seq_len - 1))
            end = idx + 1
            window_indices = emb_indices[start:end]
            actual_len = len(window_indices)

            seq = np.zeros((seq_len, embedding_dim), dtype=np.float32)
            seq[seq_len - actual_len :] = embeddings[window_indices]

            target = np.array([dval[idx], dar[idx]], dtype=np.float32)

            seq_list.append(seq)
            y_list.append(target)
            user_list.append(user_id)
            time_list.append(tstamp[idx])
            len_list.append(actual_len)

    if not seq_list:
        raise RuntimeError("No evaluation sequences constructed for Subtask 2A.")

    sequences = np.stack(seq_list, axis=0)
    y_true = np.stack(y_list, axis=0)
    user_ids = np.array(user_list)
    timestamps = np.array(time_list)
    lengths = np.array(len_list, dtype=np.int64)

    return sequences, y_true, user_ids, timestamps, lengths, embedding_dim


def compute_subtask2a_predictions(
    seq_len: int = 5,
    model_path: Path | str = Path("models/subtask2a_sequence/model.pt"),
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    hidden_dim: int = 128,
    num_layers: int = 1,
    batch_size: int = 64,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load the trained Subtask 2A sequence model, run it on evaluation sequences,
    and return a DataFrame of predictions plus global metrics.
    """
    sequences, y_true, user_ids, timestamps, lengths, embedding_dim = build_subtask2a_eval_sequences(
        seq_len=seq_len,
        embeddings_path=embeddings_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = SimpleSequenceRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_preds: List[np.ndarray] = []
    num_samples = sequences.shape[0]

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_seq = torch.from_numpy(sequences[start:end]).to(device)
            batch_lengths = torch.from_numpy(lengths[start:end]).to(device)

            outputs = model(batch_seq, batch_lengths)
            y_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    if y_pred.shape[0] != y_true.shape[0]:
        raise RuntimeError("Prediction and target sizes do not match.")

    metrics = compute_delta_metrics(y_true, y_pred)
    print("Global Δ metrics (Subtask 2A):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "timestamp": pd.to_datetime(timestamps),
            "delta_val_true": y_true[:, 0],
            "delta_aro_true": y_true[:, 1],
            "delta_val_pred": y_pred[:, 0],
            "delta_aro_pred": y_pred[:, 1],
        }
    )

    return df, metrics


def compute_subtask2a_val_user_preds_from_sequence_model(
    *,
    df_raw: pd.DataFrame,
    val_idx: np.ndarray,
    run_id: str,
    seed: int,
    model_path: Path | str,
    embeddings_path: Path | str,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
) -> pd.DataFrame:
    """
    Build per-user anchored predictions for Subtask 2A validation split.
    """
    val_df_raw = df_raw.iloc[val_idx].copy()
    val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"])
    if len(val_df_raw) != len(val_idx):
        raise RuntimeError(
            f"val_df_raw length mismatch: {len(val_df_raw)} vs val_idx {len(val_idx)}"
        )
    anchor_idx = np.asarray(val_idx, dtype=int)
    if anchor_idx.min() < 0 or anchor_idx.max() >= len(df_raw):
        raise RuntimeError("anchor_idx out of bounds for df_raw.")
    val_df_raw["anchor_idx"] = anchor_idx

    eligible_mask = val_df_raw["state_change_valence"].notna() & val_df_raw[
        "state_change_arousal"
    ].notna()
    eligible_df = val_df_raw.loc[eligible_mask].copy()

    if eligible_df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "seed",
                "user_id",
                "anchor_idx",
                "anchor_text_id",
                "anchor_timestamp",
                "delta_valence_true",
                "delta_arousal_true",
                "delta_valence_pred",
                "delta_arousal_pred",
            ]
        )

    anchors = (
        eligible_df.sort_values(["user_id", "timestamp"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(1)
        .sort_values("user_id", kind="stable")
    )

    merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)
    if not np.issubdtype(merged["timestamp"].dtype, np.datetime64):
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])

    embedding_dim = embeddings.shape[1]
    merged_by_user = {
        user_id: group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for user_id, group in merged.groupby("user_id", sort=False)
    }

    seqs: list[np.ndarray] = []
    lengths: list[int] = []
    rows: list[dict] = []

    for _, anchor in anchors.iterrows():
        user_id = anchor["user_id"]
        text_id = anchor["text_id"]
        anchor_ts = anchor["timestamp"]
        anchor_idx = int(anchor["anchor_idx"])

        group = merged_by_user.get(user_id)
        if group is None:
            raise RuntimeError(f"Missing user in embeddings merge: {user_id}")

        history = group[group["timestamp"] <= anchor_ts].reset_index(drop=True)
        if history.empty:
            raise RuntimeError(f"No history found for user {user_id} at anchor timestamp.")

        anchor_match = history[history["text_id"] == text_id]
        if anchor_match.empty:
            raise RuntimeError(
                f"Anchor (user_id={user_id}, text_id={text_id}) missing in embeddings merge."
            )

        emb_indices = history["emb_index"].to_numpy()
        actual_len = len(emb_indices)
        start = max(0, actual_len - seq_len)
        window_indices = emb_indices[start:actual_len]

        seq = np.zeros((seq_len, embedding_dim), dtype=np.float32)
        seq[seq_len - len(window_indices) :] = embeddings[window_indices]

        seqs.append(seq)
        lengths.append(len(window_indices))
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "user_id": user_id,
                "anchor_idx": anchor_idx,
                "anchor_text_id": int(text_id),
                "anchor_timestamp": anchor_ts,
                "delta_valence_true": float(anchor["state_change_valence"]),
                "delta_arousal_true": float(anchor["state_change_arousal"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    sequences = np.stack(seqs, axis=0)
    lengths_arr = np.array(lengths, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSequenceRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    model_path = Path(model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = start + batch_size
            batch_seq = torch.from_numpy(sequences[start:end]).to(device)
            batch_lengths = torch.from_numpy(lengths_arr[start:end]).to(device)
            outputs = model(batch_seq, batch_lengths)
            preds_list.append(outputs.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    df_out = pd.DataFrame(rows)
    df_out["delta_valence_pred"] = preds[:, 0]
    df_out["delta_arousal_pred"] = preds[:, 1]

    if df_out["user_id"].duplicated().any():
        raise RuntimeError("Expected one row per user in anchored preds.")
    if df_out["anchor_idx"].duplicated().any():
        raise RuntimeError("Anchor indices are not unique.")
    if df_out["anchor_idx"].min() < 0 or df_out["anchor_idx"].max() >= len(df_raw):
        raise RuntimeError("Anchor indices out of bounds.")
    if df_out[["delta_valence_pred", "delta_arousal_pred"]].isna().any().any():
        raise RuntimeError("NaNs found in prediction columns.")

    return df_out


def save_subtask2a_predictions(
    df: pd.DataFrame,
    path: Path | str = Path("reports/subtask2a_predictions.parquet"),
) -> Path:
    """
    Save Subtask 2A prediction-level DataFrame as a parquet file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved Subtask 2A predictions to: {path}")
    return path


def save_eval_summary(
    metrics: Dict[str, float],
    path: Path | str = Path("reports/eval_summary.csv"),
) -> Path:
    """
    Save global evaluation summary metrics as a one-row CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"Saved evaluation summary to: {path}")
    return path


def compute_per_user_delta_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δ metrics per user_id from the prediction DataFrame.
    """
    rows = []
    for user_id, group in pred_df.groupby("user_id"):
        y_true = group[["delta_val_true", "delta_aro_true"]].to_numpy()
        y_pred = group[["delta_val_pred", "delta_aro_pred"]].to_numpy()
        m = compute_delta_metrics(y_true, y_pred)
        m["user_id"] = user_id
        rows.append(m)
    return pd.DataFrame(rows)


def save_per_user_metrics(
    per_user_df: pd.DataFrame,
    path: Path | str = Path("reports/subtask2a_per_user_metrics.csv"),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_user_df.to_csv(path, index=False)
    print(f"Saved per-user metrics to: {path}")
    return path


def main() -> None:
    """
    Run evaluation for the Subtask 2A sequence model and save results.
    """
    df_pred, metrics = compute_subtask2a_predictions()

    save_subtask2a_predictions(df_pred)

    save_eval_summary(metrics)

    per_user_df = compute_per_user_delta_metrics(df_pred)
    save_per_user_metrics(per_user_df)


if __name__ == "__main__":
    main()

