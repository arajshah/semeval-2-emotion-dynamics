from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.sequence_models.subtask2a_sequence_dataset import load_subtask2a_with_embeddings
from src.sequence_models.simple_sequence_model import SimpleSequenceRegressor


def compute_delta_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Δ-MAE, Δ-MSE, and direction accuracy for valence and arousal.

    y_true, y_pred: shape (N, 2) with columns [ΔV, ΔA]
    """
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

