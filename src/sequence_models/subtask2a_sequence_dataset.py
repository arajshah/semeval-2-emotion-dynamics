from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loader import load_all_data
from src.eval.splits import load_frozen_split


def load_subtask2a_with_embeddings(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Subtask 2A data and align it with precomputed embeddings (by user_id, text_id).

    Returns:
        merged_df: Subtask 2A DataFrame with an extra 'emb_index' column
        embeddings: np.ndarray of shape (N, D)
    """
    data_bundle = load_all_data()
    subtask2a = data_bundle["subtask2a"].copy()

    # Contract: df_raw is unmodified here; eligibility filtering is applied only downstream.
    required_cols = {
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "state_change_valence",
        "state_change_arousal",
    }
    missing = required_cols - set(subtask2a.columns)
    if missing:
        raise ValueError(f"Subtask2A missing required columns: {sorted(missing)}")

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    user_ids = data["user_id"]
    text_ids = data["text_id"]

    emb_index_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids)),
        }
    )

    merged = subtask2a.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping entries between Subtask 2A and embeddings.")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")
    if merged["timestamp"].isna().any():
        raise ValueError("Subtask2A timestamp parse produced NaT values.")

    return merged, embeddings


def load_subtask2a_with_cached_embeddings(
    embeddings_path: Path | str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Subtask 2A data and align it with cached embeddings by (user_id, text_id).
    """
    data_bundle = load_all_data()
    df_raw = data_bundle["subtask2a"].copy().reset_index(drop=True)
    required_cols = {
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "state_change_valence",
        "state_change_arousal",
    }
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Subtask2A missing required columns: {sorted(missing)}")
    df_raw["idx"] = df_raw.index

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    user_ids = data["user_id"]
    text_ids = data["text_id"]

    emb_index_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids)),
        }
    )

    merged = df_raw.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping entries between Subtask 2A and embeddings.")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")
    if merged["timestamp"].isna().any():
        raise ValueError("Subtask2A timestamp parse produced NaT values.")

    return merged, embeddings


def build_subtask2a_val_anchored_users_from_split(
    seed: int,
    regime: str = "unseen_user",
    seq_len: int = 5,
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    quick_limit_users: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build anchored per-user validation rows from the frozen split indices.
    """
    df_raw = load_all_data()["subtask2a"].copy().reset_index(drop=True)
    if regime != "unseen_user":
        raise ValueError("Only unseen_user regime is supported for anchored val builder.")

    split_path = Path("reports") / "splits" / f"subtask2a_unseen_user_seed{seed}.json"
    train_idx, val_idx = load_frozen_split(split_path, df_raw)

    val_df_raw = df_raw.iloc[val_idx].copy()
    val_df_raw["anchor_idx"] = np.asarray(val_idx, dtype=int)
    if len(val_df_raw) != len(val_idx):
        raise RuntimeError(
            f"val_df_raw length mismatch: {len(val_df_raw)} vs val_idx {len(val_idx)}"
        )
    if val_df_raw["anchor_idx"].min() < 0 or val_df_raw["anchor_idx"].max() >= len(df_raw):
        raise RuntimeError("anchor_idx out of bounds for df_raw.")

    val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"])
    eligible = val_df_raw["state_change_valence"].notna() & val_df_raw[
        "state_change_arousal"
    ].notna()
    eligible_df = val_df_raw.loc[eligible].copy()

    anchors: List[pd.Series] = []
    for _, group in eligible_df.groupby("user_id", sort=False):
        group_sorted = group.sort_values("timestamp", kind="stable")
        anchors.append(group_sorted.iloc[-1])

    if not anchors:
        anchors_df = pd.DataFrame(
            columns=[
                "user_id",
                "anchor_idx",
                "anchor_text_id",
                "anchor_timestamp",
                "delta_valence_true",
                "delta_arousal_true",
            ]
        )
        return anchors_df, np.zeros((0, seq_len, 0), dtype=np.float32), np.array([], dtype=np.int64), np.array([0], dtype=np.int64)

    anchors_df = pd.DataFrame(anchors).reset_index(drop=True)
    if quick_limit_users is not None:
        anchors_df = anchors_df.head(int(quick_limit_users)).copy()

    merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)
    merged_by_user = {
        user_id: group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for user_id, group in merged.groupby("user_id", sort=False)
    }
    embedding_dim = embeddings.shape[1]

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    rows: List[dict] = []

    for _, anchor in anchors_df.iterrows():
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

        sequences.append(seq)
        lengths.append(len(window_indices))
        rows.append(
            {
                "user_id": user_id,
                "anchor_idx": anchor_idx,
                "anchor_text_id": int(text_id),
                "anchor_timestamp": anchor_ts,
                "delta_valence_true": float(anchor["state_change_valence"]),
                "delta_arousal_true": float(anchor["state_change_arousal"]),
            }
        )

    anchors_df_out = pd.DataFrame(rows)
    sequences_arr = np.stack(sequences, axis=0) if sequences else np.zeros((0, seq_len, embedding_dim), dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int64)
    embedding_dim_arr = np.array([embedding_dim], dtype=np.int64)

    if anchors_df_out["user_id"].duplicated().any():
        raise RuntimeError("Expected exactly one row per user in anchored outputs.")

    return anchors_df_out, sequences_arr, lengths_arr, embedding_dim_arr


class Subtask2ASequenceDataset(Dataset):
    """
    Dataset of fixed-length embedding sequences with ΔV/ΔA labels for Subtask 2A.
    This dataset is step-level; per-user anchored evaluation is handled elsewhere and must not use this Dataset.

    Each item:
        - inputs: Tensor of shape (seq_len, embedding_dim)
        - target: Tensor of shape (2,) with [ΔV, ΔA]
        - length: actual number of valid timesteps in the sequence (<= seq_len)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        seq_len: int = 5,
    ) -> None:
        self.seq_len = seq_len
        self.embedding_dim = embeddings.shape[1]
        df_sorted = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

        for _, group in df_sorted.groupby("user_id"):
            group = group.reset_index(drop=True)
            emb_indices = group["emb_index"].to_numpy()
            dval = group["state_change_valence"].to_numpy()
            dar = group["state_change_arousal"].to_numpy()

            for idx in range(len(group)):
                if np.isnan(dval[idx]) or np.isnan(dar[idx]):
                    continue

                start = max(0, idx - (seq_len - 1))
                end = idx + 1
                window_indices = emb_indices[start:end]
                actual_len = len(window_indices)

                seq = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)
                seq[self.seq_len - actual_len :] = embeddings[window_indices]

                target = np.array(
                    [dval[idx], dar[idx]],
                    dtype=np.float32,
                )

                self.samples.append((seq, target, actual_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        seq, target, length = self.samples[idx]
        return {
            "inputs": torch.from_numpy(seq),
            "target": torch.from_numpy(target),
            "length": length,
        }


def build_subtask2a_datasets(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    seq_len: int = 5,
    random_state: int = 42,
    split_mode: str = "random",
    split_path: Path | str | None = None,
    seed: int = 42,
) -> Tuple[Subtask2ASequenceDataset, Subtask2ASequenceDataset, int]:
    """
    Build train/validation datasets for Subtask 2A, splitting by user_id.

    Returns:
        train_dataset, val_dataset, embedding_dim
    """
    if split_mode == "frozen_indices":
        if split_path is None:
            raise ValueError("split_path is required for split_mode='frozen_indices'.")
        df_raw = load_all_data()["subtask2a"].copy().reset_index(drop=True)
        if len(df_raw) == 0:
            raise ValueError("Subtask2A df_raw is empty.")
        train_idx, val_idx = load_frozen_split(Path(split_path), df_raw)
        if np.min(train_idx) < 0 or np.max(train_idx) >= len(df_raw):
            raise ValueError("train_idx out of bounds for df_raw.")
        if np.min(val_idx) < 0 or np.max(val_idx) >= len(df_raw):
            raise ValueError("val_idx out of bounds for df_raw.")

        embeddings_path = Path(embeddings_path)
        data = np.load(embeddings_path)
        embeddings = data["embeddings"]
        user_ids = data["user_id"]
        text_ids = data["text_id"]

        emb_index_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "text_id": text_ids,
                "emb_index": np.arange(len(user_ids)),
            }
        )

        train_df_raw = df_raw.iloc[train_idx].copy()
        val_df_raw = df_raw.iloc[val_idx].copy()

        train_df = train_df_raw.merge(
            emb_index_df,
            on=["user_id", "text_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(train_df) != len(train_df_raw):
            expected = set(zip(train_df_raw["user_id"], train_df_raw["text_id"]))
            kept = set(zip(train_df["user_id"], train_df["text_id"]))
            missing_keys = list(expected - kept)[:20]
            raise RuntimeError(
                f"Embeddings coverage mismatch for train split: "
                f"{len(train_df_raw)} raw rows -> {len(train_df)} after inner-merge on (user_id,text_id). "
                f"Missing keys (sample): {missing_keys}"
            )
        val_df = val_df_raw.merge(
            emb_index_df,
            on=["user_id", "text_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(val_df) != len(val_df_raw):
            expected = set(zip(val_df_raw["user_id"], val_df_raw["text_id"]))
            kept = set(zip(val_df["user_id"], val_df["text_id"]))
            missing_keys = list(expected - kept)[:20]
            raise RuntimeError(
                f"Embeddings coverage mismatch for val split: "
                f"{len(val_df_raw)} raw rows -> {len(val_df)} after inner-merge on (user_id,text_id). "
                f"Missing keys (sample): {missing_keys}"
            )

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="raise")
        if train_df["timestamp"].isna().any():
            raise ValueError("Subtask2A timestamp parse produced NaT values.")
        val_df["timestamp"] = pd.to_datetime(val_df["timestamp"], errors="raise")
        if val_df["timestamp"].isna().any():
            raise ValueError("Subtask2A timestamp parse produced NaT values.")
    else:
        merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)
        users = merged["user_id"].unique()
        rng = np.random.RandomState(random_state)
        rng.shuffle(users)

        split_idx = int(len(users) * 0.8)
        train_users = set(users[:split_idx])
        val_users = set(users[split_idx:])

        train_df = merged[merged["user_id"].isin(train_users)].copy()
        val_df = merged[merged["user_id"].isin(val_users)].copy()

    train_dataset = Subtask2ASequenceDataset(train_df, embeddings, seq_len=seq_len)
    val_dataset = Subtask2ASequenceDataset(val_df, embeddings, seq_len=seq_len)
    embedding_dim = embeddings.shape[1]
    return train_dataset, val_dataset, embedding_dim


def collate_sequence_batch(batch: List[dict]) -> dict:
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    return {"inputs": inputs, "target": targets, "lengths": lengths}


def create_dataloaders(
    train_dataset: Subtask2ASequenceDataset,
    val_dataset: Subtask2ASequenceDataset,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    return train_loader, val_loader

