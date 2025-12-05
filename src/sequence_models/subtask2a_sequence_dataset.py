from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loader import load_all_data


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

    subtask2a = subtask2a.dropna(
        subset=["state_change_valence", "state_change_arousal"]
    ).copy()

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

    if not np.issubdtype(merged["timestamp"].dtype, np.datetime64):
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])

    return merged, embeddings


class Subtask2ASequenceDataset(Dataset):
    """
    Dataset of fixed-length embedding sequences with ΔV/ΔA labels for Subtask 2A.

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
        df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

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
) -> Tuple[Subtask2ASequenceDataset, Subtask2ASequenceDataset, int]:
    """
    Build train/validation datasets for Subtask 2A, splitting by user_id.

    Returns:
        train_dataset, val_dataset, embedding_dim
    """
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

