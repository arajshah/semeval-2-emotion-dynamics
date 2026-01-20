from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_repo_root() -> Path:
    """
    Return the repository root by walking upward from this file.
    """
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    return start


def get_splits_dir() -> Path:
    """
    Return the reports/splits directory at the repo root.
    """
    return get_repo_root() / "reports" / "splits"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_or_create_unseen_user_split(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    """
    Load or create a group-aware unseen-user split.
    """
    if task_name not in {"subtask1", "subtask2a", "subtask2b"}:
        raise ValueError(f"Unsupported task for unseen-user split: {task_name}")

    split_path = get_splits_dir() / f"{task_name}_unseen_user_seed{seed}.json"
    if split_path.exists():
        payload = _read_json(split_path)
        train_idx = np.asarray(payload["train_indices"], dtype=int)
        val_idx = np.asarray(payload["val_indices"], dtype=int)
        return train_idx, val_idx, split_path

    groups = df["user_id"].to_numpy()
    indices = np.arange(len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    payload = {
        "task": task_name,
        "regime": "unseen_user",
        "seed": int(seed),
        "n_total": int(len(indices)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    _write_json(split_path, payload)

    return train_idx, val_idx, split_path
