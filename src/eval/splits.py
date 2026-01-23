from __future__ import annotations

import argparse
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


def split_filename(task: str, regime: str, seed: int, split_key: str | None = None) -> str:
    if task == "subtask2b" and split_key:
        return f"subtask2b_{split_key}_{regime}_seed{seed}.json"
    return f"{task}_{regime}_seed{seed}.json"


def get_split_path(task: str, regime: str, seed: int, split_key: str | None = None) -> Path:
    return get_splits_dir() / split_filename(task, regime, seed, split_key=split_key)


def validate_split_payload(
    payload: dict,
    *,
    task: str,
    regime: str,
    seed: int,
    n_total: int,
    split_key: str | None = None,
) -> None:
    required = {"train_indices", "val_indices", "n_total", "task", "seed", "regime"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Split payload missing keys: {sorted(missing)}")

    if payload["task"] != task:
        raise ValueError(f"Split task mismatch: {payload['task']} != {task}")
    if payload["seed"] != seed:
        raise ValueError(f"Split seed mismatch: {payload['seed']} != {seed}")
    if payload["regime"] != regime:
        raise ValueError(f"Split regime mismatch: {payload['regime']} != {regime}")
    if payload["n_total"] != n_total:
        raise ValueError(
            f"Split n_total mismatch: {payload['n_total']} != {n_total} (variant mismatch)"
        )
    if split_key is not None and payload.get("split_key") != split_key:
        raise ValueError(f"Split key mismatch: {payload.get('split_key')} != {split_key}")

    train_idx = payload["train_indices"]
    val_idx = payload["val_indices"]
    if not train_idx or not val_idx:
        raise ValueError("Split indices must be non-empty.")

    if any((i < 0 or i >= n_total) for i in train_idx + val_idx):
        raise ValueError("Split indices out of bounds.")

    overlap = set(train_idx).intersection(set(val_idx))
    if overlap:
        raise ValueError(f"Split indices overlap (count={len(overlap)}).")


def validate_unseen_user_disjoint(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    user_col: str = "user_id",
) -> None:
    train_users = set(df.loc[train_idx, user_col])
    val_users = set(df.loc[val_idx, user_col])
    overlap = train_users.intersection(val_users)
    if overlap:
        raise ValueError(f"User overlap detected across train/val (count={len(overlap)}).")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_unseen_user_split(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    *,
    split_key: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    if task_name not in {"subtask1", "subtask2a", "subtask2b"}:
        raise ValueError(f"Unsupported task for unseen-user split: {task_name}")

    split_path = get_split_path(task_name, "unseen_user", seed, split_key=split_key)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    payload = _read_json(split_path)
    validate_split_payload(
        payload,
        task=task_name,
        regime="unseen_user",
        seed=seed,
        n_total=len(df),
        split_key=split_key,
    )
    train_idx = np.asarray(payload["train_indices"], dtype=int)
    val_idx = np.asarray(payload["val_indices"], dtype=int)
    validate_unseen_user_disjoint(df, train_idx.tolist(), val_idx.tolist())
    return train_idx, val_idx, split_path


def create_unseen_user_split(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    *,
    val_fraction: float = 0.2,
    split_key: str | None = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    if task_name not in {"subtask1", "subtask2a", "subtask2b"}:
        raise ValueError(f"Unsupported task for unseen-user split: {task_name}")

    split_path = get_split_path(task_name, "unseen_user", seed, split_key=split_key)
    if split_path.exists() and not overwrite:
        # If already exists, just load+validate and return.
        return load_unseen_user_split(df, task_name, seed, split_key=split_key)

    groups = df["user_id"].to_numpy()
    indices = np.arange(len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    payload: dict[str, Any] = {
        "task": task_name,
        "regime": "unseen_user",
        "seed": int(seed),
        "n_total": int(len(indices)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_fraction": float(val_fraction),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    if split_key is not None:
        payload["split_key"] = split_key

    validate_split_payload(
        payload,
        task=task_name,
        regime="unseen_user",
        seed=seed,
        n_total=len(df),
        split_key=split_key,
    )
    validate_unseen_user_disjoint(df, train_idx.tolist(), val_idx.tolist())
    _write_json(split_path, payload)

    return train_idx, val_idx, split_path


def load_or_create_unseen_user_split(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    val_fraction: float = 0.2,
    split_key: str | None = None,
    *,
    create_if_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    if create_if_missing:
        return create_unseen_user_split(
            df,
            task_name,
            seed,
            val_fraction=val_fraction,
            split_key=split_key,
            overwrite=False,
        )
    return load_unseen_user_split(df, task_name, seed, split_key=split_key)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen splits.")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--variant",
        choices=["base", "user_disposition_change", "detailed"],
        default="base",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (overrides --seed). Example: 43,44",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from src.data_loader import load_all_data

    data = load_all_data()
    split_key = None
    if args.task == "subtask2b":
        if args.variant == "base":
            df = data["subtask2b"]
            split_key = None
        elif args.variant == "user_disposition_change":
            df = data["subtask2b_user"]
            split_key = "user_disposition_change"
        else:
            df = data["subtask2b_detailed"]
            split_key = "detailed"
    else:
        df = data[args.task]

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        _, _, split_path = create_unseen_user_split(
            df,
            args.task,
            seed=seed,
            val_fraction=args.val_fraction,
            split_key=split_key,
            overwrite=bool(args.overwrite),
        )
        print(f"Split path: {split_path}")


if __name__ == "__main__":
    _cli()
