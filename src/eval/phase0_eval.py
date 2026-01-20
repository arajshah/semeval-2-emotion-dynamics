from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data_loader import load_all_data
from src.eval.analysis_tools import (
    compute_delta_metrics,
    compute_subtask1_slice_metrics,
    safe_pearsonr,
)
from src.eval.splits import get_repo_root, load_or_create_unseen_user_split


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def _get_git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _compute_config_hash(args: argparse.Namespace) -> str:
    config_path = _resolve_repo_path(args.config_path)
    payload: Dict[str, Any] = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "model_tag": args.model_tag,
        "config_path": str(config_path) if config_path else None,
        "write_per_user_diagnostics": args.write_per_user_diagnostics,
    }
    if config_path and config_path.exists():
        content_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        payload["config_file_sha256"] = content_hash

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_eval_records(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_columns: List[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
            if header:
                existing_columns = header.split(",")

    row_columns = list({key for row in rows for key in row.keys()})
    if existing_columns:
        for col in row_columns:
            if col not in existing_columns:
                existing_columns.append(col)
        columns = existing_columns
    else:
        columns = row_columns

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _per_user_subtask1(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for user_id, group in df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        if len(idx) == 0:
            continue
        rows.append(
            {
                "user_id": user_id,
                "n": int(len(idx)),
                "r_valence": safe_pearsonr(y_true[idx, 0], y_pred[idx, 0], label="per_user_valence"),
                "r_arousal": safe_pearsonr(y_true[idx, 1], y_pred[idx, 1], label="per_user_arousal"),
                "mae_valence": float(np.mean(np.abs(y_true[idx, 0] - y_pred[idx, 0]))),
                "mae_arousal": float(np.mean(np.abs(y_true[idx, 1] - y_pred[idx, 1]))),
            }
        )
    return pd.DataFrame(rows)


def _per_user_delta(
    df: pd.DataFrame,
    y_true_delta_v: np.ndarray,
    y_pred_delta_v: np.ndarray,
    y_true_delta_a: np.ndarray,
    y_pred_delta_a: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for user_id, group in df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        if len(idx) == 0:
            continue
        rows.append(
            {
                "user_id": user_id,
                "n": int(len(idx)),
                "r_delta_valence": safe_pearsonr(
                    y_true_delta_v[idx], y_pred_delta_v[idx], label="per_user_delta_valence"
                ),
                "r_delta_arousal": safe_pearsonr(
                    y_true_delta_a[idx], y_pred_delta_a[idx], label="per_user_delta_arousal"
                ),
                "mae_delta_valence": float(np.mean(np.abs(y_true_delta_v[idx] - y_pred_delta_v[idx]))),
                "mae_delta_arousal": float(np.mean(np.abs(y_true_delta_a[idx] - y_pred_delta_a[idx]))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 evaluation entrypoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default="baseline_or_external")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--write_per_user_diagnostics",
        action="store_true",
        help="Write per-user diagnostics CSVs.",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or f"phase0_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{args.seed}"
    git_commit = _get_git_commit(repo_root)
    config_hash = _compute_config_hash(args)

    data = load_all_data(data_dir=str(repo_root / "data" / "raw"))
    eval_rows: List[Dict[str, Any]] = []

    identity = {
        "run_id": run_id,
        "git_commit": git_commit,
        "seed": args.seed,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "model_tag": args.model_tag,
    }

    # Subtask 1: train-mean predictor
    df1 = data["subtask1"]
    train_idx, val_idx, _ = load_or_create_unseen_user_split(
        df1, "subtask1", seed=args.seed, val_fraction=args.val_fraction
    )
    train_df = df1.iloc[train_idx]
    val_df = df1.iloc[val_idx].reset_index(drop=True)

    mean_valence = float(train_df["valence"].mean())
    mean_arousal = float(train_df["arousal"].mean())
    y_true = val_df[["valence", "arousal"]].to_numpy(dtype=float)
    y_pred = np.column_stack(
        [
            np.full(len(val_df), mean_valence, dtype=float),
            np.full(len(val_df), mean_arousal, dtype=float),
        ]
    )

    for metrics in compute_subtask1_slice_metrics(val_df, y_true, y_pred):
        primary_score = float(
            np.mean([metrics["r_composite_valence"], metrics["r_composite_arousal"]])
        )
        eval_rows.append(
            {
                **identity,
                "task": "subtask1",
                "regime": "unseen_user",
                "slice": metrics.pop("slice"),
                "primary_score": primary_score,
                **metrics,
            }
        )

    if args.write_per_user_diagnostics:
        per_user_df = _per_user_subtask1(val_df, y_true, y_pred)
        out_path = repo_root / "reports" / f"per_user_subtask1_{run_id}.csv"
        per_user_df.to_csv(out_path, index=False)

    # Subtask 2A: delta=0 predictor
    df2a = data["subtask2a"]
    train_idx, val_idx, _ = load_or_create_unseen_user_split(
        df2a, "subtask2a", seed=args.seed, val_fraction=args.val_fraction
    )
    val_df = df2a.iloc[val_idx].copy()
    mask = val_df["state_change_valence"].notna() & val_df["state_change_arousal"].notna()
    val_df = val_df.loc[mask].reset_index(drop=True)

    y_true_delta_v = val_df["state_change_valence"].to_numpy(dtype=float)
    y_true_delta_a = val_df["state_change_arousal"].to_numpy(dtype=float)
    y_pred_delta_v = np.zeros_like(y_true_delta_v, dtype=float)
    y_pred_delta_a = np.zeros_like(y_true_delta_a, dtype=float)

    metrics = compute_delta_metrics(
        y_true_delta_v, y_pred_delta_v, y_true_delta_a, y_pred_delta_a
    )
    primary_score = float(np.mean([metrics["r_delta_valence"], metrics["r_delta_arousal"]]))
    eval_rows.append(
        {
            **identity,
            "task": "subtask2a",
            "regime": "unseen_user",
            "slice": "all",
            "primary_score": primary_score,
            **metrics,
        }
    )

    if args.write_per_user_diagnostics:
        per_user_df = _per_user_delta(
            val_df, y_true_delta_v, y_pred_delta_v, y_true_delta_a, y_pred_delta_a
        )
        out_path = repo_root / "reports" / f"per_user_subtask2a_{run_id}.csv"
        per_user_df.to_csv(out_path, index=False)

    # Subtask 2B: delta=0 predictor (user-level)
    df2b = data["subtask2b_user"]
    train_idx, val_idx, _ = load_or_create_unseen_user_split(
        df2b, "subtask2b", seed=args.seed, val_fraction=args.val_fraction
    )
    val_df = df2b.iloc[val_idx].copy()
    mask = val_df["disposition_change_valence"].notna() & val_df[
        "disposition_change_arousal"
    ].notna()
    val_df = val_df.loc[mask].reset_index(drop=True)

    y_true_delta_v = val_df["disposition_change_valence"].to_numpy(dtype=float)
    y_true_delta_a = val_df["disposition_change_arousal"].to_numpy(dtype=float)
    y_pred_delta_v = np.zeros_like(y_true_delta_v, dtype=float)
    y_pred_delta_a = np.zeros_like(y_true_delta_a, dtype=float)

    metrics = compute_delta_metrics(
        y_true_delta_v, y_pred_delta_v, y_true_delta_a, y_pred_delta_a
    )
    primary_score = float(np.mean([metrics["r_delta_valence"], metrics["r_delta_arousal"]]))
    eval_rows.append(
        {
            **identity,
            "task": "subtask2b",
            "regime": "unseen_user",
            "slice": "all",
            "primary_score": primary_score,
            **metrics,
        }
    )

    if args.write_per_user_diagnostics:
        per_user_df = _per_user_delta(
            val_df, y_true_delta_v, y_pred_delta_v, y_true_delta_a, y_pred_delta_a
        )
        out_path = repo_root / "reports" / f"per_user_subtask2b_{run_id}.csv"
        per_user_df.to_csv(out_path, index=False)

    eval_records_path = repo_root / "reports" / "eval_records.csv"
    _append_eval_records(eval_rows, eval_records_path)
    print(f"Appended {len(eval_rows)} eval records to {eval_records_path}")


if __name__ == "__main__":
    main()
