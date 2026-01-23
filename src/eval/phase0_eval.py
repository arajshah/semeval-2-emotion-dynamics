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
    compute_subtask1_metrics_from_preds,
    compute_subtask1_slice_metrics,
    load_frozen_split,
    make_seen_user_time_split,
    safe_pearsonr,
)
from src.eval.splits import get_repo_root


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
        "task": args.task,
        "regime": args.regime,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "model_tag": args.model_tag,
        "run_id": args.run_id,
        "pred_path": args.pred_path,
        "pred_dir": args.pred_dir,
        "split_path": args.split_path,
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
    
    CANONICAL_COLUMNS = [
        "run_id",
        "git_commit",
        "seed",
        "timestamp",
        "config_hash",
        "task",
        "regime",
        "slice",
        "model_tag",
        "primary_score",
        "r_composite_valence",
        "r_within_valence",
        "r_between_valence",
        "r_composite_arousal",
        "r_within_arousal",
        "r_between_arousal",
        "r_delta_valence",
        "r_delta_arousal",
        "mae_valence",
        "mae_arousal",
        "mse_valence",
        "mse_arousal",
        "mae_delta_valence",
        "mae_delta_arousal",
    ]

    columns = CANONICAL_COLUMNS

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized = {col: row.get(col, "") for col in columns}
        normalized_rows.append(normalized)

    df = pd.DataFrame(normalized_rows, columns=columns)
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
    parser.add_argument("--task", choices=["subtask1", "subtask2a", "subtask2b"], default="subtask1")
    parser.add_argument("--regime", choices=["unseen_user", "seen_user"], default="unseen_user")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default="reports/preds")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--write_per_user_diagnostics",
        action="store_true",
        help="Write per-user diagnostics CSVs.",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = args.run_id or ""
    if model_tag_is_transformer and not run_id and pred_path is None:
        raise SystemExit("Transformer eval requires --run_id when --pred_path is not provided.")
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

    model_tag_is_transformer = "transformer" in args.model_tag.lower()
    if model_tag_is_transformer and args.task != "subtask1":
        raise SystemExit("Transformer eval routing is implemented only for subtask1.")

    pred_path: Path | None = _resolve_repo_path(args.pred_path)
    split_path = _resolve_repo_path(args.split_path)
    if split_path is None and args.regime == "unseen_user":
        split_path = repo_root / "reports" / "splits" / f"{args.task}_unseen_user_seed{args.seed}.json"

    if model_tag_is_transformer:
        if (pred_path is None) == (args.run_id is None):
            # Exactly one way to identify the preds: either explicit --pred_path OR (run_id + pred_dir)
            if (pred_path is not None) and (args.run_id is not None):
                raise SystemExit("Provide either --pred_path OR --run_id (not both).")
            if (pred_path is None) and (args.run_id is None):
                raise SystemExit("Transformer eval requires either --pred_path or --run_id.")
        if pred_path is None:
            pred_path = repo_root / args.pred_dir / f"subtask1_val_preds__{args.run_id}.parquet"
        if pred_path is None or not pred_path.exists():
            raise SystemExit(f"Predictions file not found: {pred_path}")

    if args.task == "subtask1":
        df1 = data["subtask1"]
        if args.regime == "unseen_user":
            if split_path is not None and split_path.exists():
                train_idx, val_idx = load_frozen_split(split_path, df1)
            else:
                raise SystemExit(f"Split file not found: {split_path}")
        else:
            train_idx, val_idx = make_seen_user_time_split(df1, val_frac=0.2)

        if model_tag_is_transformer:
            if split_path is None or not split_path.exists():
                raise SystemExit(f"Split file not found: {split_path}")
            preds_df = pd.read_parquet(pred_path)

            required_cols = {"idx", "valence_pred", "arousal_pred"}
            missing = required_cols - set(preds_df.columns)
            if missing:
                raise SystemExit(f"Preds missing required columns {sorted(missing)} in {pred_path}")

            # Range sanity (fails fast if you forgot to invert scaled arousal)
            aro = preds_df["arousal_pred"].to_numpy(dtype=float)
            if np.nanmin(aro) < -0.25 or np.nanmax(aro) > 2.25:
                raise SystemExit(
                    f"arousal_pred out of expected [0,2] range in {pred_path} "
                    f"(min={np.nanmin(aro):.3f}, max={np.nanmax(aro):.3f}). "
                    "Did you forget --scale_arousal_to_valence_range on predict?"
                )

            metrics_rows = compute_subtask1_metrics_from_preds(df1, preds_df, np.asarray(val_idx))
            for metrics in metrics_rows:
                primary_score = float(
                    np.mean([metrics["r_composite_valence"], metrics["r_composite_arousal"]])
                )
                eval_rows.append(
                    {
                        **identity,
                        "task": "subtask1",
                        "regime": args.regime,
                        "slice": metrics.pop("slice"),
                        "primary_score": primary_score,
                        **metrics,
                    }
                )
        else:
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
                        "regime": args.regime,
                        "slice": metrics.pop("slice"),
                        "primary_score": primary_score,
                        **metrics,
                    }
                )

            if args.write_per_user_diagnostics:
                per_user_df = _per_user_subtask1(val_df, y_true, y_pred)
                out_path = repo_root / "reports" / f"per_user_subtask1_{run_id}.csv"
                per_user_df.to_csv(out_path, index=False)

    if args.task == "subtask2a":
        df2a = data["subtask2a"]
        if args.regime == "unseen_user":
            split_path_2a = split_path
            if split_path_2a is None:
                split_path_2a = repo_root / "reports" / "splits" / f"subtask2a_unseen_user_seed{args.seed}.json"
            if not split_path_2a.exists():
                raise SystemExit(f"Split file not found: {split_path_2a}")
            train_idx, val_idx = load_frozen_split(split_path_2a, df2a)
        else:
            train_idx, val_idx = make_seen_user_time_split(df2a, val_frac=0.2)
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
                "regime": args.regime,
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

    if args.task == "subtask2b":
        df2b = data["subtask2b_user"]

        split_path_2b = repo_root / "reports" / "splits" / f"subtask2b_user_disposition_change_unseen_user_seed{args.seed}.json"
        if not split_path_2b.exists():
            raise SystemExit(f"Split file not found: {split_path_2b}")
        train_idx, val_idx = load_frozen_split(split_path_2b, df2b)

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
                "regime": args.regime,
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
