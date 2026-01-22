from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_all_data
from src.eval.analysis_tools import load_frozen_split
from src.models.subtask1_transformer import (
    Subtask1Dataset,
    clip_preds,
    get_repo_root,
    load_hf_checkpoint,
    set_seed,
)
from src.utils.provenance import write_run_metadata
from src.utils.run_id import resolve_run_id, validate_run_id


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def _predict_df(
    df: pd.DataFrame,
    checkpoint_dir: Path,
    batch_size: int,
    max_length: int,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    model, tokenizer = load_hf_checkpoint(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = Subtask1Dataset(df.copy(), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(outputs.cpu().numpy())
    return clip_preds(np.concatenate(preds, axis=0))


def predict_subtask1_df(
    df: pd.DataFrame,
    checkpoint_dir: str | Path | None = None,
    batch_size: int = 16,
    max_length: int | None = None,
    seed: int = 42,
    cache_path: str | Path = "reports/subtask1_transformer_preds.parquet",
    use_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    repo_root = get_repo_root()
    cache_path = _resolve_path(cache_path)

    if checkpoint_dir is None:
        metrics_path = repo_root / "models" / "subtask1_transformer" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"metrics.json not found at {metrics_path}. "
                "Provide checkpoint_dir explicitly or run training to create metrics.json."
            )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        best_ckpt = payload.get("best_checkpoint_dir")
        best_run_id = payload.get("best_run_id")

        if best_ckpt:
            checkpoint_dir = best_ckpt
        elif best_run_id:
            checkpoint_dir = f"models/subtask1_transformer/runs/{best_run_id}"
        else:
            raise ValueError(
                f"metrics.json at {metrics_path} missing best_checkpoint_dir and best_run_id."
            )

    checkpoint_dir = _resolve_path(checkpoint_dir)

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if (
            "row_index" in cached.columns
            and len(cached) == len(df)
            and np.array_equal(cached["row_index"].to_numpy(), df.index.to_numpy())
        ):
            return cached["pred_valence"].to_numpy(), cached["pred_arousal"].to_numpy()

    effective_max_length = 256 if max_length is None else int(max_length)
    y_pred = _predict_df(df, checkpoint_dir, batch_size, effective_max_length, seed)
    pred_valence = y_pred[:, 0]
    pred_arousal = y_pred[:, 1]

    model_tag = f"{checkpoint_dir.name}_maxlen{effective_max_length}_seed{seed}"
    cache_df = pd.DataFrame(
        {
            "row_index": df.index.to_numpy(),
            "pred_valence": pred_valence,
            "pred_arousal": pred_arousal,
            "model_tag": model_tag,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)

    return pred_valence, pred_arousal


def predict_subtask1_val_split(
    split_path: Path,
    ckpt_dir: Path,
    output_path: Optional[Path],
    output_user_agg_path: Optional[Path],
    batch_size: int,
    max_length: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_all_data()
    df = data["subtask1"].copy().reset_index(drop=True)
    _, val_idx = load_frozen_split(split_path, df)
    val_idx = np.sort(np.asarray(val_idx, dtype=int))
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    y_pred = _predict_df(val_df, ckpt_dir, batch_size, max_length, seed)
    preds_df = pd.DataFrame(
        {
            "idx": val_idx,
            "user_id": val_df["user_id"].to_numpy(),
            "is_words": val_df["is_words"].to_numpy(),
            "valence_true": val_df["valence"].to_numpy(dtype=float),
            "arousal_true": val_df["arousal"].to_numpy(dtype=float),
            "valence_pred": y_pred[:, 0],
            "arousal_pred": y_pred[:, 1],
        }
    )

    user_agg = (
        preds_df.groupby("user_id", sort=False)
        .agg(
            valence_true_mean=("valence_true", "mean"),
            arousal_true_mean=("arousal_true", "mean"),
            valence_pred_mean=("valence_pred", "mean"),
            arousal_pred_mean=("arousal_pred", "mean"),
            n_rows=("idx", "count"),
        )
        .reset_index()
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_parquet(output_path, index=False)
    if output_user_agg_path is not None:
        output_user_agg_path.parent.mkdir(parents=True, exist_ok=True)
        user_agg.to_parquet(output_user_agg_path, index=False)

    return preds_df, user_agg


def _load_preds(path: Path) -> pd.DataFrame:
    preds_df = pd.read_parquet(path)
    required = {"idx", "valence_pred", "arousal_pred"}
    if not required.issubset(set(preds_df.columns)):
        raise ValueError(f"Predictions at {path} missing required columns: {required}")
    return preds_df.sort_values("idx").reset_index(drop=True)


def ensemble_predictions(
    preds_list: Iterable[pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    preds_list = list(preds_list)
    if not preds_list:
        raise ValueError("No predictions provided for ensembling.")

    base = preds_list[0]
    base_idx = base["idx"].to_numpy(dtype=int)
    for df in preds_list[1:]:
        if not np.array_equal(base_idx, df["idx"].to_numpy(dtype=int)):
            raise ValueError("Prediction indices do not match across ensemble inputs.")

    avg_val = np.mean([df["valence_pred"].to_numpy(dtype=float) for df in preds_list], axis=0)
    avg_aro = np.mean([df["arousal_pred"].to_numpy(dtype=float) for df in preds_list], axis=0)

    ensemble_df = base.copy()
    ensemble_df["valence_pred"] = avg_val
    ensemble_df["arousal_pred"] = avg_aro

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_parquet(output_path, index=False)
    return ensemble_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Subtask 1 transformer outputs.")
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--infer_run_id_from_metrics", action="store_true")
    parser.add_argument("--pred_dir", default="reports/preds")
    parser.add_argument("--task", default="subtask1")
    parser.add_argument("--task_tag", default="subtask1_transformer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble_pred_paths", default=None)
    parser.add_argument("--ensemble_ckpt_dirs", default=None)
    parser.add_argument(
        "--ensemble_output_path",
        default="reports/subtask1_transformer_ensemble_preds.parquet",
    )
    args = parser.parse_args()

    split_path = _resolve_path(args.split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Frozen split JSON not found at {split_path}")

    run_id = args.run_id
    best_checkpoint_dir = None
    if run_id is None and args.infer_run_id_from_metrics:
        metrics_path = _resolve_path("models/subtask1_transformer/metrics.json")
        if not metrics_path.exists():
            print("metrics.json not found; cannot infer run_id.", file=sys.stderr)
            raise SystemExit(1)
        payload = json.loads(metrics_path.read_text())
        run_id = payload.get("best_run_id")
        best_checkpoint_dir = payload.get("best_checkpoint_dir")
        if not run_id:
            print("best_run_id missing in metrics.json.", file=sys.stderr)
            raise SystemExit(1)

    if run_id is None:
        print("run_id is required (or use --infer_run_id_from_metrics).", file=sys.stderr)
        raise SystemExit(1)

    run_id = resolve_run_id(run_id, args.task_tag, seed=args.seed)
    validate_run_id(run_id)

    if args.ensemble_pred_paths or args.ensemble_ckpt_dirs:
        pred_paths = []
        if args.ensemble_pred_paths:
            pred_paths = [p.strip() for p in args.ensemble_pred_paths.split(",") if p.strip()]
        preds_list = []
        if pred_paths:
            preds_list = [_load_preds(_resolve_path(p)) for p in pred_paths]
        else:
            ckpt_dirs = [p.strip() for p in (args.ensemble_ckpt_dirs or "").split(",") if p.strip()]
            if not ckpt_dirs:
                raise ValueError("Provide --ensemble_pred_paths or --ensemble_ckpt_dirs.")
            for ckpt in ckpt_dirs:
                preds_df, _ = predict_subtask1_val_split(
                    split_path=split_path,
                    ckpt_dir=_resolve_path(ckpt),
                    output_path=None,
                    output_user_agg_path=None,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    seed=args.seed,
                )
                preds_list.append(preds_df.sort_values("idx").reset_index(drop=True))

        ensemble_output = _resolve_path(args.ensemble_output_path)
        ensemble_predictions(preds_list, ensemble_output)
        print(f"Wrote ensemble predictions to {ensemble_output}")
        return

    pred_dir = _resolve_path(args.pred_dir)
    preds_path = pred_dir / f"subtask1_val_preds__{run_id}.parquet"
    agg_path = pred_dir / f"subtask1_val_user_agg__{run_id}.parquet"

    checkpoint_dir = args.checkpoint_dir or args.ckpt_dir
    if checkpoint_dir is None:
        if best_checkpoint_dir:
            checkpoint_dir = best_checkpoint_dir
        else:
            checkpoint_dir = f"models/subtask1_transformer/runs/{run_id}"
    checkpoint_dir = _resolve_path(checkpoint_dir)

    print(f"RUN_ID: {run_id}")
    print(f"CHECKPOINT_DIR: {checkpoint_dir}")
    print(f"PRED_DIR: {pred_dir}")

    predict_subtask1_val_split(
        split_path=split_path,
        ckpt_dir=checkpoint_dir,
        output_path=preds_path,
        output_user_agg_path=agg_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    print(f"Wrote per-row predictions to {preds_path}")
    print(f"Wrote per-user aggregates to {agg_path}")

    repo_root = get_repo_root()
    artifacts = {
        "subtask1_val_preds": str(preds_path.relative_to(repo_root)),
        "subtask1_val_user_agg": str(agg_path.relative_to(repo_root)),
        "checkpoint_dir": str(checkpoint_dir.relative_to(repo_root)),
        "split_path": str(split_path.relative_to(repo_root)),
    }
    config = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "checkpoint_dir": str(checkpoint_dir),
    }
    write_run_metadata(
        repo_root=repo_root,
        run_id=run_id,
        task=args.task,
        task_tag=args.task_tag,
        seed=args.seed,
        regime="unseen_user",
        split_path=str(split_path.relative_to(repo_root)),
        config=config,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()
