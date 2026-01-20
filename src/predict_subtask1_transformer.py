from __future__ import annotations

import argparse
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
    checkpoint_dir: str | Path = "models/subtask1_transformer/best",
    batch_size: int = 16,
    max_length: int | None = None,
    seed: int = 42,
    cache_path: str | Path = "reports/subtask1_transformer_preds.parquet",
    use_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    repo_root = get_repo_root()
    checkpoint_dir = _resolve_path(checkpoint_dir)
    cache_path = _resolve_path(cache_path)

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
    parser.add_argument("--ckpt_dir", default="models/subtask1_transformer/best")
    parser.add_argument(
        "--output_path",
        default="reports/subtask1_transformer_val_preds.parquet",
    )
    parser.add_argument(
        "--output_user_agg_path",
        default="reports/subtask1_transformer_val_user_agg.parquet",
    )
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

    preds_path = _resolve_path(args.output_path)
    agg_path = _resolve_path(args.output_user_agg_path)
    predict_subtask1_val_split(
        split_path=split_path,
        ckpt_dir=_resolve_path(args.ckpt_dir),
        output_path=preds_path,
        output_user_agg_path=agg_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    print(f"Wrote per-row predictions to {preds_path}")
    print(f"Wrote per-user aggregates to {agg_path}")


if __name__ == "__main__":
    main()
