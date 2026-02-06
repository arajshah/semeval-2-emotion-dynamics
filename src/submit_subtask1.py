from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.subtask1_transformer import (
    Subtask1Dataset,
    clip_preds,
    get_repo_root,
    load_hf_checkpoint,
    set_seed,
)
from src.utils.run_id import validate_run_id


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (get_repo_root() / p)


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "t", "yes", "y")
    return bool(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Subtask 1 test submission CSV from a trained checkpoint."
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Checkpoint directory. Default: models/subtask1_transformer/runs/<RUN_ID>.",
    )
    parser.add_argument(
        "--test_path",
        default="data/raw/test/test_subtask1.csv",
        help="Path to official test_subtask1.csv.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output CSV path. Default: reports/submissions/pred_subtask1__<RUN_ID>.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--scale_arousal_to_valence_range",
        action="store_true",
        help="If set, invert arousal scaling on model outputs: [-2,2] -> [0,2] via (aro+2)/2, then clip.",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=1737,
        help="Expected number of rows in test_subtask1.csv.",
    )
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    validate_run_id(args.run_id)

    repo_root = get_repo_root()

    test_path = _resolve_path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    checkpoint_dir = args.checkpoint_dir or f"models/subtask1_transformer/runs/{args.run_id}"
    checkpoint_dir = _resolve_path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    output_path = (
        _resolve_path(args.output_path)
        if args.output_path
        else (repo_root / "reports" / "submissions" / f"pred_subtask1__{args.run_id}.csv")
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite 1)")

    # ---- Gate E1 logic (matches Colab) ----
    set_seed(int(args.seed))

    df_test = pd.read_csv(test_path)

    req = ["user_id", "text_id", "text", "timestamp", "collection_phase", "is_words", "is_seen_user"]
    missing = [c for c in req if c not in df_test.columns]
    if missing:
        raise ValueError(f"test_subtask1.csv missing columns: {missing}")

    df_test["is_words"] = df_test["is_words"].map(_to_bool)
    df_test["is_seen_user"] = df_test["is_seen_user"].map(_to_bool)

    # Subtask1Dataset expects labels; add safe dummies for test
    if "valence" not in df_test.columns:
        df_test["valence"] = 0.0
    if "arousal" not in df_test.columns:
        df_test["arousal"] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_hf_checkpoint(checkpoint_dir)
    model.to(device)
    model.eval()

    ds = Subtask1Dataset(df_test.copy(), tokenizer, int(args.max_length))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(out.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)

    # If checkpoint trained with scaled arousal, invert here: [-2,2] -> [0,2]
    if bool(args.scale_arousal_to_valence_range):
        y_pred = y_pred.copy()
        y_pred[:, 1] = (y_pred[:, 1] + 2.0) / 2.0

    # Clip to theoretical bounds (valence [-2,2], arousal [0,2])
    y_pred = clip_preds(y_pred)

    out_df = pd.DataFrame(
        {
            "user_id": df_test["user_id"].to_numpy(),
            "text_id": df_test["text_id"].to_numpy(),
            "pred_valence": y_pred[:, 0].astype(float),
            "pred_arousal": y_pred[:, 1].astype(float),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # ---- Gate E2 logic (matches Colab) ----
    df_sub = pd.read_csv(output_path)

    expected_cols = ["user_id", "text_id", "pred_valence", "pred_arousal"]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(f"Bad columns/order: {list(df_sub.columns)} != {expected_cols}")

    df_test2 = pd.read_csv(test_path)
    if len(df_sub) != len(df_test2):
        raise AssertionError(f"Rowcount mismatch: sub={len(df_sub)} test={len(df_test2)}")

    if args.expected_rows is not None:
        exp_n = int(args.expected_rows)
        if len(df_sub) != exp_n:
            raise AssertionError(f"Unexpected rowcount (expected {exp_n}): got {len(df_sub)}")

    df_sub["user_id"] = pd.to_numeric(df_sub["user_id"], errors="raise").astype(int)
    df_sub["text_id"] = pd.to_numeric(df_sub["text_id"], errors="raise").astype(int)
    df_sub["pred_valence"] = pd.to_numeric(df_sub["pred_valence"], errors="raise").astype(float)
    df_sub["pred_arousal"] = pd.to_numeric(df_sub["pred_arousal"], errors="raise").astype(float)

    pred_mat = df_sub[["pred_valence", "pred_arousal"]].to_numpy(dtype=float)
    if not np.isfinite(pred_mat).all():
        raise AssertionError("Non-finite values found in predictions.")

    v = df_sub["pred_valence"].to_numpy(dtype=float)
    a = df_sub["pred_arousal"].to_numpy(dtype=float)

    vmin, vmax = float(v.min()), float(v.max())
    amin, amax = float(a.min()), float(a.max())

    if not (vmin >= -2.0 - 1e-6 and vmax <= 2.0 + 1e-6):
        raise AssertionError(f"Valence out of bounds [-2,2]: min={vmin:.6f}, max={vmax:.6f}")
    if not (amin >= 0.0 - 1e-6 and amax <= 2.0 + 1e-6):
        raise AssertionError(f"Arousal out of bounds [0,2]: min={amin:.6f}, max={amax:.6f}")

    dup = int(df_sub.duplicated(subset=["user_id", "text_id"]).sum())
    if dup != 0:
        raise AssertionError(f"Duplicate (user_id, text_id) rows found: {dup}")


if __name__ == "__main__":
    main()
