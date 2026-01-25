from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.subtask2b_features import (
    load_subtask2b_df_raw,
    load_subtask2b_embeddings_npz,
    merge_embeddings,
    build_user_level_dataset,
)


def _load_split_indices(path: Path) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text())
    for train_key, val_key in [
        ("train_indices", "val_indices"),
        ("train_idx", "val_idx"),
        ("train", "val"),
    ]:
        if train_key in payload and val_key in payload:
            return [int(i) for i in payload[train_key]], [int(i) for i in payload[val_key]]
    raise ValueError(f"Unsupported split schema in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Subtask 2B ridge baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--embeddings_path",
        default="data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help="Path to frozen split JSON (unseen-user).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to model joblib. Defaults to models/subtask2b_user/runs/{run_id}/model.joblib",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    split_path = Path(args.split_path) if args.split_path else (
        repo_root / "reports" / "splits" / f"subtask2b_unseen_user_seed{args.seed}.json"
    )
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}")

    model_path = Path(args.model_path) if args.model_path else (
        repo_root / "models" / "subtask2b_user" / "runs" / args.run_id / "model.joblib"
    )
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    df_raw = load_subtask2b_df_raw()
    _, val_idx = _load_split_indices(split_path)
    val_df_raw = df_raw.iloc[val_idx].copy()

    emb_map_df, embeddings = load_subtask2b_embeddings_npz(args.embeddings_path)
    val_df = merge_embeddings(val_df_raw, emb_map_df)
    val_df.attrs["embeddings"] = embeddings

    val_users = val_df_raw["user_id"].drop_duplicates(keep="first").to_numpy()
    X_val, y_val, users_val = build_user_level_dataset(val_df, val_users)
    if len(users_val) == 0:
        raise SystemExit("No eligible validation users with group==1 rows.")

    model = joblib.load(model_path)
    preds = model.predict(X_val)

    out_df = pd.DataFrame(
        {
            "run_id": args.run_id,
            "seed": int(args.seed),
            "user_id": users_val,
            "disposition_change_valence_true": y_val[:, 0],
            "disposition_change_arousal_true": y_val[:, 1],
            "disposition_change_valence_pred": preds[:, 0],
            "disposition_change_arousal_pred": preds[:, 1],
        }
    )

    if out_df["user_id"].duplicated().any():
        raise SystemExit("Preds must contain exactly one row per user.")

    preds_dir = repo_root / "reports" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out_path = preds_dir / f"subtask2b_val_user_preds__{args.run_id}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote preds to: {out_path}")


if __name__ == "__main__":
    main()
