from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
import joblib
from tqdm.auto import tqdm

from src.subtask2b_features import (
    load_subtask2b_df_raw,
    load_subtask2b_embeddings_npz,
    merge_embeddings,
    build_user_level_dataset,
)
from src.utils.git_utils import get_git_commit
from src.utils.provenance import merge_run_metadata


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
    parser = argparse.ArgumentParser(description="Train Subtask 2B ridge baseline.")
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
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    split_path = Path(args.split_path) if args.split_path else (
        repo_root / "reports" / "splits" / f"subtask2b_unseen_user_seed{args.seed}.json"
    )

    with tqdm(total=9, desc="Subtask2B ridge", unit="step") as pbar:
        if not split_path.exists():
            raise SystemExit(f"Split file not found: {split_path}")
        pbar.update(1)

        df_raw = load_subtask2b_df_raw()
        pbar.update(1)

        train_idx, val_idx = _load_split_indices(split_path)
        train_df_raw = df_raw.iloc[train_idx].copy()
        val_df_raw = df_raw.iloc[val_idx].copy()
        pbar.update(1)

        emb_map_df, embeddings = load_subtask2b_embeddings_npz(args.embeddings_path)
        pbar.update(1)

        train_df = merge_embeddings(train_df_raw, emb_map_df)
        train_df.attrs["embeddings"] = embeddings
        pbar.update(1)

        train_users = train_df_raw["user_id"].drop_duplicates(keep="first").to_numpy()
        X_train, y_train, users_train = build_user_level_dataset(train_df, train_users)
        if len(users_train) == 0:
            raise SystemExit("No eligible training users with group==1 rows.")
        pbar.update(1)

        model = Ridge(alpha=args.alpha)
        model.fit(X_train, y_train)
        pbar.update(1)

        out_dir = repo_root / "models" / "subtask2b_user" / "runs" / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)
        pbar.update(1)

        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "task": "subtask2b",
                "task_tag": "subtask2b_ridge",
                "seed": args.seed,
                "split_path": str(split_path),
                "embeddings_path": str(args.embeddings_path),
                "alpha": args.alpha,
                "group_rule": "group==1 only",
                "label_cols": ["disposition_change_valence", "disposition_change_arousal"],
                "train_users": int(len(users_train)),
                "val_users": int(val_df_raw["user_id"].nunique()),
                "git_commit": get_git_commit(repo_root),
            },
        )
        pbar.update(1)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
