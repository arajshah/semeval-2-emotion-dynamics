from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.data_loader import load_all_data
from src.eval.analysis_tools import safe_pearsonr
from src.models.subtask2b_user import Subtask2BUserMLP, save_checkpoint
from src.subtask2b_features import (
    audit_disposition_labels,
    build_subtask2b_user_features,
    fit_norm_stats,
    apply_norm_stats,
    load_subtask2b_embeddings_npz,
)
from src.utils.git_utils import get_git_commit
from src.utils.provenance import (
    merge_run_metadata,
    sha256_file,
    artifact_ref,
    get_git_snapshot,
    get_env_snapshot,
)
from src.utils.run_id import validate_run_id


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


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False


def _prepare_targets(df_user_raw: pd.DataFrame, users: List[str]) -> np.ndarray:
    if df_user_raw["user_id"].duplicated().any():
        raise RuntimeError("df_user_raw must contain unique user_id rows.")
    label_map = df_user_raw.set_index("user_id")[
        ["disposition_change_valence", "disposition_change_arousal"]
    ]
    rows = []
    missing = []
    for uid in tqdm(users, desc="Preparing targets", unit="user"):
        if uid not in label_map.index:
            missing.append(uid)
            continue
        rows.append(label_map.loc[uid].to_numpy(dtype=np.float32))
    if missing:
        raise RuntimeError(f"Missing labels for users (sample): {missing[:10]}")
    return np.stack(rows, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Subtask 2B Phase-D user model.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_path",
        default=None,
        help="Frozen split JSON for subtask2b_user_disposition_change.",
    )
    parser.add_argument(
        "--emb_path",
        default="data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    parser.add_argument("--pooling", choices=["mean", "last", "mean_last"], default="mean")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lambda_arousal", type=float, default=1.0)
    parser.add_argument("--audit_labels", type=int, default=1)
    parser.add_argument("--fail_on_audit_mismatch", type=int, default=0)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    validate_run_id(args.run_id)
    _set_seed(args.seed)

    repo_root = Path(".").resolve()
    split_path = Path(
        args.split_path
        or f"reports/splits/subtask2b_user_disposition_change_unseen_user_seed{args.seed}.json"
    )
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}")

    data = load_all_data()
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2b",
            "stage": "train",
            "seed": args.seed,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "inputs": {
                "split": artifact_ref(split_path, repo_root),
                "embeddings": artifact_ref(Path(args.emb_path), repo_root),
            },
            "git": get_git_snapshot(repo_root),
            "env": get_env_snapshot(),
        },
    )
    df_user_raw = data["subtask2b_user"]
    df_text_raw = data["subtask2b"]

    if args.audit_labels:
        summary = audit_disposition_labels(df_text_raw, df_user_raw)
        if args.fail_on_audit_mismatch and (
            summary["n_mismatch_valence"] > 0 or summary["n_mismatch_arousal"] > 0
        ):
            raise SystemExit("Audit mismatch detected; refusing to train.")

    train_idx, val_idx = _load_split_indices(split_path)
    train_users_df = df_user_raw.iloc[train_idx].copy()
    val_users_df = df_user_raw.iloc[val_idx].copy()

    emb_map_df, emb_arr = load_subtask2b_embeddings_npz(args.emb_path)
    embeddings = (emb_map_df, emb_arr)

    X_emb_train, X_num_train, users_train, meta_train = build_subtask2b_user_features(
        train_users_df, df_text_raw, embeddings, pooling=args.pooling
    )
    X_emb_val, X_num_val, users_val, meta_val = build_subtask2b_user_features(
        val_users_df, df_text_raw, embeddings, pooling=args.pooling
    )
    if len(users_train) == 0 or len(users_val) == 0:
        raise SystemExit("No eligible users with group==1 rows in train/val.")

    y_train = _prepare_targets(df_user_raw, users_train)
    y_val = _prepare_targets(df_user_raw, users_val)

    norm_stats = fit_norm_stats(X_num_train)
    X_num_train = apply_norm_stats(X_num_train, norm_stats)
    X_num_val = apply_norm_stats(X_num_val, norm_stats)

    X_train = np.concatenate([X_emb_train, X_num_train], axis=1).astype(np.float32)
    X_val = np.concatenate([X_emb_val, X_num_val], axis=1).astype(np.float32)

    run_dir = repo_root / "models" / "subtask2b_user" / "runs" / args.run_id
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"Run directory already exists: {run_dir}")

    model = Subtask2BUserMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_score = -1e9
    best_state = None
    best_epoch = 0

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds[:, 0], y[:, 0]) + args.lambda_arousal * loss_fn(
                preds[:, 1], y[:, 1]
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x)
            total_count += len(x)

        model.eval()
        with torch.no_grad():
            preds_val = model(torch.from_numpy(X_val)).numpy()
        r_v = safe_pearsonr(y_val[:, 0], preds_val[:, 0], label="val_dispo_valence")
        r_a = safe_pearsonr(y_val[:, 1], preds_val[:, 1], label="val_dispo_arousal")
        primary = float(np.mean([r_v, r_a]))
        avg_loss = total_loss / max(total_count, 1)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} rV={r_v:.4f} rA={r_a:.4f} primary={primary:.4f}"
        )
        if primary > best_score:
            best_score = primary
            best_epoch = epoch
            best_state = model.state_dict()

    if best_state is None:
        raise SystemExit("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    config = {
        "input_dim": int(X_train.shape[1]),
        "hidden_dims": (512, 256, 128),
        "dropout": 0.2,
        "pooling": args.pooling,
    }
    save_checkpoint(run_dir, model, config, norm_stats)

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2b",
            "task_tag": "subtask2b_phaseD",
            "seed": args.seed,
            "split_path": str(split_path.relative_to(repo_root)),
            "embeddings_path": str(Path(args.emb_path)),
            "embeddings_sha256": sha256_file(Path(args.emb_path)),
            "pooling": args.pooling,
            "group_rule": "group==1 only",
            "best_epoch": best_epoch,
            "best_primary_score": best_score,
            "artifacts": {
                "model": artifact_ref(run_dir / "model.pt", repo_root),
                "norm_stats": artifact_ref(run_dir / "norm_stats.json", repo_root),
                "config": artifact_ref(run_dir / "config.json", repo_root),
            },
            "git_commit": get_git_commit(repo_root),
            "train_users": int(len(users_train)),
            "val_users": int(len(users_val)),
            "counts": {
                "n_train_users": int(len(users_train)),
                "n_val_users": int(len(users_val)),
            },
            "metrics": {
                "train_selection": {
                    "best_epoch": best_epoch,
                    "best_primary_score": best_score,
                }
            },
            "config": {
                "toggles": {
                    "subtask2b": {"pooling": args.pooling}
                }
            },
        },
    )

    tqdm.write(f"Saved best model to: {run_dir}")


if __name__ == "__main__":
    main()
