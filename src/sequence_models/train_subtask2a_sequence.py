from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.data_loader import load_all_data
from src.eval.analysis_tools import safe_pearsonr
from src.eval.splits import load_frozen_split
from src.sequence_models.subtask2a_sequence_dataset import (
    build_subtask2a_step_dataset,
    build_subtask2a_anchor_features,
    select_latest_eligible_anchors,
)
from src.sequence_models.simple_sequence_model import SequenceStateRegressor
from src.utils.provenance import (
    merge_run_metadata,
    sha256_file,
    artifact_ref,
    get_git_snapshot,
    get_env_snapshot,
)
from src.utils.run_id import validate_run_id


def _set_determinism(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _make_loader(
    X_seq: np.ndarray,
    lengths: np.ndarray,
    X_num: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(lengths),
        torch.from_numpy(X_num),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _eval_anchors(
    model: SequenceStateRegressor,
    X_seq: np.ndarray,
    lengths: np.ndarray,
    X_num: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    preds_list: List[np.ndarray] = []
    loss_fn = nn.SmoothL1Loss()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = start + batch_size
            x_seq = torch.from_numpy(X_seq[start:end]).to(device)
            lens = torch.from_numpy(lengths[start:end]).to(device)
            x_num = torch.from_numpy(X_num[start:end]).to(device)
            outputs = model(x_seq, lens, x_num)
            preds_list.append(outputs.cpu().numpy())
            y_batch = torch.from_numpy(y_true[start:end]).to(device)
            loss = loss_fn(outputs, y_batch)
            total_loss += float(loss.item()) * len(x_seq)
            total_count += len(x_seq)
    preds = np.concatenate(preds_list, axis=0)
    r_v = safe_pearsonr(y_true[:, 0], preds[:, 0], label="val_delta_valence")
    r_a = safe_pearsonr(y_true[:, 1], preds[:, 1], label="val_delta_arousal")
    val_loss = total_loss / max(total_count, 1)
    return r_v, r_a, val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Subtask 2A Phase-C sequence model.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--k_state", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--ablate_no_history", type=int, default=0)
    args = parser.parse_args()

    validate_run_id(args.run_id)
    generator = _set_determinism(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    repo_root = Path(".").resolve()
    split_path = Path(
        args.split_path
        or f"reports/splits/subtask2a_unseen_user_seed{args.seed}.json"
    )
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.is_absolute():
        embeddings_path = repo_root / embeddings_path

    args_payload = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "stage": "train",
            "seed": args.seed,
            "args": args_payload,
            "inputs": {
                "split": artifact_ref(split_path, repo_root),
                "embeddings": artifact_ref(embeddings_path, repo_root),
            },
            "git": get_git_snapshot(repo_root),
            "env": get_env_snapshot(str(device)),
        },
    )

    data = load_all_data()
    df_raw = data["subtask2a"]
    train_idx, val_idx = load_frozen_split(split_path, df_raw)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    resolved_seq_len = int(args.seq_len) if args.seq_len is not None else 5
    train_bundle = build_subtask2a_step_dataset(
        df_raw=df_raw,
        split_idx=train_idx,
        embeddings_path=embeddings_path,
        seq_len=resolved_seq_len,
        k_state=args.k_state,
        fit_norm=True,
        norm_stats=None,
        ablate_no_history=bool(args.ablate_no_history),
    )
    norm_stats = train_bundle["norm_stats"]

    val_df_raw = df_raw.iloc[val_idx].copy()
    val_df_raw["idx"] = np.asarray(val_idx, dtype=int)
    anchors_df = select_latest_eligible_anchors(val_df_raw)
    if anchors_df.empty:
        raise SystemExit("No eligible val anchors found.")
    anchors_df = anchors_df.rename(
        columns={
            "text_id": "anchor_text_id",
            "timestamp": "anchor_timestamp",
        }
    )

    val_anchor_bundle = build_subtask2a_anchor_features(
        df_raw=df_raw,
        anchors_df=anchors_df[["anchor_idx"]],
        embeddings_path=embeddings_path,
        seq_len=resolved_seq_len,
        k_state=args.k_state,
        norm_stats=norm_stats,
        ablate_no_history=bool(args.ablate_no_history),
    )

    train_loader = _make_loader(
        train_bundle["X_seq"],
        train_bundle["lengths"],
        train_bundle["X_num"],
        train_bundle["y"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = SequenceStateRegressor(
        embedding_dim=train_bundle["X_seq"].shape[2],
        num_features=train_bundle["X_num"].shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()

    best_score = -1e9
    best_epoch = 0
    best_state = None
    patience_left = args.patience
    trainlog_rows: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x_seq, lengths, x_num, y in tqdm(
            train_loader, desc=f"Epoch {epoch}", unit="batch"
        ):
            x_seq = x_seq.to(device)
            lengths = lengths.to(device)
            x_num = x_num.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x_seq, lengths, x_num)
            loss = loss_fn(outputs, y)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(x_seq)
            total_count += len(x_seq)

        train_loss = total_loss / max(total_count, 1)
        y_true_val = val_anchor_bundle["meta"][
            ["state_change_valence", "state_change_arousal"]
        ].to_numpy(dtype=float)
        r_v, r_a, val_loss = _eval_anchors(
            model,
            val_anchor_bundle["X_seq"],
            val_anchor_bundle["lengths"],
            val_anchor_bundle["X_num"],
            y_true_val,
            args.batch_size,
            device,
        )
        primary_score = float(np.mean([r_v, r_a]))
        trainlog_rows.append(
            {
                "run_id": args.run_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "r_delta_valence": r_v,
                "r_delta_arousal": r_a,
                "primary_score": primary_score,
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"rΔV={r_v:.4f} rΔA={r_a:.4f} primary={primary_score:.4f}"
        )

        if primary_score > best_score:
            best_score = primary_score
            best_epoch = epoch
            best_state = {
                "model_state": model.state_dict(),
                "config": {
                    "seq_len": resolved_seq_len,
                    "k_state": args.k_state,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "num_features": train_bundle["X_num"].shape[1],
                    "embedding_dim": train_bundle["X_seq"].shape[2],
                    "ablate_no_history": bool(args.ablate_no_history),
                },
                "norm_stats": norm_stats,
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is None:
        raise SystemExit("Training failed to produce a valid checkpoint.")

    models_dir = repo_root / "models" / "subtask2a_sequence" / "runs" / args.run_id
    if models_dir.exists():
        raise FileExistsError(f"Model run directory already exists: {models_dir}")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pt"
    torch.save(best_state, model_path)
    print(f"Saved model checkpoint to: {model_path}")

    trainlogs_dir = repo_root / "reports" / "trainlogs" / "subtask2a"
    trainlogs_dir.mkdir(parents=True, exist_ok=True)
    log_path = trainlogs_dir / f"subtask2a_trainlog__{args.run_id}.csv"
    pd.DataFrame(trainlog_rows).to_csv(log_path, index=False)

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "task_tag": "subtask2a_phaseC",
            "seed": args.seed,
            "split_path": str(split_path.relative_to(repo_root)),
            "embeddings_path": str(embeddings_path.relative_to(repo_root)),
            "embeddings_sha256": sha256_file(embeddings_path),
            "seq_len": resolved_seq_len,
            "k_state": args.k_state,
            "feature_names": norm_stats.get("feature_names"),
            "norm_stats": norm_stats,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "model_tag": args.model_tag,
            "artifacts": {
                "model": artifact_ref(model_path, repo_root),
                "trainlog": artifact_ref(log_path, repo_root),
            },
            "config": {
                "toggles": {
                    "subtask2a": {
                        "seq_len": resolved_seq_len,
                        "ablate_no_history": bool(args.ablate_no_history),
                    }
                }
            },
            "counts": {
                "n_train_steps": int(len(train_bundle["X_seq"])),
                "n_val_anchors": int(len(anchors_df)),
                "n_train_users": int(df_raw.iloc[train_idx]["user_id"].nunique()),
                "n_val_users": int(df_raw.iloc[val_idx]["user_id"].nunique()),
            },
            "metrics": {
                "train_selection": {
                    "best_epoch": best_epoch,
                    "best_primary_score": best_score,
                }
            },
        },
    )

    print(
        "Summary: "
        f"train_steps={len(train_bundle['X_seq'])}, "
        f"val_anchors={len(anchors_df)}, "
        f"best_epoch={best_epoch}, best_primary={best_score:.4f}"
    )


if __name__ == "__main__":
    main()

