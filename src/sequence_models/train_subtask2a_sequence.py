from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import argparse
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_loader import load_all_data
from src.eval.splits import load_frozen_split
from src.sequence_models.subtask2a_sequence_dataset import (
    build_subtask2a_datasets,
    build_subtask2a_val_anchored_users_from_split,
    collate_sequence_batch,
)
from src.sequence_models.simple_sequence_model import SimpleSequenceRegressor
from src.utils.provenance import merge_run_metadata, sha256_file, write_run_metadata
from src.utils.run_id import validate_run_id


def compute_delta_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Δ-MAE, Δ-MSE, and direction accuracy for valence and arousal.
    """
    mae_val = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_ar = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    mse_val = np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
    mse_ar = np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)

    def direction_acc(y_t: np.ndarray, y_p: np.ndarray) -> float:
        sign_true = np.sign(y_t)
        sign_pred = np.sign(y_p)
        return float((sign_true == sign_pred).mean())

    dir_val = direction_acc(y_true[:, 0], y_pred[:, 0])
    dir_ar = direction_acc(y_true[:, 1], y_pred[:, 1])

    return {
        "Delta_MAE_valence": mae_val,
        "Delta_MAE_arousal": mae_ar,
        "Delta_MSE_valence": mse_val,
        "Delta_MSE_arousal": mse_ar,
        "DirAcc_valence": dir_val,
        "DirAcc_arousal": dir_ar,
    }


def train_one_epoch(
    model: SimpleSequenceRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Training", unit="batch"):
        inputs = batch["inputs"].to(device)
        targets = batch["target"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        preds = model(inputs, lengths)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate(
    model: SimpleSequenceRegressor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    ys: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            inputs = batch["inputs"].to(device)
            targets = batch["target"].to(device)
            lengths = batch["lengths"].to(device)

            outputs = model(inputs, lengths)
            loss = loss_fn(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            ys.append(targets.cpu().numpy())
            preds_list.append(outputs.cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds_list, axis=0)

    metrics = compute_delta_metrics(y_true, y_pred)
    metrics["val_loss"] = avg_loss
    return metrics


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Subtask 2A sequence model.")
    parser.add_argument("--mode", choices=["train", "predict_anchors"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_path", default="models/subtask2a_sequence/model.pt")
    parser.add_argument("--quick_limit_users", type=int, default=None)
    parser.add_argument("--model_tag", default="minilm_seq_anchor_v1", required=True)
    args = parser.parse_args()

    validate_run_id(args.run_id)
    generator = _set_determinism(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.mode == "predict_anchors":
        anchors_df, sequences, lengths, embedding_dim_arr = build_subtask2a_val_anchored_users_from_split(
            seed=args.seed,
            regime="unseen_user",
            seq_len=args.seq_len,
            embeddings_path=embeddings_path,
            quick_limit_users=args.quick_limit_users,
        )
        if len(anchors_df) == 0:
            raise SystemExit("No eligible anchored users found for prediction.")

        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        if not model_path.exists():
            raise SystemExit(f"Model checkpoint not found: {model_path}")

        embedding_dim = int(embedding_dim_arr[0]) if len(embedding_dim_arr) else sequences.shape[-1]
        model = SimpleSequenceRegressor(
            embedding_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), args.batch_size):
                end = start + args.batch_size
                batch_seq = torch.from_numpy(sequences[start:end]).to(device)
                batch_lengths = torch.from_numpy(lengths[start:end]).to(device)
                outputs = model(batch_seq, batch_lengths)
                preds_list.append(outputs.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        out_df = anchors_df.copy()
        out_df.insert(0, "run_id", args.run_id)
        out_df.insert(1, "seed", int(args.seed))
        out_df["delta_valence_pred"] = preds[:, 0]
        out_df["delta_arousal_pred"] = preds[:, 1]

        if out_df["user_id"].duplicated().any():
            raise SystemExit("Anchored preds must have exactly one row per user.")
        required_cols = [
            "anchor_idx",
            "anchor_text_id",
            "anchor_timestamp",
            "delta_valence_true",
            "delta_arousal_true",
            "delta_valence_pred",
            "delta_arousal_pred",
        ]
        if out_df[required_cols].isna().any().any():
            raise SystemExit("Anchored preds contain missing required values.")
        if not np.isfinite(out_df[["delta_valence_pred", "delta_arousal_pred"]].to_numpy()).all():
            raise SystemExit("Anchored preds contain non-finite prediction values.")

        preds_dir = repo_root / "reports" / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        out_path = preds_dir / f"subtask2a_val_user_preds__{args.run_id}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"Wrote anchored preds to: {out_path}")
        return

    train_dataset, val_dataset, embedding_dim = build_subtask2a_datasets(
        embeddings_path=embeddings_path,
        seq_len=args.seq_len,
        split_mode="frozen_indices",
        split_path=split_path,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequence_batch,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
        num_workers=0,
        generator=generator,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Embedding dim: {embedding_dim}, Sequence length: {args.seq_len}")

    model = SimpleSequenceRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_per_epoch: List[Dict[str, float]] = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        val_loss = val_metrics["val_loss"]
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"ΔMAE(V)={val_metrics['Delta_MAE_valence']:.4f}, "
            f"ΔMAE(A)={val_metrics['Delta_MAE_arousal']:.4f}"
        )

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        metrics_per_epoch.append(row)

    models_dir = repo_root / "models" / "subtask2a_sequence" / "runs" / args.run_id
    if models_dir.exists():
        raise FileExistsError(f"Model run directory already exists: {models_dir}")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model checkpoint to: {model_path}")
    legacy_path = repo_root / "models" / "subtask2a_sequence" / "model.pt"
    if not legacy_path.exists():
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, legacy_path)
        print(f"Saved legacy checkpoint to: {legacy_path}")
    else:
        print(f"Legacy checkpoint exists; not overwriting: {legacy_path}")

    trainlogs_dir = repo_root / "reports" / "trainlogs" / "subtask2a"
    trainlogs_dir.mkdir(parents=True, exist_ok=True)
    log_path = trainlogs_dir / f"subtask2a_trainlog__{args.run_id}.csv"
    results_df = pd.DataFrame(metrics_per_epoch)
    results_df.to_csv(log_path, index=False)

    data = load_all_data()
    df_raw = data["subtask2a"].copy().reset_index(drop=True)
    train_idx, val_idx = load_frozen_split(split_path, df_raw)
    train_df_raw = df_raw.iloc[train_idx]
    val_df_raw = df_raw.iloc[val_idx]

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "task_tag": "subtask2a_sequence",
            "seed": args.seed,
            "split_path": str(split_path.relative_to(repo_root)),
            "embeddings_path": str(embeddings_path.relative_to(repo_root)),
            "embeddings_sha256": sha256_file(embeddings_path),
            "embeddings_source": embeddings_path.name,
            "df_raw_len": int(len(df_raw)),
            "train_idx_len": int(len(train_idx)),
            "val_idx_len": int(len(val_idx)),
            "n_train_users": int(train_df_raw["user_id"].nunique()),
            "n_val_users": int(val_df_raw["user_id"].nunique()),
            "n_train_step_samples": int(len(train_dataset)),
            "n_val_step_samples": int(len(val_dataset)),
            "checkpoint_dir": str(model_path.relative_to(repo_root)),
            "trainlog_path": str(log_path.relative_to(repo_root)),
        },
    )

    write_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        task="subtask2a",
        task_tag="subtask2a_sequence",
        seed=args.seed,
        regime="unseen_user",
        split_path=str(split_path.relative_to(repo_root)),
        config={
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "dropout": args.dropout,
        },
        artifacts={
            "checkpoint_dir": str(model_path.relative_to(repo_root)),
            "trainlog_path": str(log_path.relative_to(repo_root)),
        },
    )

    print(
        "Summary: "
        f"train_users={int(train_df_raw['user_id'].nunique())}, "
        f"val_users={int(val_df_raw['user_id'].nunique())}, "
        f"train_steps={len(train_dataset)}, "
        f"val_steps={len(val_dataset)}"
    )


if __name__ == "__main__":
    main()

