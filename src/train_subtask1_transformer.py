from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import load_all_data
from src.eval.analysis_tools import compute_regression_metrics, make_seen_user_time_split
from src.models.subtask1_transformer import (
    Subtask1Dataset,
    Subtask1Regressor,
    Subtask1TransformerConfig,
    clip_preds,
    get_repo_root,
    set_seed,
    load_hf_checkpoint,
    save_hf_checkpoint,
)

from tqdm.auto import tqdm


def _resolve_amp_mode(amp: str) -> str:
    if amp == "auto":
        return "fp16" if torch.cuda.is_available() else "off"
    return amp


def _load_subtask1_df() -> pd.DataFrame:
    data = load_all_data()
    df = data["subtask1"].copy().reset_index(drop=True)
    df = df.dropna(subset=["valence", "arousal"])
    return df


def _save_unseen_user_split(
    df: pd.DataFrame, seed: int, split_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        id_type = payload.get("id_type", "row_index")
        train_ids = payload["train_ids"]
        val_ids = payload["val_ids"]
        if id_type == "text_id" and "text_id" in df.columns:
            train_idx = df.index[df["text_id"].isin(set(train_ids))].to_numpy()
            val_idx = df.index[df["text_id"].isin(set(val_ids))].to_numpy()
        else:
            train_idx = np.array(train_ids, dtype=int)
            val_idx = np.array(val_ids, dtype=int)
        return train_idx, val_idx

    groups = df["user_id"].to_numpy()
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    if "text_id" in df.columns:
        train_ids = df.loc[train_idx, "text_id"].tolist()
        val_ids = df.loc[val_idx, "text_id"].tolist()
        payload = {
            "id_type": "text_id",
            "train_ids": train_ids,
            "val_ids": val_ids,
            "seed": seed,
        }
    else:
        payload = {
            "id_type": "row_index",
            "train_ids": train_idx.tolist(),
            "val_ids": val_idx.tolist(),
            "seed": seed,
        }

    split_path.write_text(json.dumps(payload, indent=2))
    return train_idx, val_idx


def _build_loaders(
    df: pd.DataFrame,
    tokenizer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = Subtask1Dataset(df.iloc[train_idx].copy(), tokenizer, max_length)
    val_ds = Subtask1Dataset(df.iloc[val_idx].copy(), tokenizer, max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def _evaluate(
    model: Subtask1Regressor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(outputs.cpu().numpy())
            targets.append(labels)

    y_pred = clip_preds(np.concatenate(preds, axis=0))
    y_true = np.concatenate(targets, axis=0)

    metrics_val = compute_regression_metrics(y_true[:, 0], y_pred[:, 0])
    metrics_aro = compute_regression_metrics(y_true[:, 1], y_pred[:, 1])
    score = float(
        np.nanmean([metrics_val["pearson"], metrics_aro["pearson"]])
    )
    return {
        "score": score,
        "valence_mae": metrics_val["mae"],
        "valence_mse": metrics_val["mse"],
        "valence_pearson": metrics_val["pearson"],
        "arousal_mae": metrics_aro["mae"],
        "arousal_mse": metrics_aro["mse"],
        "arousal_pearson": metrics_aro["pearson"],
        "n_val": int(len(y_true)),
    }


def _append_trainlog(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Subtask 1 transformer regressor.")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime", choices=["unseen_user", "seen_user"], default="unseen_user")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16", "auto"], default="auto")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu_smoke", action="store_true")
    args = parser.parse_args()

    if args.cpu_smoke:
        args.max_length = 64
        args.batch_size = 2
        args.epochs = 1
        args.num_workers = 0
        args.amp = "off"
        args.grad_checkpointing = False
        subset_size = 200
    elif args.quick:
        subset_size = 500
        args.epochs = min(args.epochs, 1)
    else:
        subset_size = None

    cfg = Subtask1TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        seed=args.seed,
        amp=args.amp,
        grad_checkpointing=args.grad_checkpointing,
        num_workers=args.num_workers,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode = _resolve_amp_mode(cfg.amp)

    df = _load_subtask1_df()
    if subset_size is not None:
        df = df.iloc[:subset_size].copy()

    repo_root = get_repo_root()
    reports_dir = repo_root / "reports"
    split_dir = reports_dir / "splits"
    split_path = split_dir / f"subtask1_unseen_user_seed{cfg.seed}.json"

    if args.regime == "unseen_user":
        train_idx, val_idx = _save_unseen_user_split(df, cfg.seed, split_path)
    else:
        train_idx, val_idx = make_seen_user_time_split(df, val_frac=0.2)

    if args.resume_from:
        model, tokenizer = load_hf_checkpoint(args.resume_from, model_name_fallback=cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = Subtask1Regressor(cfg.model_name, dropout=cfg.dropout)

    if cfg.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    model.to(device)

    train_loader, val_loader = _build_loaders(
        df, tokenizer, train_idx, val_idx, cfg.max_length, cfg.batch_size, cfg.num_workers
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = int(np.ceil(len(train_loader) / cfg.grad_accum_steps)) * cfg.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_mode == "fp16")
    best_score = -np.inf
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"train e{epoch}/{cfg.epochs}",
            leave=False,
        )

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if amp_mode == "fp16":
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = torch.nn.SmoothL1Loss()(outputs, labels)
                scaler.scale(loss).backward()
            elif amp_mode == "bf16":
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = torch.nn.SmoothL1Loss()(outputs, labels)
                loss.backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.SmoothL1Loss()(outputs, labels)
                loss.backward()

            if step % cfg.grad_accum_steps == 0:
                if amp_mode == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        metrics = _evaluate(model, val_loader, device)
        trainlog_row = {
            "epoch": epoch,
            "score": metrics["score"],
            "valence_mae": metrics["valence_mae"],
            "valence_mse": metrics["valence_mse"],
            "valence_pearson": metrics["valence_pearson"],
            "arousal_mae": metrics["arousal_mae"],
            "arousal_mse": metrics["arousal_mse"],
            "arousal_pearson": metrics["arousal_pearson"],
            "n_val": metrics["n_val"],
            "seed": cfg.seed,
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
        }
        _append_trainlog(reports_dir / "subtask1_transformer_trainlog.csv", trainlog_row)

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_epoch = epoch
            save_hf_checkpoint(model, tokenizer, repo_root / cfg.output_dir / "best")

    metrics_path = repo_root / cfg.output_dir / "metrics.json"
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "config": cfg.to_json(),
        "regime": args.regime,
        "seed": cfg.seed,
        "timestamp": datetime.utcnow().isoformat(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
