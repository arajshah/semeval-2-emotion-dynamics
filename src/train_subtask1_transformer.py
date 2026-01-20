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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import load_all_data
from src.eval.analysis_tools import (
    compute_subtask1_correlations,
    load_frozen_split,
    make_seen_user_time_split,
)
from src.models.subtask1_transformer import (
    Subtask1Dataset,
    Subtask1Regressor,
    Subtask1RegressorConfig,
    Subtask1TransformerConfig,
    clip_preds,
    get_repo_root,
    set_seed,
    load_hf_checkpoint,
    save_hf_checkpoint,
)
from src.utils.run_id import compute_config_hash, generate_run_id, get_git_commit

from tqdm.auto import tqdm


def _resolve_amp_mode(amp: str, device: torch.device) -> str:
    if amp in {"fp16", "bf16"} and device.type != "cuda":
        return "off"
    return amp


def _load_subtask1_df() -> pd.DataFrame:
    data = load_all_data()
    df = data["subtask1"].copy().reset_index(drop=True)
    df = df.dropna(subset=["valence", "arousal"])
    return df


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
    df_val: pd.DataFrame,
) -> Dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += float(loss.item())
            total_count += int(labels.shape[0])
            preds.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    y_pred = clip_preds(np.concatenate(preds, axis=0))
    y_true = np.concatenate(targets, axis=0)

    corr = compute_subtask1_correlations(df_val, y_true, y_pred)
    score = float(np.mean([corr["r_composite_valence"], corr["r_composite_arousal"]]))
    val_loss = total_loss / max(1, total_count)

    diff = y_pred - y_true
    mae_val = float(np.mean(np.abs(diff[:, 0])))
    mae_ar = float(np.mean(np.abs(diff[:, 1])))
    mse_val = float(np.mean(diff[:, 0] ** 2))
    mse_ar = float(np.mean(diff[:, 1] ** 2))
    return {
        "score": score,
        "val_loss": val_loss,
        "valence_mae": mae_val,
        "valence_mse": mse_val,
        "arousal_mae": mae_ar,
        "arousal_mse": mse_ar,
        **corr,
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
    default_amp = "fp16" if torch.cuda.is_available() else "off"
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default=default_amp)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--head_type", choices=["simple", "level_dev"], default="simple")
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu_smoke", action="store_true")
    args = parser.parse_args()

    if args.cpu_smoke or args.quick:
        args.max_length = 128
        args.batch_size = 2
        args.epochs = 1
        args.num_workers = 0
        args.amp = "off"
        args.grad_checkpointing = False
        subset_size = 200
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
        head_type=args.head_type,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode = _resolve_amp_mode(cfg.amp, device)

    df = _load_subtask1_df()

    repo_root = get_repo_root()
    reports_dir = repo_root / "reports"
    split_path = Path(args.split_path)
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    if not split_path.exists():
        raise FileNotFoundError(
            f"Frozen split JSON not found at {split_path}. "
            "Provide --split_path to an existing split file."
        )

    if args.regime == "unseen_user":
        train_idx, val_idx = load_frozen_split(split_path, df)
    else:
        train_idx, val_idx = make_seen_user_time_split(df, val_frac=0.2)

    if subset_size is not None:
        combined = np.unique(np.concatenate([train_idx, val_idx]))
        if len(combined) > subset_size:
            combined = combined[:subset_size]
        train_idx = np.intersect1d(train_idx, combined)
        val_idx = np.intersect1d(val_idx, combined)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError("Quick subset produced empty train or val split.")

    resume_from = args.resume_from_checkpoint or args.resume_from
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.name == "best":
            raise ValueError("Refusing to resume from 'best/' checkpoint.")
        model, tokenizer = load_hf_checkpoint(resume_path, model_name_fallback=cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model_cfg = Subtask1RegressorConfig(
            model_name=cfg.model_name, dropout=cfg.dropout, head_type=cfg.head_type
        )
        model = Subtask1Regressor(model_cfg)

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
    run_id = generate_run_id("subtask1_transformer", cfg.seed)
    config_payload = {
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "amp": amp_mode,
        "grad_checkpointing": cfg.grad_checkpointing,
        "head_type": cfg.head_type,
    }
    config_hash = compute_config_hash(config_payload)
    git_commit = get_git_commit(repo_root)

    if resume_from:
        opt_path = Path(resume_from) / "optimizer.pt"
        sched_path = Path(resume_from) / "scheduler.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_count = 0

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

            epoch_loss += float(loss.detach().cpu()) * labels.shape[0]
            epoch_count += int(labels.shape[0])
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        metrics = _evaluate(model, val_loader, device, val_df)
        train_loss = epoch_loss / max(1, epoch_count)
        trainlog_row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "score": metrics["score"],
            "r_composite_valence": metrics["r_composite_valence"],
            "r_composite_arousal": metrics["r_composite_arousal"],
            "valence_mae": metrics["valence_mae"],
            "valence_mse": metrics["valence_mse"],
            "arousal_mae": metrics["arousal_mae"],
            "arousal_mse": metrics["arousal_mse"],
            "n_val": metrics["n_val"],
            "seed": cfg.seed,
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "head_type": cfg.head_type,
            "config_hash": config_hash,
            "split_path": str(split_path),
        }
        _append_trainlog(reports_dir / "subtask1_transformer_trainlog.csv", trainlog_row)

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_epoch = epoch
            save_hf_checkpoint(model, tokenizer, repo_root / cfg.output_dir / "best")

    metrics_path = repo_root / cfg.output_dir / "metrics.json"
    metrics_payload = {
        "best_run_id": run_id,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "config": config_payload,
        "config_hash": config_hash,
        "regime": args.regime,
        "seed": cfg.seed,
        "split_path": str(split_path),
        "git_commit": git_commit,
        "timestamp": datetime.utcnow().isoformat(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
