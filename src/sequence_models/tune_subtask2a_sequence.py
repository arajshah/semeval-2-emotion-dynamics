from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sequence_models.subtask2a_sequence_dataset import (
    build_subtask2a_datasets,
    create_dataloaders,
)
from src.sequence_models.simple_sequence_model import (
    SimpleSequenceRegressor,
    TransformerSequenceRegressor,
)
from src.eval.analysis_tools import compute_delta_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
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
    model: nn.Module,
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
        for batch in loader:
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
    metrics["val_loss"] = float(avg_loss)
    return metrics


def run_tuning_subtask2a() -> pd.DataFrame:
    """
    Run a small hyperparameter sweep for Subtask 2A sequence models
    (LSTM vs Transformer) and return a metrics DataFrame.
    """
    seq_len = 5
    batch_size = 32
    random_state = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for tuning: {device}")

    train_dataset, val_dataset, embedding_dim = build_subtask2a_datasets(
        seq_len=seq_len,
        random_state=random_state,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Embedding dim: {embedding_dim}, Sequence length: {seq_len}")

    configs: List[Dict] = [
        {
            "model_name": "lstm_h128_l1",
            "arch": "lstm",
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "nhead": None,
            "lr": 1e-3,
            "num_epochs": 3,
        },
        {
            "model_name": "lstm_h256_l2",
            "arch": "lstm",
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.2,
            "nhead": None,
            "lr": 1e-3,
            "num_epochs": 3,
        },
        {
            "model_name": "transformer_h128_l2",
            "arch": "transformer",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "nhead": 4,
            "lr": 1e-3,
            "num_epochs": 3,
        },
        {
            "model_name": "transformer_h256_l2",
            "arch": "transformer",
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "nhead": 4,
            "lr": 1e-3,
            "num_epochs": 3,
        },
    ]

    all_rows: List[Dict[str, float]] = []

    for cfg in configs:
        print("\n========================================")
        print(f"Config: {cfg['model_name']} ({cfg['arch']})")
        print("========================================")

        if cfg["arch"] == "lstm":
            model = SimpleSequenceRegressor(
                embedding_dim=embedding_dim,
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            ).to(device)
        elif cfg["arch"] == "transformer":
            model = TransformerSequenceRegressor(
                embedding_dim=embedding_dim,
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                nhead=cfg["nhead"] if cfg["nhead"] is not None else 4,
                dropout=cfg["dropout"],
            ).to(device)
        else:
            raise ValueError(f"Unknown arch: {cfg['arch']}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        best_metrics: Dict[str, float] | None = None
        best_score = float("inf")

        for epoch in range(1, cfg["num_epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            score = (
                val_metrics["Delta_MAE_valence"] + val_metrics["Delta_MAE_arousal"]
            ) / 2.0

            print(
                f"Epoch {epoch}/{cfg['num_epochs']} "
                f"- train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}, "
                f"ΔMAE(V)={val_metrics['Delta_MAE_valence']:.4f}, "
                f"ΔMAE(A)={val_metrics['Delta_MAE_arousal']:.4f}"
            )

            if score < best_score:
                best_score = score
                best_metrics = val_metrics

        if best_metrics is None:
            raise RuntimeError("No best metrics recorded for config.")

        row: Dict[str, float] = {
            "model_name": cfg["model_name"],
            "arch": cfg["arch"],
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "Delta_MAE_valence": best_metrics["Delta_MAE_valence"],
            "Delta_MAE_arousal": best_metrics["Delta_MAE_arousal"],
            "Delta_MSE_valence": best_metrics["Delta_MSE_valence"],
            "Delta_MSE_arousal": best_metrics["Delta_MSE_arousal"],
            "DirAcc_valence": best_metrics["DirAcc_valence"],
            "DirAcc_arousal": best_metrics["DirAcc_arousal"],
            "val_loss": best_metrics["val_loss"],
        }
        all_rows.append(row)

    df_results = pd.DataFrame(all_rows)
    cols_order = [
        "model_name",
        "arch",
        "hidden_dim",
        "num_layers",
        "dropout",
        "lr",
        "Delta_MAE_valence",
        "Delta_MAE_arousal",
        "Delta_MSE_valence",
        "Delta_MSE_arousal",
        "DirAcc_valence",
        "DirAcc_arousal",
        "val_loss",
    ]
    cols_order = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[cols_order]

    return df_results


def main() -> None:
    df_results = run_tuning_subtask2a()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "subtask2a_sequence_model_comparison.csv"

    df_results.to_csv(out_path, index=False)
    print(f"\nSaved Subtask 2A sequence model comparison to: {out_path}")
    print(df_results)


if __name__ == "__main__":
    main()

