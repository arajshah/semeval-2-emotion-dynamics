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
from src.sequence_models.simple_sequence_model import SimpleSequenceRegressor


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
    metrics["val_loss"] = avg_loss
    return metrics


def main() -> None:
    """
    Train a simple sequence model on Subtask 2A to predict ΔV/ΔA.
    """
    seq_len = 5
    batch_size = 32
    hidden_dim = 128
    num_layers = 1
    num_epochs = 3
    lr = 1e-3
    random_state = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    model = SimpleSequenceRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_per_epoch: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        val_loss = val_metrics["val_loss"]

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"ΔMAE(V)={val_metrics['Delta_MAE_valence']:.4f}, "
            f"ΔMAE(A)={val_metrics['Delta_MAE_arousal']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        metrics_per_epoch.append(row)

    models_dir = Path("models") / "subtask2a_sequence"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model checkpoint to: {model_path}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "subtask2a_sequence_results.csv"
    results_df = pd.DataFrame(metrics_per_epoch)
    results_df.to_csv(results_path, index=False)
    print(f"Saved metrics to: {results_path}")


if __name__ == "__main__":
    main()

