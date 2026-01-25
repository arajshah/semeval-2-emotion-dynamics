from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn


class Subtask2BUserMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int, int] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h3, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(
    run_dir: Path,
    model: Subtask2BUserMLP,
    config: Dict,
    norm_stats: Dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2), encoding="utf-8"
    )


def load_checkpoint(run_dir: Path) -> Tuple[Subtask2BUserMLP, Dict, Dict]:
    config_path = run_dir / "config.json"
    norm_path = run_dir / "norm_stats.json"
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm stats: {norm_path}")

    config = json.loads(config_path.read_text())
    norm_stats = json.loads(norm_path.read_text())
    model = Subtask2BUserMLP(
        input_dim=int(config["input_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config, norm_stats
