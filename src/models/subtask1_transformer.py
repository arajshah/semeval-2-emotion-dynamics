from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError("Could not resolve repo root from __file__ path.")


def set_seed(seed: int) -> None:
    from src.utils.seed_utils import set_seed as _set_seed

    _set_seed(seed)


@dataclass
class Subtask1TransformerConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    dropout: float = 0.1
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    grad_accum_steps: int = 1
    epochs: int = 3
    seed: int = 42
    amp: str = "auto"
    grad_checkpointing: bool = False
    num_workers: int = 0
    head_type: str = "simple"
    output_dir: str = "models/subtask1_transformer"

    def to_json(self) -> Dict[str, object]:
        cfg = asdict(self)
        if cfg["amp"] == "auto":
            cfg["amp"] = "fp16" if torch.cuda.is_available() else "off"
        return cfg


def get_text_column(df: pd.DataFrame) -> str:
    preferred = ["text", "entry", "entry_text", "essay", "content"]
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a text column. Available columns: {list(df.columns)}"
    )


class Subtask1Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = get_text_column(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            row[self.text_col],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [float(row["valence"]), float(row["arousal"])], dtype=torch.float
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


class Subtask1RegressorConfig(PretrainedConfig):
    model_type = "subtask1_regressor"

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",  
        dropout: float = 0.1,
        head_type: str = "simple",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.dropout = dropout
        self.head_type = head_type


class Subtask1Regressor(PreTrainedModel):
    config_class = Subtask1RegressorConfig

    def __init__(self, config: Subtask1RegressorConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.head_type = config.head_type

        def build_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 2),
            )

        if self.head_type == "simple":
            self.head = build_mlp()
        elif self.head_type == "level_dev":
            self.level_head = build_mlp()
            self.dev_head = build_mlp()
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        if self.head_type == "simple":
            return self.head(pooled)
        level = self.level_head(pooled)
        dev = self.dev_head(pooled)
        return level + dev


def clip_preds(preds_np: np.ndarray) -> np.ndarray:
    clipped = preds_np.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], -2, 2)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, 2)
    return clipped


def save_hf_checkpoint(
    model: Subtask1Regressor, 
    tokenizer, 
    out_dir: str | Path, 
    run_cfg: Subtask1TransformerConfig | None = None,) -> Path:

    out_dir = Path(out_dir)
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if run_cfg is not None:
        with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_cfg.to_json(), f, indent=2, sort_keys=True)
    return out_dir


def load_hf_checkpoint(
    out_dir: str | Path, model_name_fallback: str | None = None
) -> tuple[Subtask1Regressor, object]:
    out_dir = Path(out_dir)

    # 1) Try the clean path: load the full saved model + config directly.
    #    IMPORTANT: force-disable meta-device / device_map behavior.
    try:
        model = Subtask1Regressor.from_pretrained(
            out_dir,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    except Exception:
        # 2) Fallback path: DO NOT use AutoConfig here (it can't resolve model_type=subtask1_regressor).
        try:
            cfg = Subtask1RegressorConfig.from_pretrained(out_dir)
            model_name = cfg.model_name
            dropout = getattr(cfg, "dropout", 0.1)
            head_type = getattr(cfg, "head_type", "simple")
        except Exception:
            # Last-resort fallback if config load fails
            model_name = model_name_fallback or str(out_dir)
            dropout = 0.1
            head_type = "simple"

        model_cfg = Subtask1RegressorConfig(
            model_name=model_name, dropout=dropout, head_type=head_type
        )
        model = Subtask1Regressor(model_cfg)

        # Load weights (support both safetensors + pytorch bin)
        safetensors_path = out_dir / "model.safetensors"
        bin_path = out_dir / "pytorch_model.bin"

        if safetensors_path.exists():
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No weights found in {out_dir} (expected model.safetensors or pytorch_model.bin)"
            )

        model.load_state_dict(state_dict, strict=True)

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    return model, tokenizer

