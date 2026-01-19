from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.subtask1_transformer import (
    Subtask1Dataset,
    clip_preds,
    get_repo_root,
    load_hf_checkpoint,
    set_seed,
)


def predict_subtask1_df(
    df: pd.DataFrame,
    checkpoint_dir: str | Path = "models/subtask1_transformer/best",
    batch_size: int = 16,
    max_length: int | None = None,
    seed: int = 42,
    cache_path: str | Path = "reports/subtask1_transformer_preds.parquet",
    use_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    repo_root = get_repo_root()
    checkpoint_dir = repo_root / checkpoint_dir
    cache_path = repo_root / cache_path

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if (
            "row_index" in cached.columns
            and len(cached) == len(df)
            and np.array_equal(cached["row_index"].to_numpy(), df.index.to_numpy())
        ):
            return cached["pred_valence"].to_numpy(), cached["pred_arousal"].to_numpy()

    set_seed(seed)
    model, tokenizer = load_hf_checkpoint(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    effective_max_length = 256 if max_length is None else int(max_length)
    dataset = Subtask1Dataset(df.copy(), tokenizer, effective_max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(outputs.cpu().numpy())

    y_pred = clip_preds(np.concatenate(preds, axis=0))
    pred_valence = y_pred[:, 0]
    pred_arousal = y_pred[:, 1]

    model_tag = f"{checkpoint_dir.name}_maxlen{effective_max_length}_seed{seed}"
    cache_df = pd.DataFrame(
        {
            "row_index": df.index.to_numpy(),
            "pred_valence": pred_valence,
            "pred_arousal": pred_arousal,
            "model_tag": model_tag,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)

    return pred_valence, pred_arousal


if __name__ == "__main__":
    raise RuntimeError(
        "This module is intended to be imported, not run as a script."
    )
