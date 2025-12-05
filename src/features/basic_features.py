from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_loader import load_all_data


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def add_basic_text_features(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Return a copy of `df` with simple numeric text features added.

    Added columns:
        - text_len_tokens: number of whitespace-separated tokens
        - text_len_chars: number of characters
        - text_sent_punct_count: count of '.', '!', or '?' characters
        - is_words_int: integer version of is_words (1 if True, 0 if False, NaN if missing)
        - text_len_tokens_z: z-score of token length within this DataFrame
        - text_len_chars_z: z-score of char length within this DataFrame
    """
    out = df.copy()

    if text_col not in out.columns:
        raise ValueError(f"Expected text column '{text_col}' not found in DataFrame")

    out["text_len_tokens"] = (
        out[text_col]
        .astype(str)
        .str.split()
        .str.len()
        .astype("float")
    )

    out["text_len_chars"] = out[text_col].astype(str).str.len().astype("float")

    out["text_sent_punct_count"] = (
        out[text_col]
        .astype(str)
        .str.count(r"[\.!\?]")
        .astype("float")
    )

    if "is_words" in out.columns:
        out["is_words_int"] = out["is_words"].astype("boolean").astype("Int8")
    else:
        out["is_words_int"] = pd.Series(pd.NA, index=out.index, dtype="Int8")

    out["text_len_tokens_z"] = _zscore(out["text_len_tokens"])
    out["text_len_chars_z"] = _zscore(out["text_len_chars"])

    return out


def add_basic_features_subtask1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper to add basic features for Subtask 1 data.
    Currently just calls `add_basic_text_features` and returns the result.
    """
    return add_basic_text_features(df, text_col="text")


def _get_processed_dir() -> Path:
    """
    Ensure data/processed exists under the project root and return its Path.
    """
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def build_and_save_subtask1_basic_features() -> Path:
    """
    Load Subtask 1 data, add basic features, and save to data/processed/.

    Returns the path to the saved file.
    """
    data = load_all_data()
    subtask1 = data["subtask1"]

    subtask1_feat = add_basic_features_subtask1(subtask1)

    processed_dir = _get_processed_dir()
    out_path = processed_dir / "subtask1_basic_features.parquet"
    subtask1_feat.to_parquet(out_path, index=False)

    print(f"Saved Subtask 1 basic features to: {out_path}")
    return out_path


def main() -> None:
    """
    Entry point for building and saving basic features.
    """
    build_and_save_subtask1_basic_features()


if __name__ == "__main__":
    main()

