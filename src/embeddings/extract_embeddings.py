from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_data


def _get_processed_dir() -> Path:
    """
    Ensure data/processed exists under the project root and return its Path.
    """
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def load_subtask1_for_embeddings() -> pd.DataFrame:
    """
    Load Subtask 1 data for embedding extraction.

    Prefer the feature-augmented parquet file if available:
        data/processed/subtask1_basic_features.parquet
    Otherwise, fall back to raw Subtask 1 from load_all_data().
    """
    processed_path = Path("data/processed/subtask1_basic_features.parquet")

    if processed_path.exists():
        df = pd.read_parquet(processed_path)
        print(f"Loaded Subtask 1 basic features from: {processed_path}")
    else:
        data_bundle = load_all_data()
        df = data_bundle["subtask1"].copy()
        print("Loaded raw Subtask 1 data via load_all_data() (basic features parquet not found).")

    required_cols = {"user_id", "text_id", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Subtask 1 data: {missing}")

    return df


def load_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a sentence-transformers encoder.
    """
    print(f"Loading sentence encoder: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def compute_embeddings_for_subtask1(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dense text embeddings for Subtask 1 texts.

    Returns:
        embeddings: np.ndarray of shape (N, D)
        user_ids: np.ndarray of shape (N,)
        text_ids: np.ndarray of shape (N,)
    """
    texts = df["text"].astype(str).tolist()
    user_ids = df["user_id"].to_numpy()
    text_ids = df["text_id"].to_numpy()

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if embeddings.shape[0] != len(texts):
        raise RuntimeError("Embedding count does not match number of texts.")

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, user_ids, text_ids


def save_subtask1_embeddings(
    embeddings: np.ndarray,
    user_ids: np.ndarray,
    text_ids: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
) -> Path:
    """
    Save embeddings and identifiers to a compressed .npz file under data/processed/.

    The filename should encode the model name, e.g.:
        subtask1_embeddings_all-MiniLM-L6-v2.npz
    """
    processed_dir = _get_processed_dir()
    safe_model_name = model_name.replace("/", "-")
    out_path = processed_dir / f"subtask1_embeddings_{safe_model_name}.npz"

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        user_id=user_ids,
        text_id=text_ids,
    )
    print(f"Saved embeddings to: {out_path}")
    return out_path


def main() -> None:
    """
    Entry point for extracting and saving Subtask 1 text embeddings.
    """
    df = load_subtask1_for_embeddings()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = load_sentence_encoder(model_name=model_name)

    embeddings, user_ids, text_ids = compute_embeddings_for_subtask1(df, model)

    save_subtask1_embeddings(
        embeddings=embeddings,
        user_ids=user_ids,
        text_ids=text_ids,
        model_name="all-MiniLM-L6-v2",
    )


if __name__ == "__main__":
    main()

