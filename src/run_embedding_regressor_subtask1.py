from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loader import load_all_data


def print_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute and print MAE and MSE for a 2D target (valence, arousal).
    Returns a dict with these metrics.
    """
    mae_valence = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_arousal = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mse_valence = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_arousal = mean_squared_error(y_true[:, 1], y_pred[:, 1])

    print(f"[{prefix}] MAE: valence={mae_valence:.4f}, arousal={mae_arousal:.4f}")
    print(f"[{prefix}] MSE: valence={mse_valence:.4f}, arousal={mse_arousal:.4f}")

    return {
        "MAE_valence": mae_valence,
        "MAE_arousal": mae_arousal,
        "MSE_valence": mse_valence,
        "MSE_arousal": mse_arousal,
    }


def load_subtask1_embeddings_and_labels(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Subtask 1 embeddings and align them with valence/arousal labels.

    Returns:
        X: embeddings array of shape (N, D)
        y: labels array of shape (N, 2) with columns [valence, arousal]
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    user_ids = data["user_id"]
    text_ids = data["text_id"]

    bundle = load_all_data()
    subtask1 = bundle["subtask1"].copy()
    labels_df = subtask1[["user_id", "text_id", "valence", "arousal"]].copy()

    emb_index_df = pd.DataFrame({"user_id": user_ids, "text_id": text_ids})
    merged = emb_index_df.merge(
        labels_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )

    if merged.shape[0] != embeddings.shape[0]:
        raise RuntimeError(
            f"Mismatch after alignment: merged rows={merged.shape[0]}, embeddings rows={embeddings.shape[0]}"
        )

    y = merged[["valence", "arousal"]].to_numpy()
    return embeddings, y


def train_val_split_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split embeddings and labels into train/validation sets.
    Returns X_train, X_val, y_train, y_val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_val, y_train, y_val


def run_embedding_regressor_subtask1(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train and evaluate a Ridge regression model on Subtask 1 embeddings.
    Returns a dict of metrics.
    """
    X, y = load_subtask1_embeddings_and_labels(embeddings_path=embeddings_path)
    print(f"Loaded embeddings X shape: {X.shape}, labels y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_val_split_embeddings(X, y, random_state=random_state)

    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = print_regression_metrics(
        y_val,
        y_pred,
        prefix="Subtask 1 | Embedding Ridge",
    )
    metrics["model"] = "embedding_ridge"
    return metrics


def load_baseline_results(
    path: Path | str = Path("reports/baseline_comparison.csv"),
) -> pd.DataFrame:
    """
    Load baseline comparison table produced in Phase 6.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline comparison file not found: {path}")
    df = pd.read_csv(path)
    return df


def build_embedding_comparison_table(
    baseline_df: pd.DataFrame,
    embedding_metrics: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame containing:
        - TF-IDF ridge baseline row (model == 'tfidf_ridge')
        - embedding-based ridge row
    """
    tfidf_row = baseline_df[baseline_df["model"] == "tfidf_ridge"].copy()
    if tfidf_row.empty:
        raise ValueError("No TF-IDF ridge baseline found in baseline comparison table.")

    embedding_row = pd.DataFrame(
        [
            {
                "model": embedding_metrics.get("model", "embedding_ridge"),
                "MAE_valence": embedding_metrics["MAE_valence"],
                "MAE_arousal": embedding_metrics["MAE_arousal"],
                "MSE_valence": embedding_metrics["MSE_valence"],
                "MSE_arousal": embedding_metrics["MSE_arousal"],
            }
        ]
    )

    comparison_df = pd.concat([tfidf_row, embedding_row], ignore_index=True)
    cols = ["model"] + [c for c in comparison_df.columns if c != "model"]
    comparison_df = comparison_df[cols]
    return comparison_df


def save_embedding_comparison(
    comparison_df: pd.DataFrame,
    path: Path | str = Path("reports/embedding_regressor_subtask1.csv"),
) -> Path:
    """
    Save the embedding vs TF-IDF comparison table to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(path, index=False)
    print(f"\nSaved embedding comparison to: {path}")
    return path


def main() -> None:
    """
    Entry point for running Subtask 1 embedding-based regressor and comparison.
    """
    embedding_metrics = run_embedding_regressor_subtask1()
    baseline_df = load_baseline_results()
    comparison_df = build_embedding_comparison_table(baseline_df, embedding_metrics)

    print("\n=== Embedding vs TF-IDF comparison (Subtask 1) ===")
    print(comparison_df)

    save_embedding_comparison(comparison_df)


if __name__ == "__main__":
    main()

