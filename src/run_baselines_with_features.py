from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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
    Returns a dict with the metrics for further aggregation.
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


def train_val_split_subtask1(
    df: pd.DataFrame,
    target_cols: List[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split Subtask 1 DataFrame into train/validation sets and return:
        train_df, val_df, y_train, y_val
    """
    if target_cols is None:
        target_cols = ["valence", "arousal"]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )

    y_train = train_df[target_cols].to_numpy()
    y_val = val_df[target_cols].to_numpy()

    return train_df, val_df, y_train, y_val


def run_subtask1_baselines_with_features(
    processed_path: Path | str = Path("data/processed/subtask1_basic_features.parquet"),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run Subtask 1 baselines:
        - global mean
        - TF-IDF-only ridge regression
        - TF-IDF + basic numeric features ridge regression

    Returns a DataFrame summarizing the metrics for each model.
    """
    processed_path = Path(processed_path)
    df = pd.read_parquet(processed_path)

    train_df, val_df, y_train, y_val = train_val_split_subtask1(df, random_state=random_state)

    results: List[Dict[str, float | str]] = []

    # Model 1: Global mean
    mean_valence = y_train[:, 0].mean()
    mean_arousal = y_train[:, 1].mean()
    y_pred_mean = np.column_stack(
        [
            np.full_like(y_val[:, 0], fill_value=mean_valence, dtype=float),
            np.full_like(y_val[:, 1], fill_value=mean_arousal, dtype=float),
        ]
    )
    metrics = print_regression_metrics(y_val, y_pred_mean, prefix="Subtask 1 | Global mean")
    metrics["model"] = "global_mean"
    results.append(metrics)

    # Model 2: TF-IDF-only Ridge
    vectorizer_tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
    )
    X_train_text = vectorizer_tfidf.fit_transform(train_df["text"].astype(str))
    X_val_text = vectorizer_tfidf.transform(val_df["text"].astype(str))

    model_tfidf = Ridge(alpha=1.0, random_state=random_state)
    model_tfidf.fit(X_train_text, y_train)
    y_pred_tfidf = model_tfidf.predict(X_val_text)

    metrics = print_regression_metrics(
        y_val, y_pred_tfidf, prefix="Subtask 1 | TF-IDF + Ridge"
    )
    metrics["model"] = "tfidf_ridge"
    results.append(metrics)

    # Model 3: TF-IDF + numeric features Ridge
    candidate_num_cols = [
        "text_len_tokens",
        "text_len_chars",
        "text_sent_punct_count",
        "is_words_int",
        "text_len_tokens_z",
        "text_len_chars_z",
    ]
    num_cols = [c for c in candidate_num_cols if c in train_df.columns]

    X_train_num = csr_matrix(train_df[num_cols].to_numpy(dtype=float))
    X_val_num = csr_matrix(val_df[num_cols].to_numpy(dtype=float))

    vectorizer_tfidf_feats = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
    )
    X_train_text_feats = vectorizer_tfidf_feats.fit_transform(train_df["text"].astype(str))
    X_val_text_feats = vectorizer_tfidf_feats.transform(val_df["text"].astype(str))

    X_train_combined = hstack([X_train_text_feats, X_train_num])
    X_val_combined = hstack([X_val_text_feats, X_val_num])

    model_combined = Ridge(alpha=1.0, random_state=random_state)
    model_combined.fit(X_train_combined, y_train)
    y_pred_combined = model_combined.predict(X_val_combined)

    metrics = print_regression_metrics(
        y_val, y_pred_combined, prefix="Subtask 1 | TF-IDF + numeric features + Ridge"
    )
    metrics["model"] = "tfidf_numeric_ridge"
    results.append(metrics)

    results_df = pd.DataFrame(results)
    cols = ["model"] + [c for c in results_df.columns if c != "model"]
    results_df = results_df[cols]

    print("\n=== Baseline comparison (Subtask 1) ===")
    print(results_df)

    return results_df


def _ensure_reports_dir() -> Path:
    """
    Ensure that the reports/ directory exists and return its Path.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def save_comparison_table(results_df: pd.DataFrame) -> Path:
    """
    Save the baseline comparison table to reports/baseline_comparison.csv.
    Returns the path to the saved file.
    """
    reports_dir = _ensure_reports_dir()
    out_path = reports_dir / "baseline_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved baseline comparison to: {out_path}")
    return out_path


def main() -> None:
    """
    Entry point for running Subtask 1 baselines with basic features.
    """
    # Optionally verify that data loads, but primary input is the processed Subtask 1 file.
    try:
        load_all_data()
    except Exception:
        pass

    results_df = run_subtask1_baselines_with_features()
    save_comparison_table(results_df)


if __name__ == "__main__":
    main()

