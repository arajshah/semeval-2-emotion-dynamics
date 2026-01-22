from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.data_loader import load_all_data
from src.eval.analysis_tools import (
    compute_subtask1_metrics_from_preds,
    evaluate_subtask1,
    iter_slices,
    load_frozen_split,
    make_seen_user_time_split,
    make_unseen_user_splits,
)
from src.models.subtask1_transformer import get_repo_root


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute per-fold MAE and MSE for valence and arousal.

    y_true, y_pred: shape (n_samples, 2) in order [valence, arousal].
    """
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == 2

    diff = y_pred - y_true
    mae_val = np.mean(np.abs(diff[:, 0]))
    mae_ar = np.mean(np.abs(diff[:, 1]))
    mse_val = np.mean(diff[:, 0] ** 2)
    mse_ar = np.mean(diff[:, 1] ** 2)

    return {
        "MAE_valence": float(mae_val),
        "MAE_arousal": float(mae_ar),
        "MSE_valence": float(mse_val),
        "MSE_arousal": float(mse_ar),
    }


def _aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of per-fold metric dicts into mean/std."""
    if not fold_metrics:
        raise ValueError("No fold metrics to aggregate.")

    keys = fold_metrics[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std(ddof=1))
    return aggregated


def _prepare_subtask1_dataframe() -> pd.DataFrame:
    """Load Subtask 1 data and add simple numeric features."""
    data = load_all_data()
    df = data["subtask1"].copy().reset_index(drop=True)

    df = df.dropna(subset=["text", "valence", "arousal"]).reset_index(drop=True)

    df["text_length_chars"] = df["text"].str.len().astype("float32")
    df["text_length_tokens"] = df["text"].str.split().apply(len).astype("float32")
    df["is_words_int"] = df["is_words"].astype(int).astype("float32")

    return df


def _load_embeddings_for_subtask1(df: pd.DataFrame):
    """
    Load precomputed embeddings and align them with df by (user_id, text_id).

    Expects:
      data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz
    with arrays: embeddings, user_id, text_id.
    """
    npz_path = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Expected embeddings file not found: {npz_path}. "
            "Run `python -m src.embeddings.extract_embeddings` first."
        )

    npz = np.load(npz_path)
    embeddings = npz["embeddings"]
    emb_user = npz["user_id"]
    emb_text = npz["text_id"]

    emb_df = pd.DataFrame(
        {
            "user_id": emb_user,
            "text_id": emb_text,
            "emb_index": np.arange(len(emb_user)),
        }
    )

    merged = df.merge(
        emb_df, on=["user_id", "text_id"], how="inner", validate="one_to_one"
    ).reset_index(drop=True)

    emb_matrix = embeddings[merged["emb_index"].to_numpy()]
    return merged, emb_matrix


def _run_cv_global_mean(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """
    Cross-validated metrics for a trivial global-mean predictor.

    The prediction is the global mean of valence/arousal from the full dataset.
    """
    y = df[["valence", "arousal"]].to_numpy(dtype=float)
    global_mean = y.mean(axis=0, keepdims=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for _, val_idx in kf.split(y):
        y_val = y[val_idx]
        y_pred = np.repeat(global_mean, repeats=len(val_idx), axis=0)
        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _run_cv_tfidf_ridge(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """Cross-validated TF-IDF + Ridge model."""
    texts = df["text"].to_numpy()
    y = df[["valence", "arousal"]].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(texts):
        X_train_text = texts[train_idx]
        X_val_text = texts[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _fit_tfidf_ridge(
    train_text: np.ndarray, train_targets: np.ndarray
) -> Tuple[TfidfVectorizer, Ridge]:
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_text)
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_targets)
    return vectorizer, model


def _predict_tfidf_ridge(
    vectorizer: TfidfVectorizer, model: Ridge, texts: np.ndarray
) -> np.ndarray:
    X_val = vectorizer.transform(texts)
    return model.predict(X_val)


def _aggregate_unseen_user_metrics(
    fold_rows: List[Dict[str, float]]
) -> Dict[str, float]:
    if not fold_rows:
        raise ValueError("No fold rows provided for aggregation.")

    base = fold_rows[0]
    agg: Dict[str, float] = {
        "regime": base["regime"],
        "slice": base["slice"],
        "n": int(sum(row["n"] for row in fold_rows)),
    }

    metric_keys = [
        key
        for key in base.keys()
        if key not in {"regime", "slice", "n", "model"}
    ]
    for key in metric_keys:
        values = np.array([row[key] for row in fold_rows], dtype=float)
        agg[key] = float(np.nanmean(values)) if not np.isnan(values).all() else np.nan

    return agg


def run_phase0_subtask1_eval(df: pd.DataFrame, model_name: str = "tfidf_ridge") -> pd.DataFrame:
    """
    Run Phase 0 evaluation for Subtask 1 using a lightweight model.

    For unseen-user folds, n is the sum of validation rows across folds.
    """
    texts = df["text"].to_numpy()
    y = df[["valence", "arousal"]].to_numpy(dtype=float)

    unseen_splits = make_unseen_user_splits(df, n_splits=5, seed=42)
    seen_train_idx, seen_val_idx = make_seen_user_time_split(df, val_frac=0.2)

    rows: List[Dict[str, float]] = []

    # Unseen-user evaluation: aggregate metrics across folds.
    slice_fold_rows: Dict[str, List[Dict[str, float]]] = {}
    for train_idx, val_idx in unseen_splits:
        vectorizer, model = _fit_tfidf_ridge(texts[train_idx], y[train_idx])
        y_pred = _predict_tfidf_ridge(vectorizer, model, texts[val_idx])

        pred_valence = np.full(len(df), np.nan, dtype=float)
        pred_arousal = np.full(len(df), np.nan, dtype=float)
        pred_valence[val_idx] = y_pred[:, 0]
        pred_arousal[val_idx] = y_pred[:, 1]

        for slice_name, slice_idx in iter_slices(df):
            eval_idx = np.intersect1d(val_idx, slice_idx)
            metrics = evaluate_subtask1(
                df, pred_valence, pred_arousal, "unseen_user", slice_name, eval_idx
            )
            metrics["model"] = model_name
            slice_fold_rows.setdefault(slice_name, []).append(metrics)

    for slice_name, fold_rows in slice_fold_rows.items():
        agg_row = _aggregate_unseen_user_metrics(fold_rows)
        agg_row["model"] = model_name
        rows.append(agg_row)

    # Seen-user evaluation: single time-aware split.
    vectorizer, model = _fit_tfidf_ridge(texts[seen_train_idx], y[seen_train_idx])
    y_pred = _predict_tfidf_ridge(vectorizer, model, texts[seen_val_idx])

    pred_valence = np.full(len(df), np.nan, dtype=float)
    pred_arousal = np.full(len(df), np.nan, dtype=float)
    pred_valence[seen_val_idx] = y_pred[:, 0]
    pred_arousal[seen_val_idx] = y_pred[:, 1]

    for slice_name, slice_idx in iter_slices(df):
        eval_idx = np.intersect1d(seen_val_idx, slice_idx)
        metrics = evaluate_subtask1(
            df, pred_valence, pred_arousal, "seen_user", slice_name, eval_idx
        )
        metrics["model"] = model_name
        rows.append(metrics)

    df_report = pd.DataFrame(rows)
    cols_order = [
        "model",
        "regime",
        "slice",
        "n",
        "valence_mae",
        "valence_mse",
        "valence_pearson",
        "arousal_mae",
        "arousal_mse",
        "arousal_pearson",
    ]
    df_report = df_report[cols_order]
    return df_report


def _run_cv_tfidf_plus_features_ridge(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """Cross-validated TF-IDF + simple numeric features + Ridge model."""
    texts = df["text"].to_numpy()
    y = df[["valence", "arousal"]].to_numpy(dtype=float)
    X_num = df[["text_length_chars", "text_length_tokens", "is_words_int"]].to_numpy(
        dtype=float
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(texts):
        X_train_text = texts[train_idx]
        X_val_text = texts[val_idx]
        X_train_num = X_num[train_idx]
        X_val_num = X_num[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_val_tfidf = vectorizer.transform(X_val_text)

        X_train = sparse.hstack(
            [X_train_tfidf, sparse.csr_matrix(X_train_num)], format="csr"
        )
        X_val = sparse.hstack(
            [X_val_tfidf, sparse.csr_matrix(X_val_num)], format="csr"
        )

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _run_cv_embeddings_ridge(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """Cross-validated Ridge model on precomputed embeddings."""
    aligned_df, X_emb = _load_embeddings_for_subtask1(df)
    y = aligned_df[["valence", "arousal"]].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(X_emb):
        X_train = X_emb[train_idx]
        X_val = X_emb[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def _run_cv_embeddings_random_forest(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
    """Cross-validated RandomForestRegressor on precomputed embeddings."""
    aligned_df, X_emb = _load_embeddings_for_subtask1(df)
    y = aligned_df[["valence", "arousal"]].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in kf.split(X_emb):
        X_train = X_emb[train_idx]
        X_val = X_emb[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_fold_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    return _aggregate_metrics(fold_metrics)


def run_subtask1_model_comparison(n_splits: int = 5) -> pd.DataFrame:
    """
    Run k-fold cross-validation for several Subtask 1 models and
    return a unified metrics DataFrame.
    """
    df = _prepare_subtask1_dataframe()

    results: List[Dict[str, float]] = []

    gm_metrics = _run_cv_global_mean(df, n_splits=n_splits)
    gm_metrics["model_name"] = "global_mean"
    results.append(gm_metrics)

    tfidf_metrics = _run_cv_tfidf_ridge(df, n_splits=n_splits)
    tfidf_metrics["model_name"] = "tfidf_ridge"
    results.append(tfidf_metrics)

    tfidf_feat_metrics = _run_cv_tfidf_plus_features_ridge(df, n_splits=n_splits)
    tfidf_feat_metrics["model_name"] = "tfidf_plus_features_ridge"
    results.append(tfidf_feat_metrics)

    emb_ridge_metrics = _run_cv_embeddings_ridge(df, n_splits=n_splits)
    emb_ridge_metrics["model_name"] = "embeddings_ridge"
    results.append(emb_ridge_metrics)

    emb_rf_metrics = _run_cv_embeddings_random_forest(df, n_splits=n_splits)
    emb_rf_metrics["model_name"] = "embeddings_random_forest"
    results.append(emb_rf_metrics)

    df_results = pd.DataFrame(results)
    cols_order = [
        "model_name",
        "MAE_valence_mean",
        "MAE_valence_std",
        "MAE_arousal_mean",
        "MAE_arousal_std",
        "MSE_valence_mean",
        "MSE_valence_std",
        "MSE_arousal_mean",
        "MSE_arousal_std",
    ]
    cols_order = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[cols_order]

    return df_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Subtask 1 model comparison.")
    parser.add_argument("--include_transformer", action="store_true")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--pred_path", default=None)
    parser.add_argument("--pred_dir", default="reports/preds")
    parser.add_argument(
        "--split_path",
        default="reports/splits/subtask1_unseen_user_seed42.json",
    )

    args = parser.parse_args()

    df = _prepare_subtask1_dataframe()
    phase0_report = run_phase0_subtask1_eval(df)

    reports_dir = get_repo_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    phase0_path = reports_dir / "subtask1_phase0_eval.csv"

    if args.include_transformer:
        split_path = Path(args.split_path)
        if not split_path.is_absolute():
            split_path = get_repo_root() / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Frozen split JSON not found at {split_path}")

        pred_path = None
        if args.pred_path:
            pred_path = Path(args.pred_path)
        elif args.run_id:
            pred_path = Path(args.pred_dir) / f"subtask1_val_preds__{args.run_id}.parquet"

        if pred_path is None:
            print(
                "Transformer included but no --run_id/--pred_path provided; skipping transformer."
            )
        else:
            if not pred_path.is_absolute():
                pred_path = get_repo_root() / pred_path
            if not pred_path.exists():
                print(f"Transformer preds not found at {pred_path}; skipping transformer.")
            else:
                _, val_idx = load_frozen_split(split_path, df)
                preds_df = pd.read_parquet(pred_path)
                metrics_rows = compute_subtask1_metrics_from_preds(df, preds_df, val_idx)
                label = "subtask1_transformer"
                if args.run_id:
                    label = f"{label}[{args.run_id}]"
                for row in metrics_rows:
                    row["model"] = label
                    row["regime"] = "unseen_user"
                    row["primary_score"] = float(
                        np.mean([row["r_composite_valence"], row["r_composite_arousal"]])
                    )
                transformer_report = pd.DataFrame(metrics_rows)
                phase0_report = pd.concat([phase0_report, transformer_report], ignore_index=True)

    df_results = run_subtask1_model_comparison(n_splits=5)

    comparison_rows = []
    for _, r in df_results.iterrows():
        comparison_rows.append(
            {
                "model": r["model_name"],
                "regime": "cv",
                "slice": "all",
                "n": np.nan,
                "valence_mae": r.get("MAE_valence_mean", np.nan),
                "valence_mse": r.get("MSE_valence_mean", np.nan),
                "valence_pearson": np.nan,
                "arousal_mae": r.get("MAE_arousal_mean", np.nan),
                "arousal_mse": r.get("MSE_arousal_mean", np.nan),
                "arousal_pearson": np.nan,
            }
        )
    phase0_report = pd.concat([phase0_report, pd.DataFrame(comparison_rows)], ignore_index=True)

    phase0_report.to_csv(phase0_path, index=False)
    print(f"Wrote {phase0_path}")

    best_valence = phase0_report.loc[phase0_report["valence_mae"].idxmin()]
    best_arousal = phase0_report.loc[phase0_report["arousal_mae"].idxmin()]
    print("Best valence MAE row:")
    print(best_valence.to_string())
    print("Best arousal MAE row:")
    print(best_arousal.to_string())


if __name__ == "__main__":
    main()

