from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loader import load_all_data
from src.eval.splits import load_frozen_split
from src.sequence_models.baselines_subtask2a import (
    add_prev_delta_features,
    fit_linear_prev,
    predict_linear_prev,
    LinearPrevModel,
)

EMA_ALPHA = 0.3
NUMERIC_FEATURE_COLS = [
    "dv_prev",
    "da_prev",
    "dv_prev_missing",
    "da_prev_missing",
    "dt_prev_seconds",
    "history_len",
    "dv_mean_prev",
    "da_mean_prev",
    "dv_ema_prev",
    "da_ema_prev",
    "base_pred_dv",
    "base_pred_da",
]

def _linear_prev_from_stats(stats: Dict) -> LinearPrevModel:
    """
    Reconstruct a LinearPrevModel from a serialized dict stored in norm_stats.
    """
    return LinearPrevModel(
        coef_valence=np.asarray(stats["coef_valence"], dtype=float),
        coef_arousal=np.asarray(stats["coef_arousal"], dtype=float),
        feature_names=tuple(stats.get("feature_names", ("intercept", "dv_prev", "da_prev"))),
        fit_rows=int(stats.get("fit_rows", 0)),
        n_users=int(stats.get("n_users", 0)),
    )


def _finalize_numeric_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce: all NUMERIC_FEATURE_COLS exist, are finite, float32, and filled with 0.0 where undefined.
    """
    out = df.copy()

    if "dv_prev" not in out.columns or "da_prev" not in out.columns:
        raise ValueError("Expected dv_prev/da_prev to exist before finalize_numeric_feature_cols().")

    out["dv_prev_missing"] = out["dv_prev"].isna()
    out["da_prev_missing"] = out["da_prev"].isna()

    # Fill required columns
    fill0 = [
        "dv_prev",
        "da_prev",
        "dt_prev_seconds",
        "history_len",
        "dv_mean_prev",
        "da_mean_prev",
        "dv_ema_prev",
        "da_ema_prev",
        "base_pred_dv",
        "base_pred_da",
    ]
    for c in fill0:
        if c not in out.columns:
            raise ValueError(f"Expected numeric feature column missing: {c}")
        out[c] = out[c].fillna(0.0)

    # Cast flags to 0/1 float
    out["dv_prev_missing"] = out["dv_prev_missing"].astype(np.float32)
    out["da_prev_missing"] = out["da_prev_missing"].astype(np.float32)

    # Cast everything in NUMERIC_FEATURE_COLS to float32
    for c in NUMERIC_FEATURE_COLS:
        out[c] = out[c].astype(np.float32)

    # Hard check: no NaNs/infs
    nf = out[NUMERIC_FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(nf).all():
        bad = np.where(~np.isfinite(nf))
        raise RuntimeError(f"Non-finite numeric features found at positions (row,col): {list(zip(bad[0][:10], bad[1][:10]))}")

    return out


def _attach_c2_features(
    df: pd.DataFrame,
    *,
    baseline_model: Optional[LinearPrevModel],
    fit_baseline_on: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, LinearPrevModel]:
    """
    Canonical C2 feature attachment:
      1) prev-delta features: dv_prev/da_prev + missing flags (based on state_change columns)
      2) history aggregates: dt_prev_seconds/history_len/running means/EMA
      3) linear(prev) base preds: base_pred_dv/base_pred_da (fit on TRAIN only; applied to any df)
      4) finalize NUMERIC_FEATURE_COLS as float32 + finite
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out = out.sort_values(["user_id", "timestamp", "text_id"], kind="mergesort")

    # 1) prev-delta features: based on *state_change_* history (not valence diffs)
    out = add_prev_delta_features(out)

    # 2) history aggregates (deterministic)
    out = _add_history_features(out)

    # 3) baseline model: fit on TRAIN ONLY if not provided
    if baseline_model is None:
        fit_df = fit_baseline_on if fit_baseline_on is not None else out
        baseline_model = fit_linear_prev(fit_df)

    # attach baseline predictions (safe: val never used for fitting if caller passes train-fit model)
    out = predict_linear_prev(
        out,
        baseline_model,
        fill_value=0.0,
        out_v_col="base_pred_dv",
        out_a_col="base_pred_da",
    )

    # 4) finalize schema
    out = _finalize_numeric_feature_cols(out)
    return out, baseline_model

def load_subtask2a_with_embeddings(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Subtask 2A data and align it with precomputed embeddings (by user_id, text_id).

    Returns:
        merged_df: Subtask 2A DataFrame with an extra 'emb_index' column
        embeddings: np.ndarray of shape (N, D)
    """
    data_bundle = load_all_data()
    subtask2a = data_bundle["subtask2a"].copy()

    # Contract: df_raw is unmodified here; eligibility filtering is applied only downstream.
    required_cols = {
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "state_change_valence",
        "state_change_arousal",
    }
    missing = required_cols - set(subtask2a.columns)
    if missing:
        raise ValueError(f"Subtask2A missing required columns: {sorted(missing)}")

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    user_ids = data["user_id"]
    text_ids = data["text_id"]

    emb_index_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids)),
        }
    )

    merged = subtask2a.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping entries between Subtask 2A and embeddings.")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")
    if merged["timestamp"].isna().any():
        raise ValueError("Subtask2A timestamp parse produced NaT values.")

    return merged, embeddings


def load_subtask2a_with_cached_embeddings(
    embeddings_path: Path | str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Subtask 2A data and align it with cached embeddings by (user_id, text_id).
    """
    data_bundle = load_all_data()
    df_raw = data_bundle["subtask2a"].copy().reset_index(drop=True)
    required_cols = {
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "state_change_valence",
        "state_change_arousal",
    }
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Subtask2A missing required columns: {sorted(missing)}")
    df_raw["idx"] = df_raw.index

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    user_ids = data["user_id"]
    text_ids = data["text_id"]

    emb_index_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids)),
        }
    )

    merged = df_raw.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping entries between Subtask 2A and embeddings.")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")
    if merged["timestamp"].isna().any():
        raise ValueError("Subtask2A timestamp parse produced NaT values.")

    return merged, embeddings


def build_subtask2a_val_anchored_users_from_split(
    seed: int,
    regime: str = "unseen_user",
    seq_len: int = 5,
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    quick_limit_users: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build anchored per-user validation rows from the frozen split indices.
    """
    df_raw = load_all_data()["subtask2a"].copy()
    if regime != "unseen_user":
        raise ValueError("Only unseen_user regime is supported for anchored val builder.")

    split_path = Path("reports") / "splits" / f"subtask2a_unseen_user_seed{seed}.json"
    train_idx, val_idx = load_frozen_split(split_path, df_raw)

    val_df_raw = df_raw.iloc[val_idx].copy()
    val_df_raw["anchor_idx"] = np.asarray(val_idx, dtype=int)
    if len(val_df_raw) != len(val_idx):
        raise RuntimeError(
            f"val_df_raw length mismatch: {len(val_df_raw)} vs val_idx {len(val_idx)}"
        )
    if val_df_raw["anchor_idx"].min() < 0 or val_df_raw["anchor_idx"].max() >= len(df_raw):
        raise RuntimeError("anchor_idx out of bounds for df_raw.")

    val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"])
    eligible = val_df_raw["state_change_valence"].notna() & val_df_raw[
        "state_change_arousal"
    ].notna()
    eligible_df = val_df_raw.loc[eligible].copy()

    anchors: List[pd.Series] = []
    for _, group in eligible_df.groupby("user_id", sort=False):
        group_sorted = group.sort_values("timestamp", kind="stable")
        anchors.append(group_sorted.iloc[-1])

    if not anchors:
        anchors_df = pd.DataFrame(
            columns=[
                "user_id",
                "anchor_idx",
                "anchor_text_id",
                "anchor_timestamp",
                "delta_valence_true",
                "delta_arousal_true",
            ]
        )
        return anchors_df, np.zeros((0, seq_len, 0), dtype=np.float32), np.array([], dtype=np.int64), np.array([0], dtype=np.int64)

    anchors_df = pd.DataFrame(anchors).reset_index(drop=True)
    if quick_limit_users is not None:
        anchors_df = anchors_df.head(int(quick_limit_users)).copy()

    merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)
    merged_by_user = {
        user_id: group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for user_id, group in merged.groupby("user_id", sort=False)
    }
    embedding_dim = embeddings.shape[1]

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    rows: List[dict] = []

    for _, anchor in anchors_df.iterrows():
        user_id = anchor["user_id"]
        text_id = anchor["text_id"]
        anchor_ts = anchor["timestamp"]
        anchor_idx = int(anchor["anchor_idx"])

        group = merged_by_user.get(user_id)
        if group is None:
            raise RuntimeError(f"Missing user in embeddings merge: {user_id}")

        history = group[group["timestamp"] <= anchor_ts].reset_index(drop=True)
        if history.empty:
            raise RuntimeError(f"No history found for user {user_id} at anchor timestamp.")

        anchor_match = history[history["text_id"] == text_id]
        if anchor_match.empty:
            raise RuntimeError(
                f"Anchor (user_id={user_id}, text_id={text_id}) missing in embeddings merge."
            )

        emb_indices = history["emb_index"].to_numpy()
        actual_len = len(emb_indices)
        start = max(0, actual_len - seq_len)
        window_indices = emb_indices[start:actual_len]

        seq = np.zeros((seq_len, embedding_dim), dtype=np.float32)
        seq[seq_len - len(window_indices) :] = embeddings[window_indices]

        sequences.append(seq)
        lengths.append(len(window_indices))
        rows.append(
            {
                "user_id": user_id,
                "anchor_idx": anchor_idx,
                "anchor_text_id": int(text_id),
                "anchor_timestamp": anchor_ts,
                "delta_valence_true": float(anchor["state_change_valence"]),
                "delta_arousal_true": float(anchor["state_change_arousal"]),
            }
        )

    anchors_df_out = pd.DataFrame(rows)
    sequences_arr = np.stack(sequences, axis=0) if sequences else np.zeros((0, seq_len, embedding_dim), dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int64)
    embedding_dim_arr = np.array([embedding_dim], dtype=np.int64)

    if anchors_df_out["user_id"].duplicated().any():
        raise RuntimeError("Expected exactly one row per user in anchored outputs.")

    return anchors_df_out, sequences_arr, lengths_arr, embedding_dim_arr


def _load_embeddings_map(
    embeddings_path: Path | str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    data = np.load(embeddings_path)
    embeddings = data["embeddings"].astype(np.float32)
    user_ids = data["user_id"]
    text_ids = data["text_id"]
    emb_index_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids)),
        }
    )
    return emb_index_df, embeddings


def select_latest_eligible_anchors(df_subset_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_subset_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    eligible = df["state_change_valence"].notna() & df["state_change_arousal"].notna()
    eligible_df = df.loc[eligible].copy()
    if eligible_df.empty:
        return eligible_df
    anchors = (
        eligible_df.sort_values(["user_id", "timestamp"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(1)
    )
    anchors = anchors.rename(columns={"idx": "anchor_idx"})
    return anchors


def select_forecast_anchors(
    df_raw: pd.DataFrame,
    marker_df: pd.DataFrame,
    cutoff_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return exactly one forecast anchor per user.

    Preferred (official) mode:
      - cutoff_df provided (e.g., test_subtask2.csv with user_id,timestamp_min)
      - anchor(u) = latest row with timestamp < timestamp_min(u)
      - deterministic tie-break by (timestamp, idx)

    Fallback mode (no cutoff_df):
      - If marker_df has is_forecasting_user: use those users; anchor = latest row overall
      - If marker_df has anchor_idx: use it directly
      - If marker_df has (user_id, text_id/anchor_text_id): first collapse marker_df to 1 row/user (latest timestamp if available),
        then merge to find anchor row.
      - If marker_df has timestamp_min-like col: use per-user cutoff from marker_df (first row per user).
    """
    df = df_raw.copy()
    df["idx"] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    if cutoff_df is not None:
        if "user_id" not in cutoff_df.columns or "timestamp_min" not in cutoff_df.columns:
            raise ValueError("cutoff_df must contain columns: user_id, timestamp_min")

        cut = cutoff_df[["user_id", "timestamp_min"]].copy()
        cut["timestamp_min"] = pd.to_datetime(cut["timestamp_min"], errors="raise")

        cut = cut.sort_values(["user_id", "timestamp_min"], kind="stable")
        cut = cut.drop_duplicates(subset=["user_id"], keep="first")

        tmp = df.merge(cut, on="user_id", how="inner", validate="many_to_one")
        tmp = tmp[tmp["timestamp"] < tmp["timestamp_min"]].copy()
        if tmp.empty:
            raise RuntimeError("No forecast anchors found (no row had timestamp < timestamp_min).")

        anchors = (
            tmp.sort_values(["user_id", "timestamp", "idx"], kind="stable")
            .groupby("user_id", sort=False)
            .tail(1)
            .rename(columns={"idx": "anchor_idx"})
        )

        missing_users = set(cut["user_id"]) - set(anchors["user_id"])
        if missing_users:
            sample = list(missing_users)[:10]
            raise RuntimeError(
                f"Missing forecast anchors for {len(missing_users)} users (no timestamp < timestamp_min). "
                f"Sample: {sample}"
            )

        out = anchors[["user_id", "anchor_idx", "text_id", "timestamp"]].reset_index(drop=True)
        if out["user_id"].duplicated().any():
            raise RuntimeError("Forecast anchors must contain exactly one row per user.")
        return out

    if "anchor_idx" in marker_df.columns:
        anchor_idx = marker_df["anchor_idx"].astype(int).to_numpy()
        anchors = df.loc[anchor_idx].copy()
        anchors = anchors.rename(columns={"idx": "anchor_idx"})
        out = anchors[["user_id", "anchor_idx", "text_id", "timestamp"]].reset_index(drop=True)
        if out["user_id"].duplicated().any():
            raise RuntimeError("Forecast anchors must contain exactly one row per user.")
        return out

    if "is_forecasting_user" in marker_df.columns:
        mu = marker_df[marker_df["is_forecasting_user"] == True]
        forecast_users = mu["user_id"].dropna().unique()
        if len(forecast_users) == 0:
            raise RuntimeError("marker_df has is_forecasting_user, but no True rows were found.")
        sub = df[df["user_id"].isin(forecast_users)].copy()
        if sub.empty:
            raise RuntimeError("No df_raw rows found for forecast users derived from marker_df.")
        anchors = (
            sub.sort_values(["user_id", "timestamp", "idx"], kind="stable")
            .groupby("user_id", sort=False)
            .tail(1)
            .rename(columns={"idx": "anchor_idx"})
        )
        out = anchors[["user_id", "anchor_idx", "text_id", "timestamp"]].reset_index(drop=True)
        if out["user_id"].duplicated().any():
            raise RuntimeError("Forecast anchors must contain exactly one row per user.")
        return out

    anchor_text_col = None
    for col in ["anchor_text_id", "text_id"]:
        if col in marker_df.columns:
            anchor_text_col = col
            break

    if anchor_text_col is not None:
        m = marker_df.copy()
        if "timestamp" in m.columns:
            m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
            m = m.sort_values(["user_id", "timestamp"], kind="stable").groupby("user_id", sort=False).tail(1)
        else:
            m = m.groupby("user_id", sort=False).head(1)

        merged = m.merge(
            df,
            left_on=["user_id", anchor_text_col],
            right_on=["user_id", "text_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["idx"].isna().any():
            missing = merged[merged["idx"].isna()]["user_id"].tolist()[:10]
            raise RuntimeError(f"Missing forecast anchor rows for users (sample): {missing}")
        anchors = merged.rename(columns={"idx": "anchor_idx"})
        out = anchors[["user_id", "anchor_idx", "text_id", "timestamp"]].reset_index(drop=True)
        if out["user_id"].duplicated().any():
            raise RuntimeError("Forecast anchors must contain exactly one row per user.")
        return out

    ts_col = None
    for col in ["timestamp_min", "timestamp_cutoff", "cutoff_timestamp", "forecast_timestamp_min"]:
        if col in marker_df.columns:
            ts_col = col
            break
    if ts_col is None:
        raise ValueError("Marker file must include one of: anchor_idx, is_forecasting_user, text_id, or timestamp_min.")

    m = marker_df.copy()
    m[ts_col] = pd.to_datetime(m[ts_col], errors="raise")

    anchors = []
    for user_id, row in m.groupby("user_id", sort=False).first().iterrows():
        cutoff = row[ts_col]
        user_rows = df[(df["user_id"] == user_id) & (df["timestamp"] < cutoff)]
        if user_rows.empty:
            raise RuntimeError(f"No forecast anchor found for user {user_id}")
        anchor_row = user_rows.sort_values(["timestamp", "idx"], kind="stable").iloc[-1]
        anchors.append(anchor_row)

    out = pd.DataFrame(anchors).rename(columns={"idx": "anchor_idx"})
    out = out[["user_id", "anchor_idx", "text_id", "timestamp"]].reset_index(drop=True)
    if out["user_id"].duplicated().any():
        raise RuntimeError("Forecast anchors must contain exactly one row per user.")
    return out


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - dt_prev_seconds
      - history_len (count of prior eligible labeled rows per user)
      - dv_mean_prev/da_mean_prev (running mean of prior deltas)
      - dv_ema_prev/da_ema_prev (EMA of prior deltas; alpha=EMA_ALPHA)
    Requires df to have:
      - user_id, timestamp, text_id
      - state_change_valence, state_change_arousal
    Returns a copy with new columns.
    """
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    working = working.sort_values(
        ["user_id", "timestamp", "text_id"], kind="mergesort"
    )

    eligible = working["state_change_valence"].notna() & working[
        "state_change_arousal"
    ].notna()
    working["_eligible"] = eligible

    working["dt_prev_seconds"] = (
        working.groupby("user_id", sort=False)["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    def _compute_group(group: pd.DataFrame) -> pd.DataFrame:
        elig = group["_eligible"]
        dv = group["state_change_valence"].astype(float)
        da = group["state_change_arousal"].astype(float)

        count_prev = elig.cumsum().shift(1).fillna(0)
        dv_cum = dv.where(elig, 0.0).cumsum().shift(1).fillna(0.0)
        da_cum = da.where(elig, 0.0).cumsum().shift(1).fillna(0.0)
        count_safe = count_prev.replace(0, 1)

        group["history_len"] = count_prev.astype(int)
        group["dv_mean_prev"] = np.where(
            count_prev.to_numpy() > 0, (dv_cum / count_safe).to_numpy(), 0.0
        )
        group["da_mean_prev"] = np.where(
            count_prev.to_numpy() > 0, (da_cum / count_safe).to_numpy(), 0.0
        )

        dv_series = dv.where(elig)
        da_series = da.where(elig)

        group["dv_ema_prev"] = (
            dv_series.ewm(alpha=EMA_ALPHA, adjust=False, ignore_na=True)
            .mean().shift(1).fillna(0.0)
        )
        group["da_ema_prev"] = (
            da_series.ewm(alpha=EMA_ALPHA, adjust=False, ignore_na=True)
            .mean().shift(1).fillna(0.0)
        )
        return group

    working = working.groupby("user_id", sort=False, group_keys=False).apply(
        _compute_group
    )
    working = working.drop(columns=["_eligible"])
    return working.sort_index()


def _fit_and_attach_linear_prev(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, LinearPrevModel]:
    """
    Fit linear(prev) on train_df ONLY, then attach baseline preds to both train and val.
    Adds columns base_pred_dv, base_pred_da (float).
    """
    model = fit_linear_prev(train_df)
    train_out = predict_linear_prev(
        train_df,
        model,
        fill_value=0.0,
        out_v_col="base_pred_dv",
        out_a_col="base_pred_da",
    )
    val_out = predict_linear_prev(
        val_df,
        model,
        fill_value=0.0,
        out_v_col="base_pred_dv",
        out_a_col="base_pred_da",
    )
    return train_out, val_out, model


def _build_sequence_and_features(
    user_df: pd.DataFrame,
    embeddings: np.ndarray,
    seq_len: int,
    target_idx_set: Optional[Set[int]],
    require_eligible: bool,
    *,
    include_numeric_features: bool,
) -> Tuple[
    List[np.ndarray],
    List[int],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[dict],
]:
    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    x_num_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    base_pred_list: List[np.ndarray] = []
    meta_rows: List[dict] = []

    idx_vals = user_df["idx"].to_numpy(dtype=int)
    emb_idx_vals = user_df["emb_index"].to_numpy(dtype=int)
    text_ids = user_df["text_id"].to_numpy()
    timestamps = user_df["timestamp"].to_numpy()

    delta_v = user_df["state_change_valence"].to_numpy(dtype=float)
    delta_a = user_df["state_change_arousal"].to_numpy(dtype=float)

    for i in range(len(user_df)):
        idx = int(idx_vals[i])
        if target_idx_set is not None and idx not in target_idx_set:
            continue

        eligible = not (np.isnan(delta_v[i]) or np.isnan(delta_a[i]))
        if require_eligible and not eligible:
            continue

        # sequence window up to i
        hist_emb_idx = emb_idx_vals[: i + 1]
        start = max(0, len(hist_emb_idx) - seq_len)
        window_idx = hist_emb_idx[start:]

        seq = np.zeros((seq_len, embeddings.shape[1]), dtype=np.float32)
        seq[seq_len - len(window_idx) :] = embeddings[window_idx]

        sequences.append(seq)
        lengths.append(len(window_idx))

        # target: always true delta (residualization happens outside if desired)
        y_list.append(np.array([delta_v[i], delta_a[i]], dtype=np.float32))

        # baseline preds must come from train-fit linear(prev), already attached as columns
        base_pred = np.array(
            [float(user_df.iloc[i]["base_pred_dv"]), float(user_df.iloc[i]["base_pred_da"])],
            dtype=np.float32,
        )
        base_pred_list.append(base_pred)

        # numeric features: canonical C2 schema
        if include_numeric_features:
            feats = user_df.iloc[i][NUMERIC_FEATURE_COLS].to_numpy(dtype=np.float32)
            x_num_list.append(feats)

        meta_rows.append(
            {
                "idx": idx,
                "user_id": user_df["user_id"].iloc[i],
                "text_id": text_ids[i],
                "timestamp": timestamps[i],
                "state_change_valence": float(delta_v[i]) if not np.isnan(delta_v[i]) else np.nan,
                "state_change_arousal": float(delta_a[i]) if not np.isnan(delta_a[i]) else np.nan,
            }
        )

    return sequences, lengths, x_num_list, y_list, base_pred_list, meta_rows


def build_subtask2a_step_dataset(
    df_raw: pd.DataFrame,
    split_idx: np.ndarray,
    embeddings_path: Path | str,
    seq_len: int,
    k_state: int,  # kept for signature compatibility; unused in C2 numeric feature path
    fit_norm: bool,
    norm_stats: dict | None = None,
    ablate_no_history: bool = False,
    use_residual_targets: bool = False,
) -> dict:
    df_subset = df_raw.iloc[split_idx].copy()
    df_subset["idx"] = np.asarray(split_idx, dtype=int)
    df_subset["row_pos"] = np.arange(len(df_subset))

    emb_index_df, embeddings = _load_embeddings_map(embeddings_path)
    merged = df_subset.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_subset):
        missing = set(zip(df_subset["user_id"], df_subset["text_id"])) - set(
            zip(merged["user_id"], merged["text_id"])
        )
        sample = list(missing)[:10]
        raise RuntimeError(
            f"Embeddings coverage mismatch: {len(df_subset)} raw rows -> {len(merged)} after merge. "
            f"Missing keys (sample): {sample}"
        )

    merged = merged.sort_values("row_pos", kind="stable").drop(columns=["row_pos"])
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")

    # Attach C2 features + linear(prev) baseline preds
    baseline_model = None
    if not fit_norm:
        if norm_stats is None or "linear_prev" not in norm_stats:
            raise ValueError("norm_stats with key 'linear_prev' required when fit_norm=False.")
        baseline_model = _linear_prev_from_stats(norm_stats["linear_prev"])

    merged, baseline_model = _attach_c2_features(
        merged,
        baseline_model=baseline_model,
        fit_baseline_on=merged if fit_norm else None,
    )

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    x_num_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    base_pred_list: List[np.ndarray] = []
    meta_rows: List[dict] = []

    for _, group in merged.groupby("user_id", sort=False):
        group_sorted = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        seqs, lens, nums, ys, base_preds, metas = _build_sequence_and_features(
            group_sorted,
            embeddings,
            seq_len,
            target_idx_set=None,
            require_eligible=True,
            include_numeric_features=True,
        )
        sequences.extend(seqs)
        lengths.extend(lens)
        x_num_list.extend(nums)
        y_list.extend(ys)
        base_pred_list.extend(base_preds)
        meta_rows.extend(metas)

    X_seq = np.stack(sequences, axis=0).astype(np.float32) if sequences else np.zeros((0, seq_len, embeddings.shape[1]), dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int64)

    n_feats = len(NUMERIC_FEATURE_COLS)
    X_num = np.stack(x_num_list, axis=0).astype(np.float32) if x_num_list else np.zeros((0, n_feats), dtype=np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32) if y_list else np.zeros((0, 2), dtype=np.float32)
    base_pred = np.stack(base_pred_list, axis=0).astype(np.float32) if base_pred_list else np.zeros((0, 2), dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    # Normalization stats: computed on TRAIN only (fit_norm=True)
    if fit_norm:
        mean = X_num.mean(axis=0) if len(X_num) else np.zeros((n_feats,), dtype=np.float32)
        std = X_num.std(axis=0) if len(X_num) else np.ones((n_feats,), dtype=np.float32)
        std = np.where(std == 0, 1.0, std)

        norm_stats = {
            "feature_names": list(NUMERIC_FEATURE_COLS),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "linear_prev": {
                "coef_valence": baseline_model.coef_valence.tolist(),
                "coef_arousal": baseline_model.coef_arousal.tolist(),
                "feature_names": list(baseline_model.feature_names),
                "fit_rows": int(getattr(baseline_model, "fit_rows", 0)),
                "n_users": int(getattr(baseline_model, "n_users", 0)),
            },
        }

    if norm_stats is None:
        raise ValueError("norm_stats required when fit_norm is False.")

    mean = np.asarray(norm_stats["mean"], dtype=np.float32)
    std = np.asarray(norm_stats["std"], dtype=np.float32)
    if mean.shape[0] != X_num.shape[1] or std.shape[0] != X_num.shape[1]:
        raise ValueError(
            f"norm_stats mean/std dim mismatch: mean={mean.shape}, std={std.shape}, X_num={X_num.shape}"
        )

    X_num = (X_num - mean) / std

    if ablate_no_history:
        X_num[:] = 0.0

    y_resid = (y - base_pred) if use_residual_targets else None

    return {
        "X_seq": X_seq,
        "lengths": lengths_arr,
        "X_num": X_num,
        "y": y,
        "y_resid": y_resid,
        "base_pred": base_pred,
        "meta": meta,
        "norm_stats": norm_stats,
    }


def build_subtask2a_anchor_features(
    df_raw: pd.DataFrame,
    anchors_df: pd.DataFrame,
    embeddings_path: Path | str,
    seq_len: int,
    k_state: int,  # kept for signature compatibility; unused in C2 numeric feature path
    norm_stats: dict,
    ablate_no_history: bool = False,
    include_residual_targets: bool = False,
) -> dict:
    n_feats = len(NUMERIC_FEATURE_COLS)

    if anchors_df.empty:
        return {
            "X_seq": np.zeros((0, seq_len, 0), dtype=np.float32),
            "lengths": np.array([], dtype=np.int64),
            "X_num": np.zeros((0, n_feats), dtype=np.float32),
            "base_pred": np.zeros((0, 2), dtype=np.float32),
            "meta": pd.DataFrame(),
        }

    if "linear_prev" not in norm_stats:
        raise ValueError("norm_stats must include 'linear_prev' to build anchored features without leakage.")

    baseline_model = _linear_prev_from_stats(norm_stats["linear_prev"])

    df_raw = df_raw.copy()
    df_raw["idx"] = df_raw.index
    target_idx_set = set(anchors_df["anchor_idx"].astype(int).tolist())

    emb_index_df, embeddings = _load_embeddings_map(embeddings_path)
    merged = df_raw.merge(
        emb_index_df,
        on=["user_id", "text_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_raw):
        missing = set(zip(df_raw["user_id"], df_raw["text_id"])) - set(
            zip(merged["user_id"], merged["text_id"])
        )
        sample = list(missing)[:10]
        raise RuntimeError(
            f"Embeddings coverage mismatch: {len(df_raw)} raw rows -> {len(merged)} after merge. "
            f"Missing keys (sample): {sample}"
        )
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")

    # Attach C2 features using TRAIN-fit baseline model from norm_stats
    merged, _ = _attach_c2_features(
        merged,
        baseline_model=baseline_model,
        fit_baseline_on=None,
    )

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    x_num_list: List[np.ndarray] = []
    meta_rows: List[dict] = []
    base_pred_list: List[np.ndarray] = []

    for _, group in merged.groupby("user_id", sort=False):
        group_sorted = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        seqs, lens, nums, ys, base_preds, metas = _build_sequence_and_features(
            group_sorted,
            embeddings,
            seq_len,
            target_idx_set=target_idx_set,
            require_eligible=False,
            include_numeric_features=True,
        )
        sequences.extend(seqs)
        lengths.extend(lens)
        x_num_list.extend(nums)
        base_pred_list.extend(base_preds)
        meta_rows.extend(metas)

    X_seq = np.stack(sequences, axis=0).astype(np.float32) if sequences else np.zeros((0, seq_len, embeddings.shape[1]), dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int64)

    X_num = np.stack(x_num_list, axis=0).astype(np.float32) if x_num_list else np.zeros((0, n_feats), dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    base_pred = np.stack(base_pred_list, axis=0).astype(np.float32) if base_pred_list else np.zeros((0, 2), dtype=np.float32)

    # normalize with train stats
    mean = np.asarray(norm_stats["mean"], dtype=np.float32)
    std = np.asarray(norm_stats["std"], dtype=np.float32)
    if mean.shape[0] != X_num.shape[1] or std.shape[0] != X_num.shape[1]:
        raise ValueError(
            f"norm_stats mean/std dim mismatch: mean={mean.shape}, std={std.shape}, X_num={X_num.shape}"
        )
    X_num = (X_num - mean) / std

    if ablate_no_history:
        X_num[:] = 0.0

    # enforce: one row per anchor
    if len(meta) != len(anchors_df):
        missing = set(anchors_df["anchor_idx"].astype(int).tolist()) - set(
            meta["idx"].astype(int).tolist()
        )
        sample = list(missing)[:10]
        raise RuntimeError(
            f"Anchor feature build mismatch: expected {len(anchors_df)} anchors, got {len(meta)}. "
            f"Missing anchor_idx sample: {sample}"
        )

    out = {
        "X_seq": X_seq,
        "lengths": lengths_arr,
        "X_num": X_num,
        "base_pred": base_pred,
        "meta": meta,
    }

    if include_residual_targets:
        y_true = meta[["state_change_valence", "state_change_arousal"]].to_numpy(dtype=np.float32)
        out["y_resid"] = y_true - base_pred

    return out


class Subtask2ASequenceDataset(Dataset):
    """
    Dataset of fixed-length embedding sequences with ΔV/ΔA labels for Subtask 2A.
    This dataset is step-level; per-user anchored evaluation is handled elsewhere and must not use this Dataset.

    Each item:
        - inputs: Tensor of shape (seq_len, embedding_dim)
        - target: Tensor of shape (2,) with [ΔV, ΔA]
        - length: actual number of valid timesteps in the sequence (<= seq_len)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        seq_len: int = 5,
        *,
        include_numeric_features: bool = False,
        use_residual_targets: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.embedding_dim = embeddings.shape[1]
        self.include_numeric_features = include_numeric_features
        self.use_residual_targets = use_residual_targets

        if self.include_numeric_features or self.use_residual_targets:
            missing = set(NUMERIC_FEATURE_COLS) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing numeric feature columns: {sorted(missing)}"
                )

        df_sorted = df.sort_values(
            ["user_id", "timestamp", "text_id"], kind="mergesort"
        ).reset_index(drop=True)

        self.samples: List[Tuple[np.ndarray, ...]] = []
        sanity_checked = False

        for _, group in df_sorted.groupby("user_id"):
            group = group.reset_index(drop=True)
            emb_indices = group["emb_index"].to_numpy()
            dval = group["state_change_valence"].to_numpy()
            dar = group["state_change_arousal"].to_numpy()

            for idx in range(len(group)):
                if np.isnan(dval[idx]) or np.isnan(dar[idx]):
                    continue

                start = max(0, idx - (seq_len - 1))
                end = idx + 1
                window_indices = emb_indices[start:end]
                actual_len = len(window_indices)

                seq = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)
                seq[self.seq_len - actual_len :] = embeddings[window_indices]

                target_raw = np.array([dval[idx], dar[idx]], dtype=np.float32)

                if self.include_numeric_features or self.use_residual_targets:
                    row = group.iloc[idx]
                    base_pred = np.array(
                        [row["base_pred_dv"], row["base_pred_da"]],
                        dtype=np.float32,
                    )
                    target_used = (
                        target_raw - base_pred
                        if self.use_residual_targets
                        else target_raw
                    )
                    numeric_features = None
                    if self.include_numeric_features:
                        numeric_features = row[NUMERIC_FEATURE_COLS].to_numpy(
                            dtype=np.float32
                        )
                    if not sanity_checked:
                        if self.include_numeric_features and numeric_features is not None:
                            assert len(numeric_features) == len(NUMERIC_FEATURE_COLS)
                            assert np.isfinite(numeric_features).all()
                        assert np.isfinite(target_raw).all()
                        sanity_checked = True
                    self.samples.append(
                        (
                            seq,
                            target_used,
                            target_raw,
                            actual_len,
                            numeric_features,
                            base_pred,
                        )
                    )
                else:
                    self.samples.append((seq, target_raw, actual_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if not (self.include_numeric_features or self.use_residual_targets):
            seq, target, length = self.samples[idx]
            return {
                "inputs": torch.from_numpy(seq),
                "target": torch.from_numpy(target),
                "length": length,
            }

        seq, target, target_raw, length, numeric_features, base_pred = self.samples[idx]
        out = {
            "inputs": torch.from_numpy(seq),
            "target": torch.from_numpy(target),
            "target_raw": torch.from_numpy(target_raw),
            "base_pred": torch.from_numpy(base_pred),
            "length": length,
        }
        if self.include_numeric_features and numeric_features is not None:
            out["numeric_features"] = torch.from_numpy(numeric_features)
        return out


def build_subtask2a_datasets(
    embeddings_path: Path | str = Path("data/processed/subtask1_embeddings_all-MiniLM-L6-v2.npz"),
    seq_len: int = 5,
    random_state: int = 42,
    split_mode: str = "random",
    split_path: Path | str | None = None,
    seed: int = 42,
    include_numeric_features: bool = False,
    use_residual_targets: bool = False,
) -> Tuple[Subtask2ASequenceDataset, Subtask2ASequenceDataset, int]:
    """
    Build train/validation datasets for Subtask 2A, splitting by user_id.

    Returns:
        train_dataset, val_dataset, embedding_dim
    """
    if split_mode == "frozen_indices":
        if split_path is None:
            raise ValueError("split_path is required for split_mode='frozen_indices'.")
        df_raw = load_all_data()["subtask2a"].copy().reset_index(drop=True)
        if len(df_raw) == 0:
            raise ValueError("Subtask2A df_raw is empty.")
        train_idx, val_idx = load_frozen_split(Path(split_path), df_raw)
        if np.min(train_idx) < 0 or np.max(train_idx) >= len(df_raw):
            raise ValueError("train_idx out of bounds for df_raw.")
        if np.min(val_idx) < 0 or np.max(val_idx) >= len(df_raw):
            raise ValueError("val_idx out of bounds for df_raw.")

        embeddings_path = Path(embeddings_path)
        data = np.load(embeddings_path)
        embeddings = data["embeddings"]
        user_ids = data["user_id"]
        text_ids = data["text_id"]

        emb_index_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "text_id": text_ids,
                "emb_index": np.arange(len(user_ids)),
            }
        )

        train_df_raw = df_raw.iloc[train_idx].copy()
        val_df_raw = df_raw.iloc[val_idx].copy()

        train_df = train_df_raw.merge(
            emb_index_df,
            on=["user_id", "text_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(train_df) != len(train_df_raw):
            expected = set(zip(train_df_raw["user_id"], train_df_raw["text_id"]))
            kept = set(zip(train_df["user_id"], train_df["text_id"]))
            missing_keys = list(expected - kept)[:20]
            raise RuntimeError(
                f"Embeddings coverage mismatch for train split: "
                f"{len(train_df_raw)} raw rows -> {len(train_df)} after inner-merge on (user_id,text_id). "
                f"Missing keys (sample): {missing_keys}"
            )
        val_df = val_df_raw.merge(
            emb_index_df,
            on=["user_id", "text_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(val_df) != len(val_df_raw):
            expected = set(zip(val_df_raw["user_id"], val_df_raw["text_id"]))
            kept = set(zip(val_df["user_id"], val_df["text_id"]))
            missing_keys = list(expected - kept)[:20]
            raise RuntimeError(
                f"Embeddings coverage mismatch for val split: "
                f"{len(val_df_raw)} raw rows -> {len(val_df)} after inner-merge on (user_id,text_id). "
                f"Missing keys (sample): {missing_keys}"
            )

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="raise")
        if train_df["timestamp"].isna().any():
            raise ValueError("Subtask2A timestamp parse produced NaT values.")
        val_df["timestamp"] = pd.to_datetime(val_df["timestamp"], errors="raise")
        if val_df["timestamp"].isna().any():
            raise ValueError("Subtask2A timestamp parse produced NaT values.")
    else:
        merged, embeddings = load_subtask2a_with_embeddings(embeddings_path)
        users = merged["user_id"].unique()
        rng = np.random.RandomState(random_state)
        rng.shuffle(users)

        split_idx = int(len(users) * 0.8)
        train_users = set(users[:split_idx])
        val_users = set(users[split_idx:])

        train_df = merged[merged["user_id"].isin(train_users)].copy()
        val_df = merged[merged["user_id"].isin(val_users)].copy()

    if include_numeric_features or use_residual_targets:
        # Canonical C2 attachment: fit baseline on TRAIN only, then apply to VAL
        train_df, baseline_model = _attach_c2_features(
            train_df,
            baseline_model=None,
            fit_baseline_on=train_df,
        )
        val_df, _ = _attach_c2_features(
            val_df,
            baseline_model=baseline_model,
            fit_baseline_on=None,
        )

    train_dataset = Subtask2ASequenceDataset(
        train_df,
        embeddings,
        seq_len=seq_len,
        include_numeric_features=include_numeric_features,
        use_residual_targets=use_residual_targets,
    )
    val_dataset = Subtask2ASequenceDataset(
        val_df,
        embeddings,
        seq_len=seq_len,
        include_numeric_features=include_numeric_features,
        use_residual_targets=use_residual_targets,
    )
    embedding_dim = embeddings.shape[1]
    return train_dataset, val_dataset, embedding_dim


def collate_sequence_batch(batch: List[dict]) -> dict:
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    out = {"inputs": inputs, "target": targets, "lengths": lengths}
    if "numeric_features" in batch[0]:
        out["numeric_features"] = torch.stack(
            [b["numeric_features"] for b in batch], dim=0
        )
    if "target_raw" in batch[0]:
        out["target_raw"] = torch.stack([b["target_raw"] for b in batch], dim=0)
    if "base_pred" in batch[0]:
        out["base_pred"] = torch.stack([b["base_pred"] for b in batch], dim=0)
    return out


def create_dataloaders(
    train_dataset: Subtask2ASequenceDataset,
    val_dataset: Subtask2ASequenceDataset,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    return train_loader, val_loader

