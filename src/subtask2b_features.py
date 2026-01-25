from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.data_loader import load_all_data


REQUIRED_COLUMNS = {
    "user_id",
    "text_id",
    "text",
    "group",
    "valence",
    "arousal",
    "disposition_change_valence",
    "disposition_change_arousal",
}


def load_subtask2b_df_raw() -> pd.DataFrame:
    df_raw = load_all_data()["subtask2b"].copy().reset_index(drop=True)
    missing = REQUIRED_COLUMNS - set(df_raw.columns)
    if missing:
        raise ValueError(f"Subtask2B missing required columns: {sorted(missing)}")
    return df_raw


def compute_user_halves(df_text: pd.DataFrame) -> pd.DataFrame:
    df = df_text.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "text_id" not in df.columns:
        df["text_id"] = np.arange(len(df))
    df["row_idx"] = np.arange(len(df))
    df = df.sort_values(["user_id", "timestamp", "text_id", "row_idx"], kind="stable")

    counts = df.groupby("user_id", sort=False)["user_id"].count().rename("n_texts")
    df = df.join(counts, on="user_id")
    df["cut"] = (df["n_texts"] // 2).astype(int)
    df["rank"] = df.groupby("user_id", sort=False).cumcount()
    df["half_group"] = np.where(df["rank"] < df["cut"], 1, 2)
    return df.drop(columns=["row_idx", "rank"])


def audit_disposition_labels(
    df_text_raw: pd.DataFrame,
    df_user_raw: pd.DataFrame,
    *,
    atol: float = 1e-6,
) -> Dict[str, float]:
    if "group" in df_text_raw.columns:
        halves = df_text_raw.copy()
        halves["half_group"] = pd.to_numeric(halves["group"], errors="raise").astype(int)
    elif "half_group" in df_text_raw.columns:
        halves = df_text_raw.copy()
    else:
        halves = compute_user_halves(df_text_raw)

    valid = set(pd.Series(halves["half_group"]).dropna().unique().tolist())
    if not valid.issubset({1, 2}):
        raise ValueError(f"Invalid half_group values found: {sorted(valid)} (expected only 1/2)")

    g1 = halves[halves["half_group"] == 1]
    g2 = halves[halves["half_group"] == 2]

    mean1 = g1.groupby("user_id", sort=False)[["valence", "arousal"]].mean()
    mean2 = g2.groupby("user_id", sort=False)[["valence", "arousal"]].mean()
    joined = mean1.join(mean2, lsuffix="_1", rsuffix="_2", how="inner")

    joined["delta_v"] = joined["valence_2"] - joined["valence_1"]
    joined["delta_a"] = joined["arousal_2"] - joined["arousal_1"]
    user_labels = df_user_raw.set_index("user_id")[
        ["disposition_change_valence", "disposition_change_arousal"]
    ]
    aligned = joined.join(user_labels, how="inner")
    if aligned.empty:
        return {
            "n_users_compared": 0,
            "n_users_insufficient_texts": int(df_user_raw["user_id"].nunique()),
            "max_abs_diff_valence": 0.0,
            "max_abs_diff_arousal": 0.0,
            "n_mismatch_valence": 0,
            "n_mismatch_arousal": 0,
        }

    diff_v = np.abs(aligned["delta_v"] - aligned["disposition_change_valence"])
    diff_a = np.abs(aligned["delta_a"] - aligned["disposition_change_arousal"])

    summary = {
        "n_users_compared": int(len(aligned)),
        "n_users_insufficient_texts": int(df_user_raw["user_id"].nunique() - len(aligned)),
        "max_abs_diff_valence": float(diff_v.max()),
        "max_abs_diff_arousal": float(diff_a.max()),
        "n_mismatch_valence": int((diff_v > atol).sum()),
        "n_mismatch_arousal": int((diff_a > atol).sum()),
    }
    print(
        "2B label audit: "
        f"compared={summary['n_users_compared']} "
        f"insufficient={summary['n_users_insufficient_texts']} "
        f"max_diff_v={summary['max_abs_diff_valence']:.6f} "
        f"max_diff_a={summary['max_abs_diff_arousal']:.6f} "
        f"mismatch_v={summary['n_mismatch_valence']} "
        f"mismatch_a={summary['n_mismatch_arousal']}"
    )
    return summary


def load_subtask2b_embeddings_npz(path: Path | str) -> Tuple[pd.DataFrame, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings NPZ not found: {path}")
    npz = np.load(path, allow_pickle=False)
    required = {"embeddings", "user_id", "text_id"}
    if not required.issubset(set(npz.files)):
        raise ValueError(f"Embeddings NPZ missing arrays: {sorted(required - set(npz.files))}")
    embeddings = npz["embeddings"].astype(np.float32)
    user_ids = npz["user_id"]
    text_ids = npz["text_id"]
    if embeddings.shape[0] != len(user_ids) or embeddings.shape[0] != len(text_ids):
        raise ValueError("Embeddings NPZ array lengths do not match.")
    emb_map_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "text_id": text_ids,
            "emb_index": np.arange(len(user_ids), dtype=int),
        }
    )
    return emb_map_df, embeddings


def load_subtask2b_embeddings_mapping(
    path: Path | str,
) -> Tuple[Dict[Tuple, int], np.ndarray]:
    emb_map_df, embeddings = load_subtask2b_embeddings_npz(path)
    key_to_idx = {
        (row.user_id, row.text_id): int(row.emb_index)
        for row in emb_map_df.itertuples(index=False)
    }
    return key_to_idx, embeddings


def merge_embeddings(df_raw: pd.DataFrame, emb_map_df: pd.DataFrame) -> pd.DataFrame:
    merged = df_raw.merge(
        emb_map_df, on=["user_id", "text_id"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(df_raw):
        missing = set(zip(df_raw["user_id"], df_raw["text_id"])) - set(
            zip(merged["user_id"], merged["text_id"])
        )
        sample = list(missing)[:10]
        raise ValueError(
            f"Embeddings coverage mismatch: {len(df_raw)} rows -> {len(merged)} after merge. "
            f"Missing keys (sample): {sample}"
        )
    if merged["emb_index"].isna().any():
        raise ValueError("Missing emb_index after embeddings merge.")
    return merged


def build_user_level_dataset(
    df_raw_with_emb: pd.DataFrame, user_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = df_raw_with_emb.attrs.get("embeddings")
    if embeddings is None:
        raise ValueError("Embeddings array not attached to df_raw_with_emb.attrs['embeddings'].")

    feature_rows = []
    target_rows = []
    kept_users = []

    for user_id in user_ids:
        user_df = df_raw_with_emb[df_raw_with_emb["user_id"] == user_id]
        group1_df = user_df[user_df["group"] == 1]
        if group1_df.empty:
            continue

        emb_indices = group1_df["emb_index"].to_numpy()
        emb_mat = embeddings[emb_indices]
        emb_mean = emb_mat.mean(axis=0)
        mean_valence = float(group1_df["valence"].mean())
        mean_arousal = float(group1_df["arousal"].mean())
        features = np.concatenate([emb_mean, np.array([mean_valence, mean_arousal])])
        target_v = float(user_df["disposition_change_valence"].iloc[0])
        target_a = float(user_df["disposition_change_arousal"].iloc[0])

        feature_rows.append(features)
        target_rows.append([target_v, target_a])
        kept_users.append(user_id)

    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(target_rows, dtype=np.float32)
    users = np.asarray(kept_users)
    return X, y, users


def fit_norm_stats(X_num: np.ndarray) -> Dict[str, List[float]]:
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_norm_stats(X_num: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (X_num - mean) / std


def build_subtask2b_user_features(
    df_user: pd.DataFrame,
    df_text_raw: pd.DataFrame,
    embeddings: Tuple[pd.DataFrame, np.ndarray] | Tuple[Dict[Tuple, int], np.ndarray],
    *,
    enforce_group1_only: bool = True,
    pooling: str = "mean",
    add_traj_stats: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    if isinstance(embeddings[0], pd.DataFrame):
        emb_map_df, emb_arr = embeddings  # type: ignore[assignment]
        key_to_idx = {
            (row.user_id, row.text_id): int(row.emb_index)
            for row in emb_map_df.itertuples(index=False)
        }
    else:
        key_to_idx, emb_arr = embeddings  # type: ignore[assignment]

    if "group" in df_text_raw.columns:
        halves = df_text_raw.copy()
        halves["half_group"] = pd.to_numeric(halves["group"], errors="raise").astype(int)
    elif "half_group" in df_text_raw.columns:
        halves = df_text_raw.copy()
    else:
        halves = compute_user_halves(df_text_raw)

    valid = set(pd.Series(halves["half_group"]).dropna().unique().tolist())
    if not valid.issubset({1, 2}):
        raise ValueError(f"Invalid half_group values found: {sorted(valid)} (expected only 1/2)")

    g1 = halves[halves["half_group"] == 1]

    feature_rows: List[np.ndarray] = []
    num_rows: List[np.ndarray] = []
    user_list: List[str] = []
    meta_rows: List[dict] = []

    for user_id in df_user["user_id"].tolist():
        user_rows = g1[g1["user_id"] == user_id]
        if user_rows.empty:
            continue

        user_rows = user_rows.sort_values(["timestamp", "text_id"], kind="stable")
        emb_indices = []
        for _, row in user_rows.iterrows():
            key = (row["user_id"], row["text_id"])
            if key not in key_to_idx:
                raise RuntimeError(f"Missing embedding for key {key}")
            emb_indices.append(key_to_idx[key])
        emb_mat = emb_arr[np.array(emb_indices, dtype=int)]
        mean_emb = emb_mat.mean(axis=0)
        last_emb = emb_mat[-1]
        if pooling == "mean_last":
            X_emb = np.concatenate([mean_emb, last_emb])
        elif pooling == "last":
            X_emb = last_emb
        else:
            X_emb = mean_emb

        v = user_rows["valence"].to_numpy(dtype=float)
        a = user_rows["arousal"].to_numpy(dtype=float)
        v_mean = float(np.mean(v))
        a_mean = float(np.mean(a))
        v_last = float(v[-1])
        a_last = float(a[-1])
        v_std = float(np.std(v)) if len(v) > 1 else 0.0
        a_std = float(np.std(a)) if len(a) > 1 else 0.0
        v_last_minus_first = float(v[-1] - v[0])
        a_last_minus_first = float(a[-1] - a[0])
        if len(v) > 1:
            idx = np.arange(len(v))
            v_slope = float(np.polyfit(idx, v, 1)[0])
            a_slope = float(np.polyfit(idx, a, 1)[0])
        else:
            v_slope = 0.0
            a_slope = 0.0

        if "n_texts" in user_rows.columns:
            n_total = int(user_rows["n_texts"].iloc[0])
        else:
            n_total = int((halves["user_id"] == user_id).sum())

        if "cut" in user_rows.columns:
            cut = int(user_rows["cut"].iloc[0])
        else:
            cut = int(len(user_rows))

        X_num = np.array(
            [
                v_mean,
                a_mean,
                v_last,
                a_last,
                v_std,
                a_std,
                v_last_minus_first,
                a_last_minus_first,
                v_slope,
                a_slope,
                float(len(user_rows)),  # n_group1
                float(n_total),         # n_total texts
                float(cut),             # group1 size
            ],
            dtype=np.float32,
        )

        feature_rows.append(X_emb.astype(np.float32))
        num_rows.append(X_num)
        user_list.append(user_id)
        meta_rows.append(
            {
                "user_id": user_id,
                "n_total": int(n_total),
                "cut": int(cut),
                "n_group1": int(len(user_rows)),
                "timestamp_last_g1": user_rows["timestamp"].iloc[-1],
                "text_id_last_g1": user_rows["text_id"].iloc[-1],
            }
        )

    if pooling == "mean_last":
        emb_dim = emb_arr.shape[1] * 2
    elif pooling == "last":
        emb_dim = emb_arr.shape[1]
    else:
        emb_dim = emb_arr.shape[1]
    X_emb = (
        np.stack(feature_rows, axis=0).astype(np.float32)
        if feature_rows
        else np.zeros((0, emb_dim), dtype=np.float32)
    )
    X_num = np.stack(num_rows, axis=0).astype(np.float32) if num_rows else np.zeros((0, 13), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X_emb, X_num, user_list, meta_df
