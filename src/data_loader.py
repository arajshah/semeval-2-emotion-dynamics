from pathlib import Path
from typing import Dict
import math

import pandas as pd

DataBundle = Dict[str, pd.DataFrame]

EXPECTED_FILES: Dict[str, str] = {
    "subtask1": "train_subtask1.csv",
    "subtask2a": "train_subtask2a.csv",
    "subtask2b": "train_subtask2b.csv",
    "subtask2b_detailed": "train_subtask2b_detailed.csv",
    "subtask2b_user": "train_subtask2b_user_disposition_change.csv",
}

EXPECTED_COLUMNS: Dict[str, list[str]] = {
    "subtask1": [
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "collection_phase",
        "is_words",
        "valence",
        "arousal",
    ],
    "subtask2a": [
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "collection_phase",
        "is_words",
        "valence",
        "arousal",
        "state_change_valence",
        "state_change_arousal",
    ],
    "subtask2b": [
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "valence",
        "arousal",
        "group",
        "disposition_change_valence",
        "disposition_change_arousal",
    ],
    "subtask2b_detailed": [
        "user_id",
        "text_id",
        "text",
        "timestamp",
        "collection_phase",
        "is_words",
        "valence",
        "arousal",
        "text_num",
        "num_texts_per_user",
        "group",
        "mean_valence_half1",
        "mean_valence_half2",
        "mean_arousal_half1",
        "mean_arousal_half2",
        "disposition_change_valence",
        "disposition_change_arousal",
    ],
    "subtask2b_user": [
        "user_id",
        "disposition_change_valence",
        "disposition_change_arousal",
    ],
}


def _validate_columns(df: pd.DataFrame, expected_cols: list[str], name: str) -> None:
    """
    Ensure that `df` has exactly the expected columns (at least these columns).
    Raise a ValueError with a clear message if any are missing.
    """
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {name}: {missing}")


def _normalize_and_sort(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Cast columns to appropriate dtypes and sort rows for a given dataset `name`.
    Returns a new DataFrame.
    """
    normalized = df.copy()

    if "user_id" in normalized.columns:
        normalized["user_id"] = pd.to_numeric(normalized["user_id"], errors="raise").astype(int)
    if "text_id" in normalized.columns:
        normalized["text_id"] = pd.to_numeric(normalized["text_id"], errors="raise").astype(int)
    if "text" in normalized.columns:
        normalized["text"] = normalized["text"].astype(str)
    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="raise")
    if "collection_phase" in normalized.columns:
        normalized["collection_phase"] = pd.to_numeric(normalized["collection_phase"], errors="raise").astype(int)
    if "is_words" in normalized.columns:
        normalized["is_words"] = normalized["is_words"].astype("bool")

    for col in normalized.columns:
        if "valence" in col or "arousal" in col:
            normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(float)

    if "user_id" in normalized.columns and "timestamp" in normalized.columns:
        normalized = normalized.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    elif "user_id" in normalized.columns:
        normalized = normalized.sort_values(["user_id"]).reset_index(drop=True)

    return normalized


def _validate_value_ranges(df: pd.DataFrame, name: str) -> None:
    """
    Perform simple sanity checks on valence/arousal-like columns.
    Raise ValueError if values are wildly out of expected ranges.
    Print min/max for each checked column.
    """
    for col in df.columns:
        if "valence" in col or "arousal" in col:
            series = df[col]
            nan_count = series.isna().sum()
            col_values = series.dropna()

            if col_values.empty:
                print(f"[{name}] column '{col}' has only NaN values (skipping range check)")
                continue

            if not col_values.map(math.isfinite).all():
                raise ValueError(f"Non-finite values found in column '{col}' for {name}")

            col_min = col_values.min()
            col_max = col_values.max()
            if col_min < -5 or col_max > 5:
                raise ValueError(
                    f"Values out of expected range [-5, 5] for column '{col}' in {name}: min={col_min}, max={col_max}"
                )
            print(f"[{name}] column '{col}' min={col_min} max={col_max} (nan_count={nan_count})")


def load_all_data(data_dir: str = "data/raw") -> DataBundle:
    """
    Load and validate all SemEval Task 2 training CSVs from `data_dir`.

    Returns a dict mapping:
        'subtask1' -> train_subtask1 DataFrame
        'subtask2a' -> train_subtask2a DataFrame
        'subtask2b' -> train_subtask2b DataFrame
        'subtask2b_detailed' -> train_subtask2b_detailed DataFrame
        'subtask2b_user' -> train_subtask2b_user_disposition_change DataFrame
    """
    base_dir = _resolve_data_dir(data_dir)
    data_bundle: DataBundle = {}

    for name, filename in EXPECTED_FILES.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found for '{name}': {path}")

        df = pd.read_csv(path)
        _validate_columns(df, EXPECTED_COLUMNS[name], name)
        normalized = _normalize_and_sort(df, name)
        _validate_value_ranges(normalized, name)
        data_bundle[name] = normalized

    return data_bundle


def _find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking up from `start`.
    """
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    return start


def _resolve_data_dir(data_dir: str) -> Path:
    """
    Resolve `data_dir` relative to the repo root if it is not absolute.
    """
    base_dir = Path(data_dir)
    if base_dir.is_absolute():
        return base_dir

    repo_root = _find_repo_root(Path(__file__).resolve())
    return repo_root / base_dir


def print_data_summary(data: DataBundle) -> None:
    """
    Print basic summary statistics for each dataset in the bundle:
    - number of rows
    - number of unique users
    - for datasets with 'is_words': counts of essays vs feeling-words.
    """
    for name, df in data.items():
        rows = len(df)
        unique_users = df["user_id"].nunique() if "user_id" in df.columns else 0
        print(f"{name}: rows={rows}, unique_users={unique_users}")
        if "is_words" in df.columns:
            counts = df["is_words"].value_counts(dropna=False)
            feeling_words = counts.get(True, 0)
            essays = counts.get(False, 0)
            print(f"  is_words=True (feeling-words): {feeling_words}")
            print(f"  is_words=False (essays): {essays}")


if __name__ == "__main__":
    data = load_all_data()
    print_data_summary(data)

