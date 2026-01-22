from __future__ import annotations

from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd

from src.data_loader import load_all_data
from src.verify.shared import CheckResult, VerifyContext, pass_result, fail_result


def _load_split_indices(payload: dict, split_path: Path) -> tuple[list, list]:
    candidates = [
        ("train_indices", "val_indices"),
        ("train_idx", "val_idx"),
        ("train", "val"),
    ]
    for train_key, val_key in candidates:
        if train_key in payload and val_key in payload:
            return payload[train_key], payload[val_key]
    raise ValueError(f"Unsupported split schema in {split_path}")


def run_checks(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root

    raw_path = repo_root / "data" / "raw" / "train_subtask2a.csv"
    if raw_path.exists():
        results.append(pass_result("subtask2a_raw_present", f"Found {raw_path}"))
    else:
        results.append(
            fail_result(
                "subtask2a_raw_present",
                f"Missing raw data file: {raw_path}",
                hint="Place the SemEval CSVs under data/raw/.",
            )
        )
        return results

    try:
        data = load_all_data(data_dir=str(repo_root / "data" / "raw"))
        df = data["subtask2a"]
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_columns_ok",
                f"Failed to load Subtask 2A data from {raw_path}: {exc}",
                hint="Ensure data/raw files are present and valid.",
            )
        )
        return results

    try:
        raw_df = pd.read_csv(raw_path)
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_delta_nans_expected",
                f"Failed to read raw Subtask 2A CSV {raw_path}: {exc}",
                hint="Ensure train_subtask2a.csv is readable.",
            )
        )
        return results

    required_cols = [
        "user_id",
        "text",
        "valence",
        "arousal",
        "state_change_valence",
        "state_change_arousal",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        results.append(
            fail_result(
                "subtask2a_columns_ok",
                f"Missing required columns in {raw_path}: {missing}",
                hint="Verify train_subtask2a.csv schema matches expected columns.",
            )
        )
    else:
        results.append(
            pass_result(
                "subtask2a_columns_ok",
                f"Required columns present in {raw_path}",
            )
        )

    delta_cols = ["state_change_valence", "state_change_arousal"]
    missing_delta_cols = [col for col in delta_cols if col not in raw_df.columns]
    if missing_delta_cols:
        results.append(
            fail_result(
                "subtask2a_raw_delta_nans_allowed",
                f"Delta columns missing in {raw_path}: {missing_delta_cols}",
                hint="Ensure state_change_valence/state_change_arousal exist.",
            )
        )
        return results

    raw_nan_counts = {
        col: int(raw_df[col].isna().sum()) for col in delta_cols
    }
    results.append(
        pass_result(
            "subtask2a_raw_delta_nans_allowed",
            f"Raw delta NaNs allowed in {raw_path}: {raw_nan_counts}",
        )
    )
    results[-1].hint = (
        "NaNs in raw delta columns are allowed; strict mode validates model-ready data via loader."
    )

    if ctx.mode == "strict":
        loaded_nan_counts = {
            col: int(df[col].isna().sum()) for col in delta_cols if col in df.columns
        }
        if any(count > 0 for count in loaded_nan_counts.values()):
            results.append(
                fail_result(
                    "subtask2a_loaded_delta_no_nans",
                    f"Loader produced NaNs in delta columns: {loaded_nan_counts}",
                    hint=(
                        "Loader must fill/drop NaNs for Subtask 2A delta columns; "
                        "fix in data_loader preprocessing."
                    ),
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2a_loaded_delta_no_nans",
                    "Loaded Subtask 2A delta columns have no NaNs",
                )
            )

    split_path = repo_root / "reports" / "splits" / f"subtask2a_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        results.append(
            fail_result(
                "subtask2a_split_exists",
                f"Missing split file: {split_path}",
                hint="Run Phase 0 to generate frozen splits.",
            )
        )
        return results

    results.append(
        pass_result("subtask2a_split_exists", f"Found split file: {split_path}")
    )

    try:
        payload = json.loads(split_path.read_text())
        train_idx, val_idx = _load_split_indices(payload, split_path)
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_split_indices_valid",
                f"Failed to parse split indices from {split_path}: {exc}",
                hint="Ensure split JSON uses train_indices/val_indices schema.",
            )
        )
        return results

    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    if not train_idx or not val_idx:
        results.append(
            fail_result(
                "subtask2a_split_indices_valid",
                f"Empty train/val indices in {split_path}",
                hint="Regenerate the frozen split with valid indices.",
            )
        )
        return results

    train_set = set(train_idx)
    val_set = set(val_idx)
    overlap = train_set.intersection(val_set)
    if overlap:
        results.append(
            fail_result(
                "subtask2a_split_indices_valid",
                f"Train/val overlap in {split_path} (count={len(overlap)})",
                hint="Check that frozen split is unseen-user and disjoint.",
            )
        )
    else:
        max_idx = max(train_set.union(val_set))
        min_idx = min(train_set.union(val_set))
        if min_idx < 0 or max_idx >= len(df):
            results.append(
                fail_result(
                    "subtask2a_split_indices_valid",
                    f"Index out of bounds in {split_path} (n_rows={len(df)})",
                    hint="Ensure split indices align to train_subtask2a.csv.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2a_split_indices_valid",
                    f"Split indices valid for {split_path}",
                )
            )

    train_users = set(df.iloc[train_idx]["user_id"].tolist())
    val_users = set(df.iloc[val_idx]["user_id"].tolist())
    overlap_users = sorted(train_users.intersection(val_users))
    if overlap_users:
        sample = overlap_users[:10]
        results.append(
            fail_result(
                "subtask2a_unseen_user_ok",
                f"Overlapping user_id across train/val in {split_path}: {sample}",
                hint="Regenerate split with group-based user_id separation.",
            )
        )
    else:
        results.append(
            pass_result(
                "subtask2a_unseen_user_ok",
                "No user_id overlap between train and val",
            )
        )

    preds_path = repo_root / "reports" / "subtask2a_predictions.parquet"
    if preds_path.exists():
        try:
            preds = pd.read_parquet(preds_path)
        except Exception as exc:
            results.append(
                fail_result(
                    "subtask2a_predictions_artifact_ok",
                    f"Failed to read predictions artifact {preds_path}: {exc}",
                    hint="Regenerate predictions with src.eval.analysis_tools utilities.",
                )
            )
            return results

        candidate_sets = [
            ("delta_val_pred", "delta_aro_pred"),
            ("valence_pred", "arousal_pred"),
        ]
        pred_cols = None
        for col_a, col_b in candidate_sets:
            if col_a in preds.columns and col_b in preds.columns:
                pred_cols = (col_a, col_b)
                break

        if pred_cols is None:
            results.append(
                fail_result(
                    "subtask2a_predictions_artifact_ok",
                    f"Missing prediction columns in {preds_path}",
                    hint="Expected delta_val_pred/delta_aro_pred or valence_pred/arousal_pred.",
                )
            )
        else:
            vals = preds[list(pred_cols)].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                results.append(
                    fail_result(
                        "subtask2a_predictions_artifact_ok",
                        f"Non-finite values in predictions {preds_path}",
                        hint="Regenerate predictions and ensure outputs are finite.",
                    )
                )
            else:
                results.append(
                    pass_result(
                        "subtask2a_predictions_artifact_ok",
                        f"Predictions artifact OK: {preds_path}",
                    )
                )
    else:
        results.append(
            pass_result(
                "subtask2a_predictions_artifact_ok",
                f"SKIP: predictions artifact not found at {preds_path}",
            )
        )

    return results
