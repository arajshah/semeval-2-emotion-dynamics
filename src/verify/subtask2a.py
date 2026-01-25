from __future__ import annotations

from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd

from src.data_loader import load_all_data
from src.verify.shared import (
    CheckResult,
    VerifyContext,
    pass_result,
    fail_result,
    warn_result,
    print_results,
    exit_code,
    audit_record,
    write_audits_to_manifest,
)


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
                pass_result(
                    "subtask2a_loaded_delta_nans_allowed",
                    f"Loaded df contains NaNs in delta cols (expected): {loaded_nan_counts}",
                )
            )
            results[-1].hint = "df_raw may contain NaNs; eligibility = both deltas non-NaN."
        else:
            results.append(
                warn_result(
                    "subtask2a_loaded_delta_nans_allowed",
                    "No NaNs in loaded df delta cols; ensure df_raw row set is unchanged.",
                    hint="df_raw may contain NaNs; eligibility = both deltas non-NaN.",
                )
            )

    if len(df) != len(raw_df):
        results.append(
            fail_result(
                "subtask2a_df_raw_rowcount_matches_csv",
                f"df_raw rowcount mismatch: load_all_data returned {len(df)} rows but CSV has {len(raw_df)}.",
                hint=(
                    "Fix load_all_data() to return an unfiltered df for subtask2a; "
                    "split indices require df_raw be unmodified."
                ),
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2a_df_raw_rowcount_matches_csv",
            f"df_raw rowcount matches CSV ({len(df)} rows).",
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

    if "n_total" in payload and int(payload["n_total"]) != len(df):
        results.append(
            fail_result(
                "subtask2a_split_indices_valid",
                f"Split n_total mismatch in {split_path}: {payload['n_total']} != {len(df)}",
                hint="Regenerate split against the correct subtask2a dataframe.",
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

    if ctx.mode == "strict":
        results.extend(_phase2_artifact_checks(ctx, df))
        if ctx.run_id:
            results.extend(_phaseC_checks(ctx, df))
            results.extend(_phaseE2_audits(ctx, df))

    if ctx.run_id is None:
        results.append(
            warn_result(
                "subtask2a_anchored_preds_ok",
                "SKIP: run_id not provided; cannot validate anchored preds under reports/preds/.",
                hint="Run verify with --run_id <RUN_ID>.",
            )
        )
        return results

    preds_path = repo_root / "reports" / "preds" / f"subtask2a_val_user_preds__{ctx.run_id}.parquet"
    if not preds_path.exists():
        results.append(
            warn_result(
                "subtask2a_anchored_preds_ok",
                f"SKIP: anchored preds not found at {preds_path}",
                hint=f"Generate via phase0_eval with --task subtask2a --run_id {ctx.run_id}.",
            )
        )
        return results

    try:
        preds = pd.read_parquet(preds_path)
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Failed to read {preds_path}: {exc}",
                hint="Regenerate anchored preds and ensure parquet is readable.",
            )
        )
        return results

    required_cols = {
        "user_id",
        "anchor_idx",
        "anchor_text_id",
        "anchor_timestamp",
        "delta_valence_pred",
        "delta_arousal_pred",
        "delta_valence_true",
        "delta_arousal_true",
    }
    missing_cols = required_cols - set(preds.columns)
    if missing_cols:
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Missing required columns in {preds_path}: {sorted(missing_cols)}",
                hint="Regenerate anchored preds with the canonical schema.",
            )
        )
        return results

    if preds["user_id"].duplicated().any():
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Expected one row per user in {preds_path}",
                hint="Ensure anchoring selects exactly one row per user.",
            )
        )
        return results

    val_df_raw = df.iloc[val_idx].copy()
    val_df_raw["anchor_idx"] = np.asarray(val_idx, dtype=int)
    val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"])
    eligible_mask = val_df_raw["state_change_valence"].notna() & val_df_raw[
        "state_change_arousal"
    ].notna()
    eligible_df = val_df_raw.loc[eligible_mask].copy()
    expected_anchors = (
        eligible_df.sort_values(["user_id", "timestamp"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(1)
    )
    expected_users = set(expected_anchors["user_id"].tolist())
    pred_users = set(preds["user_id"].tolist())
    if expected_users != pred_users:
        missing_users = sorted(expected_users - pred_users)
        extra_users = sorted(pred_users - expected_users)
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Anchored preds users mismatch (missing={missing_users[:5]}, extra={extra_users[:5]}).",
                hint="Ensure anchored preds cover exactly users with eligible rows.",
            )
        )
        return results

    anchor_idx = preds["anchor_idx"].astype(int).to_numpy()
    if len(anchor_idx) != len(set(anchor_idx)):
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Duplicate anchor_idx values in {preds_path}",
                hint="Ensure one unique anchor per user.",
            )
        )
        return results

    val_set = set(val_idx)
    if not set(anchor_idx).issubset(val_set):
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"anchor_idx includes rows outside val split in {preds_path}",
                hint="Ensure anchor_idx comes from the frozen val indices.",
            )
        )
        return results

    if np.any(anchor_idx < 0) or np.any(anchor_idx >= len(df)):
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"anchor_idx out of bounds in {preds_path}",
                hint="Ensure anchor_idx refers to df_raw positional indices.",
            )
        )
        return results

    df_raw_ts = pd.to_datetime(df["timestamp"])
    for _, row in preds.iterrows():
        idx = int(row["anchor_idx"])
        raw_row = df.iloc[idx]
        if row["user_id"] != raw_row["user_id"]:
            results.append(
                fail_result(
                    "subtask2a_anchored_preds_ok",
                    f"user_id mismatch at anchor_idx={idx}",
                    hint="Ensure anchor_idx points to the correct user row.",
                )
            )
            return results
        if int(row["anchor_text_id"]) != int(raw_row["text_id"]):
            results.append(
                fail_result(
                    "subtask2a_anchored_preds_ok",
                    f"anchor_text_id mismatch at anchor_idx={idx}",
                    hint="Ensure anchor_text_id matches df_raw text_id.",
                )
            )
            return results
        if pd.to_datetime(row["anchor_timestamp"]) != df_raw_ts.iloc[idx]:
            results.append(
                fail_result(
                    "subtask2a_anchored_preds_ok",
                    f"anchor_timestamp mismatch at anchor_idx={idx}",
                    hint="Ensure anchor_timestamp matches df_raw timestamp.",
                )
            )
            return results

    expected_anchor_idx_by_user = expected_anchors.set_index("user_id")["anchor_idx"]
    for _, row in preds.iterrows():
        user_id = row["user_id"]
        expected_idx = int(expected_anchor_idx_by_user.loc[user_id])
        if int(row["anchor_idx"]) != expected_idx:
            results.append(
                fail_result(
                    "subtask2a_anchored_preds_ok",
                    f"Anchor not latest eligible for user {user_id}",
                    hint="Ensure anchor selection uses latest eligible timestamp.",
                )
            )
            return results

        raw_row = df.iloc[int(row["anchor_idx"])]
        if pd.isna(raw_row["state_change_valence"]) or pd.isna(raw_row["state_change_arousal"]):
            results.append(
                fail_result(
                    "subtask2a_anchored_preds_ok",
                    f"Anchor not eligible for user {user_id} at idx={int(row['anchor_idx'])}",
                    hint="Ensure anchors are selected only from eligible rows.",
                )
            )
            return results

    vals = preds[["delta_valence_pred", "delta_arousal_pred"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        results.append(
            fail_result(
                "subtask2a_anchored_preds_ok",
                f"Non-finite prediction values in {preds_path}",
                hint="Regenerate anchored preds with finite outputs.",
            )
        )
        return results

    results.append(
        pass_result(
            "subtask2a_anchored_preds_ok",
            f"Anchored preds schema and alignment OK: {preds_path}",
        )
    )

    return results


def _phaseC_checks(ctx: VerifyContext, df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root
    run_id = ctx.run_id or ""

    model_path = repo_root / "models" / "subtask2a_sequence" / "runs" / run_id / "model.pt"
    if not model_path.exists():
        results.append(
            fail_result(
                "subtask2a_phaseC_model_exists",
                f"Missing model checkpoint: {model_path}",
                hint="Train Phase C model for this run_id.",
            )
        )
    else:
        results.append(
            pass_result(
                "subtask2a_phaseC_model_exists",
                f"Found model checkpoint: {model_path}",
            )
        )

    preds_path = repo_root / "reports" / "preds" / f"subtask2a_val_user_preds__{run_id}.parquet"
    if not preds_path.exists():
        results.append(
            fail_result(
                "subtask2a_phaseC_preds_exists",
                f"Missing anchored dev preds: {preds_path}",
                hint="Run predict_subtask2a.py for this run_id.",
            )
        )
        return results

    preds = pd.read_parquet(preds_path)
    required_cols = {
        "run_id",
        "seed",
        "user_id",
        "anchor_idx",
        "anchor_text_id",
        "anchor_timestamp",
        "delta_valence_true",
        "delta_arousal_true",
        "delta_valence_pred",
        "delta_arousal_pred",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        results.append(
            fail_result(
                "subtask2a_phaseC_preds_schema",
                f"Missing required columns in {preds_path}: {sorted(missing)}",
                hint="Regenerate anchored dev preds with canonical schema.",
            )
        )
        return results
    if preds["user_id"].duplicated().any():
        results.append(
            fail_result(
                "subtask2a_phaseC_preds_schema",
                f"Expected one row per user in {preds_path}",
                hint="Ensure anchoring selects exactly one row per user.",
            )
        )
        return results

    split_path = repo_root / "reports" / "splits" / f"subtask2a_unseen_user_seed{ctx.seed}.json"
    payload = json.loads(split_path.read_text())
    _, val_idx = _load_split_indices(payload, split_path)
    val_df_raw = df.iloc[val_idx].copy()
    val_df_raw["idx"] = np.asarray(val_idx, dtype=int)
    val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"], errors="raise")
    eligible_df = val_df_raw[
        val_df_raw["state_change_valence"].notna()
        & val_df_raw["state_change_arousal"].notna()
    ].copy()
    expected_anchors = (
        eligible_df.sort_values(["user_id", "timestamp"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(1)
    )
    expected = expected_anchors.set_index("user_id")["idx"].to_dict()
    sample_users = list(expected.keys())[:20]
    for user_id in sample_users:
        pred_row = preds[preds["user_id"] == user_id]
        if pred_row.empty:
            results.append(
                fail_result(
                    "subtask2a_phaseC_anchor_spotcheck",
                    f"Missing user in preds: {user_id}",
                    hint="Anchored preds must include all eligible users.",
                )
            )
            return results
        if int(pred_row["anchor_idx"].iloc[0]) != int(expected[user_id]):
            results.append(
                fail_result(
                    "subtask2a_phaseC_anchor_spotcheck",
                    f"Anchor mismatch for user {user_id}",
                    hint="Anchors must be latest eligible timestamp per user.",
                )
            )
            return results

    results.append(
        pass_result(
            "subtask2a_phaseC_anchor_spotcheck",
            "Anchors match latest eligible rows for sampled users.",
        )
    )

    eval_path = repo_root / "reports" / "eval_records.csv"
    if not eval_path.exists():
        results.append(
            fail_result(
                "subtask2a_phaseC_eval_records",
                f"Missing eval_records.csv at {eval_path}",
                hint="Run phase0_eval for subtask2a to append a row.",
            )
        )
    else:
        eval_df = pd.read_csv(eval_path)
        match = eval_df[
            (eval_df["task"] == "subtask2a")
            & (eval_df["run_id"] == run_id)
            & (eval_df["seed"] == ctx.seed)
        ]
        if match.empty:
            results.append(
                fail_result(
                    "subtask2a_phaseC_eval_records",
                    f"No eval_records row for run_id={run_id} seed={ctx.seed}.",
                    hint="Run phase0_eval for subtask2a with --run_id.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2a_phaseC_eval_records",
                    f"Found eval_records row for run_id={run_id}.",
                )
            )

    require_submission = bool(getattr(ctx, "require_submission", False))
    forecast_path = repo_root / "reports" / "preds" / f"subtask2a_forecast_user_preds__{run_id}.parquet"
    submission_path = repo_root / "reports" / "submissions" / f"subtask2a_submission__{run_id}.csv"
    if require_submission or forecast_path.exists() or submission_path.exists():
        if not forecast_path.exists():
            results.append(
                fail_result(
                    "subtask2a_phaseC_forecast_preds",
                    f"Missing forecast preds: {forecast_path}",
                    hint="Run predict_subtask2a.py with --write_forecast 1.",
                )
            )
            return results
        fdf = pd.read_parquet(forecast_path)
        f_required = {
            "run_id",
            "seed",
            "user_id",
            "anchor_idx",
            "anchor_text_id",
            "anchor_timestamp",
            "delta_valence_pred",
            "delta_arousal_pred",
        }
        missing = f_required - set(fdf.columns)
        if missing:
            results.append(
                fail_result(
                    "subtask2a_phaseC_forecast_preds",
                    f"Missing forecast columns: {sorted(missing)}",
                    hint="Regenerate forecast preds with canonical schema.",
                )
            )
            return results
        if fdf["user_id"].duplicated().any():
            results.append(
                fail_result(
                    "subtask2a_phaseC_forecast_preds",
                    "Forecast preds must contain exactly one row per user.",
                    hint="Ensure forecast anchors are unique per user.",
                )
            )
            return results
        if require_submission and not submission_path.exists():
            results.append(
                fail_result(
                    "subtask2a_phaseC_submission",
                    f"Missing submission CSV: {submission_path}",
                    hint="Run submit_subtask2a.py for this run_id.",
                )
            )
            return results
        if submission_path.exists():
            sub_df = pd.read_csv(submission_path)
            required_cols = {
                "user_id",
                "pred_state_change_valence",
                "pred_state_change_arousal",
            }
            missing = required_cols - set(sub_df.columns)
            if missing:
                results.append(
                    fail_result(
                        "subtask2a_phaseC_submission",
                        f"Submission missing columns: {sorted(missing)}",
                        hint="Submission must follow the canonical schema.",
                    )
                )
            elif sub_df["user_id"].duplicated().any():
                results.append(
                    fail_result(
                        "subtask2a_phaseC_submission",
                        "Submission must contain exactly one row per user.",
                        hint="Remove duplicate user_id rows.",
                    )
                )
            else:
                results.append(
                    pass_result(
                        "subtask2a_phaseC_submission",
                        f"Submission CSV OK: {submission_path}",
                    )
                )

    return results


def _phaseE2_audits(ctx: VerifyContext, df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    audits: List[dict] = []
    repo_root = ctx.repo_root
    run_id = ctx.run_id or ""
    allow_warn = bool(getattr(ctx, "allow_warn", False))

    val_path = repo_root / "reports" / "preds" / f"subtask2a_val_user_preds__{run_id}.parquet"
    forecast_path = repo_root / "reports" / "preds" / f"subtask2a_forecast_user_preds__{run_id}.parquet"

    if not val_path.exists():
        audits.append(
            audit_record(
                "2a_val_preds_exists",
                "FAIL",
                "Missing dev preds parquet.",
                {"path": str(val_path)},
            )
        )
    else:
        preds = pd.read_parquet(val_path)
        required_cols = {
            "user_id",
            "anchor_idx",
            "anchor_text_id",
            "anchor_timestamp",
            "delta_valence_pred",
            "delta_arousal_pred",
            "delta_valence_true",
            "delta_arousal_true",
        }
        missing = required_cols - set(preds.columns)
        if missing:
            audits.append(
                audit_record(
                    "2a_val_schema",
                    "FAIL",
                    "Missing required columns in val preds.",
                    {"missing": sorted(missing)},
                )
            )
        elif preds.empty:
            audits.append(
                audit_record(
                    "2a_val_schema",
                    "FAIL",
                    "Val preds is empty.",
                    {"rows": int(len(preds))},
                )
            )
        elif preds["user_id"].duplicated().any():
            dupes = preds["user_id"][preds["user_id"].duplicated()].unique().tolist()
            audits.append(
                audit_record(
                    "2a_val_one_row_per_user",
                    "FAIL",
                    "Expected exactly one row per user in val preds.",
                    {"duplicate_users_sample": dupes[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2a_val_one_row_per_user",
                    "PASS",
                    "Val preds has one row per user.",
                    {"rows": int(len(preds)), "users": int(preds["user_id"].nunique())},
                )
            )

    if not forecast_path.exists():
        audits.append(
            audit_record(
                "2a_forecast_preds_exists",
                "WARN",
                "Forecast preds missing; skipping forecast audits.",
                {"path": str(forecast_path)},
            )
        )
    else:
        fp = pd.read_parquet(forecast_path)
        f_required = {
            "user_id",
            "anchor_idx",
            "anchor_text_id",
            "anchor_timestamp",
            "delta_valence_pred",
            "delta_arousal_pred",
        }
        f_missing = f_required - set(fp.columns)
        if f_missing:
            audits.append(
                audit_record(
                    "2a_forecast_schema",
                    "FAIL",
                    "Missing required columns in forecast preds.",
                    {"missing": sorted(f_missing)},
                )
            )
        elif fp.empty:
            audits.append(
                audit_record(
                    "2a_forecast_schema",
                    "FAIL",
                    "Forecast preds is empty.",
                    {"rows": int(len(fp))},
                )
            )
        elif fp["user_id"].duplicated().any():
            dupes = fp["user_id"][fp["user_id"].duplicated()].unique().tolist()
            audits.append(
                audit_record(
                    "2a_forecast_one_row_per_user",
                    "FAIL",
                    "Expected exactly one row per user in forecast preds.",
                    {"duplicate_users_sample": dupes[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2a_forecast_one_row_per_user",
                    "PASS",
                    "Forecast preds has one row per user.",
                    {"rows": int(len(fp)), "users": int(fp["user_id"].nunique())},
                )
            )

    # Anchor semantics (dev)
    split_path = repo_root / "reports" / "splits" / f"subtask2a_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        audits.append(
            audit_record(
                "2a_anchor_dev_semantics",
                "FAIL",
                "Missing split file for anchor audit.",
                {"path": str(split_path)},
            )
        )
    elif val_path.exists():
        payload = json.loads(split_path.read_text())
        _, val_idx = _load_split_indices(payload, split_path)
        val_df_raw = df.iloc[val_idx].copy()
        val_df_raw["idx"] = np.asarray(val_idx, dtype=int)
        val_df_raw["timestamp"] = pd.to_datetime(val_df_raw["timestamp"], errors="raise")
        eligible_df = val_df_raw[
            val_df_raw["state_change_valence"].notna()
            & val_df_raw["state_change_arousal"].notna()
        ].copy()
        expected = (
            eligible_df.sort_values(["user_id", "timestamp", "idx"], kind="stable")
            .groupby("user_id", sort=False)
            .tail(1)
        )
        expected_map = expected.set_index("user_id")["idx"].to_dict()
        preds = pd.read_parquet(val_path)
        pred_users = set(preds["user_id"])
        expected_users = set(expected_map.keys())
        missing_users = sorted(expected_users - pred_users)
        extra_users = sorted(pred_users - expected_users)
        if missing_users or extra_users:
            audits.append(
                audit_record(
                    "2a_anchor_dev_semantics",
                    "FAIL",
                    "Val preds users mismatch eligible users.",
                    {
                        "missing_users": missing_users[:10],
                        "extra_users": extra_users[:10],
                    },
                )
            )
        else:
            mismatch = 0
            for _, row in preds.iterrows():
                uid = row["user_id"]
                if int(row["anchor_idx"]) != int(expected_map[uid]):
                    mismatch += 1
            if mismatch:
                audits.append(
                    audit_record(
                        "2a_anchor_dev_semantics",
                        "FAIL",
                        "Anchor_idx mismatch for eligible users.",
                        {"n_mismatch_anchor_idx": int(mismatch)},
                    )
                )
            else:
                audits.append(
                    audit_record(
                        "2a_anchor_dev_semantics",
                        "PASS",
                        "Anchors match latest eligible timestamp per user.",
                        {
                            "n_val_users_total": int(val_df_raw["user_id"].nunique()),
                            "n_val_users_eligible": int(len(expected_map)),
                            "n_val_users_scored": int(len(preds)),
                        },
                    )
                )

    # Forecast cutoff semantics (if forecast preds present)
    if forecast_path.exists():
        cutoff_path = repo_root / "data" / "raw" / "test" / "test_subtask2.csv"
        if cutoff_path.exists():
            cutoff_df = pd.read_csv(cutoff_path)
            if {"user_id", "timestamp_min"}.issubset(cutoff_df.columns):
                cutoff_df["timestamp_min"] = pd.to_datetime(
                    cutoff_df["timestamp_min"], errors="raise"
                )
                df_ts = df.copy()
                df_ts["idx"] = df_ts.index
                df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], errors="raise")
                expected = []
                for uid, row in cutoff_df.groupby("user_id", sort=False).first().iterrows():
                    cutoff = row["timestamp_min"]
                    user_rows = df_ts[(df_ts["user_id"] == uid) & (df_ts["timestamp"] < cutoff)]
                    if user_rows.empty:
                        audits.append(
                            audit_record(
                                "2a_forecast_anchor_semantics",
                                "FAIL",
                                "No row before cutoff for forecasting user.",
                                {"user_id": uid},
                            )
                        )
                        continue
                    anchor_row = user_rows.sort_values(["timestamp", "idx"], kind="stable").iloc[-1]
                    expected.append(anchor_row)
                if expected:
                    expected_df = pd.DataFrame(expected)
                    expected_map = expected_df.set_index("user_id")["idx"].to_dict()
                    fp = pd.read_parquet(forecast_path)
                    fp_users = set(fp["user_id"])
                    expected_users = set(expected_map.keys())
                    missing = sorted(expected_users - fp_users)
                    extra = sorted(fp_users - expected_users)
                    if missing or extra:
                        audits.append(
                            audit_record(
                                "2a_forecast_anchor_semantics",
                                "FAIL",
                                "Forecast preds users mismatch cutoff users.",
                                {"missing_users": missing[:10], "extra_users": extra[:10]},
                            )
                        )
                    else:
                        mismatch = 0
                        for _, row in fp.iterrows():
                            if int(row["anchor_idx"]) != int(expected_map[row["user_id"]]):
                                mismatch += 1
                        if mismatch:
                            audits.append(
                                audit_record(
                                    "2a_forecast_anchor_semantics",
                                    "FAIL",
                                    "Forecast anchor_idx mismatch.",
                                    {"n_mismatch_anchor_idx": int(mismatch)},
                                )
                            )
                        else:
                            audits.append(
                                audit_record(
                                    "2a_forecast_anchor_semantics",
                                    "PASS",
                                    "Forecast anchors match cutoff semantics.",
                                    {"n_forecast_users": int(len(fp))},
                                )
                            )
            else:
                audits.append(
                    audit_record(
                        "2a_forecast_anchor_semantics",
                        "WARN",
                        "Cutoff file missing timestamp_min; skipping cutoff audit.",
                        {"path": str(cutoff_path)},
                    )
                )
        else:
            audits.append(
                audit_record(
                    "2a_forecast_anchor_semantics",
                    "WARN",
                    "Cutoff file missing; skipping forecast anchor audit.",
                    {"path": str(cutoff_path)},
                )
            )

    # Embedding coverage for anchor rows
    run_manifest_path = repo_root / "reports" / "runs" / f"{run_id}.json"
    embeddings_path = None
    if run_manifest_path.exists():
        try:
            manifest = json.loads(run_manifest_path.read_text())
            embeddings_path = (
                manifest.get("inputs", {})
                .get("embeddings", {})
                .get("path")
                or manifest.get("embeddings_path")
            )
        except Exception:
            embeddings_path = None
    if embeddings_path:
        emb_path = Path(embeddings_path)
        if not emb_path.is_absolute():
            emb_path = repo_root / emb_path
        if emb_path.exists() and val_path.exists():
            npz = np.load(emb_path, allow_pickle=False)
            if {"user_id", "text_id"}.issubset(npz.files):
                emb_pairs = set(zip(npz["user_id"], npz["text_id"]))
                preds = pd.read_parquet(val_path)
                anchor_pairs = set(zip(preds["user_id"], preds["anchor_text_id"]))
                missing_pairs = list(anchor_pairs - emb_pairs)
                if missing_pairs:
                    status = "WARN" if allow_warn else "FAIL"
                    audits.append(
                        audit_record(
                            "2a_embedding_coverage",
                            status,
                            "Missing embeddings for anchor rows.",
                            {"missing_pairs_sample": missing_pairs[:10]},
                        )
                    )
                else:
                    audits.append(
                        audit_record(
                            "2a_embedding_coverage",
                            "PASS",
                            "All anchor rows have embeddings.",
                            {"anchors": int(len(anchor_pairs))},
                        )
                    )
            else:
                audits.append(
                    audit_record(
                        "2a_embedding_coverage",
                        "FAIL",
                        "Embeddings file missing user_id/text_id arrays.",
                        {"path": str(emb_path)},
                    )
                )
        else:
            audits.append(
                audit_record(
                    "2a_embedding_coverage",
                    "WARN",
                    "Embeddings file missing; skipping embedding coverage audit.",
                    {"path": str(emb_path)},
                )
            )
    else:
        audits.append(
            audit_record(
                "2a_embedding_coverage",
                "WARN",
                "Run manifest missing embeddings path; cannot audit coverage.",
                {},
            )
        )

    # Leakage manifest check
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text())
        feature_names = (
            manifest.get("feature_names")
            or manifest.get("norm_stats", {}).get("feature_names")
            or []
        )
        if any("state_change" in str(name) for name in feature_names):
            audits.append(
                audit_record(
                    "2a_label_as_feature_manifest_check",
                    "FAIL",
                    "Feature list includes state_change labels.",
                    {"feature_names": feature_names},
                )
            )
        elif feature_names:
            audits.append(
                audit_record(
                    "2a_label_as_feature_manifest_check",
                    "PASS",
                    "Feature list excludes label columns.",
                    {"n_features": len(feature_names)},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2a_label_as_feature_manifest_check",
                    "WARN",
                    "Manifest lacks feature list; cannot audit label usage.",
                    {},
                )
            )
    else:
        audits.append(
            audit_record(
                "2a_label_as_feature_manifest_check",
                "WARN",
                "Run manifest missing; cannot audit label usage.",
                {"path": str(run_manifest_path)},
            )
        )

    # Convert audit records into CheckResults
    for audit in audits:
        if audit["status"] == "PASS":
            results.append(pass_result(audit["name"], audit["message"]))
        elif audit["status"] == "WARN":
            results.append(warn_result(audit["name"], audit["message"]))
        else:
            results.append(fail_result(audit["name"], audit["message"]))

    write_audits_to_manifest(repo_root, run_id, "subtask2a", audits)
    return results


def _phase2_artifact_checks(ctx: VerifyContext, df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root
    run_id = ctx.run_id
    if run_id is None:
        results.append(
            warn_result(
                "subtask2a_phase2_artifacts_ok",
                "SKIP: run_id not provided; cannot validate Phase 2 artifacts.",
                hint="Run verify with --run_id <RUN_ID>.",
            )
        )
        return results

    embeddings_path = repo_root / "data" / "processed" / f"subtask2a_embeddings__{run_id}.npz"
    if not embeddings_path.exists():
        results.append(
            warn_result(
                "subtask2a_phase2_artifacts_ok",
                f"Missing embeddings NPZ: {embeddings_path}",
                hint="Build embeddings with extract_embeddings for subtask2a.",
            )
        )
        return results

    try:
        npz = np.load(embeddings_path)
        if not {"embeddings", "user_id", "text_id"}.issubset(set(npz.files)):
            results.append(
                warn_result(
                    "subtask2a_phase2_artifacts_ok",
                    f"Embeddings NPZ missing required arrays: {embeddings_path}",
                    hint="Ensure embeddings NPZ includes embeddings/user_id/text_id.",
                )
            )
            return results
        if npz["embeddings"].shape[0] != len(npz["user_id"]) or npz["embeddings"].shape[0] != len(npz["text_id"]):
            results.append(
                warn_result(
                    "subtask2a_phase2_artifacts_ok",
                    f"Embeddings NPZ array length mismatch: {embeddings_path}",
                    hint="Regenerate embeddings NPZ.",
                )
            )
            return results
    except Exception as exc:
        results.append(
            warn_result(
                "subtask2a_phase2_artifacts_ok",
                f"Failed to load embeddings NPZ {embeddings_path}: {exc}",
                hint="Regenerate embeddings NPZ.",
            )
        )
        return results

    emb_index_df = pd.DataFrame(
        {
            "user_id": npz["user_id"],
            "text_id": npz["text_id"],
            "emb_index": np.arange(len(npz["user_id"])),
        }
    )
    try:
        merged = df.merge(
            emb_index_df,
            on=["user_id", "text_id"],
            how="inner",
            validate="one_to_one",
        )
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Embedding alignment failed: {exc}",
                hint="Ensure embeddings align one-to-one on user_id/text_id.",
            )
        )
        return results
    if len(merged) == 0:
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                "Embedding alignment returned zero rows.",
                hint="Ensure embeddings were built from the same subtask2a CSV.",
            )
        )
        return results

    model_path = repo_root / "models" / "subtask2a_sequence" / "runs" / run_id / "model.pt"
    if not model_path.exists():
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Missing model checkpoint: {model_path}",
                hint="Train Subtask2A sequence model with frozen split.",
            )
        )
        return results

    trainlog_path = repo_root / "reports" / "trainlogs" / "subtask2a" / f"subtask2a_trainlog__{run_id}.csv"
    if not trainlog_path.exists():
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Missing trainlog: {trainlog_path}",
                hint="Train Subtask2A sequence model to generate trainlog.",
            )
        )
        return results
    try:
        if pd.read_csv(trainlog_path).empty:
            results.append(
                fail_result(
                    "subtask2a_phase2_artifacts_ok",
                    f"Trainlog is empty: {trainlog_path}",
                    hint="Ensure training writes per-epoch logs.",
                )
            )
            return results
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Failed to read trainlog {trainlog_path}: {exc}",
                hint="Ensure trainlog CSV is readable.",
            )
        )
        return results

    run_json = repo_root / "reports" / "runs" / f"{run_id}.json"
    if not run_json.exists():
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Missing run metadata: {run_json}",
                hint="Ensure training writes reports/runs/{run_id}.json.",
            )
        )
        return results
    try:
        payload = json.loads(run_json.read_text())
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Failed to read run metadata {run_json}: {exc}",
                hint="Ensure run metadata JSON is valid.",
            )
        )
        return results

    required_keys = {"task", "seed", "split_path", "embeddings_path", "embeddings_sha256", "df_raw_len"}
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        results.append(
            fail_result(
                "subtask2a_phase2_artifacts_ok",
                f"Run metadata missing keys: {missing_keys}",
                hint="Ensure training/provenance writes required fields.",
            )
        )
        return results

    results.append(
        pass_result(
            "subtask2a_phase2_artifacts_ok",
            "Phase2 artifacts and metadata OK.",
        )
    )
    return results


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify Subtask 2A anchored preds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--require_submission", type=int, default=0)
    parser.add_argument("--allow_warn", type=int, default=0)
    args = parser.parse_args()

    ctx = VerifyContext(
        repo_root=Path(__file__).resolve().parents[2],
        mode="strict",
        tasks=["subtask2a"],
        seed=args.seed,
        run_id=args.run_id,
    )
    setattr(ctx, "require_submission", bool(args.require_submission))
    setattr(ctx, "allow_warn", bool(args.allow_warn))
    results = run_checks(ctx)
    print_results(
        f"Verify Subtask2a (seed={args.seed}, run_id={args.run_id})",
        results,
    )
    raise SystemExit(exit_code(results))


if __name__ == "__main__":
    _cli()
