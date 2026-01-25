from __future__ import annotations

from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd

from src.eval.splits import get_split_path, validate_split_payload, validate_unseen_user_disjoint
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
from src.subtask2b_features import (
    load_subtask2b_df_raw,
    load_subtask2b_embeddings_npz,
    merge_embeddings,
)
from src.data_loader import load_all_data
from src.utils.provenance import sha256_file

try:
    from src.data_loader import EXPECTED_COLUMNS
except Exception:
    EXPECTED_COLUMNS = {}


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


def _check_schema(
    df: pd.DataFrame, expected: List[str], check_id: str, path: Path
) -> CheckResult:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        return fail_result(
            check_id,
            f"Missing required columns in {path}: {missing}",
            hint="Ensure the raw CSV schema matches data_loader.EXPECTED_COLUMNS.",
        )
    return pass_result(check_id, f"Schema OK for {path}")


def run_checks(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root

    train_path = repo_root / "data" / "raw" / "train_subtask2b.csv"
    detailed_path = repo_root / "data" / "raw" / "train_subtask2b_detailed.csv"
    user_path = repo_root / "data" / "raw" / "train_subtask2b_user_disposition_change.csv"

    def read_csv(path: Path, check_id: str) -> pd.DataFrame | None:
        if not path.exists():
            results.append(
                fail_result(
                    check_id,
                    f"Missing raw data file: {path}",
                    hint="Place official SemEval files under data/raw/.",
                )
            )
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            results.append(
                fail_result(
                    check_id,
                    f"Failed to read {path}: {exc}",
                    hint="Place official SemEval files under data/raw/.",
                )
            )
            return None
        results.append(pass_result(check_id, f"Loaded {path}"))
        return df

    train_df = read_csv(train_path, "subtask2b_raw_train_exists")
    detailed_df = read_csv(detailed_path, "subtask2b_raw_detailed_exists")
    user_df = read_csv(user_path, "subtask2b_raw_user_exists")

    if train_df is not None:
        expected = EXPECTED_COLUMNS.get(
            "subtask2b",
            ["user_id", "text_id", "valence", "arousal", "group"],
        )
        results.append(_check_schema(train_df, expected, "subtask2b_schema_train", train_path))

    if detailed_df is not None:
        expected = EXPECTED_COLUMNS.get(
            "subtask2b_detailed",
            ["user_id", "text_id", "valence", "arousal", "group"],
        )
        results.append(
            _check_schema(detailed_df, expected, "subtask2b_schema_detailed", detailed_path)
        )

    if user_df is not None:
        expected = EXPECTED_COLUMNS.get(
            "subtask2b_user",
            ["user_id", "disposition_change_valence", "disposition_change_arousal"],
        )
        results.append(_check_schema(user_df, expected, "subtask2b_schema_user", user_path))

    split_path = repo_root / "reports" / "splits" / f"subtask2b_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        results.append(
            fail_result(
                "subtask2b_split_exists",
                f"Missing split file: {split_path}",
                hint="Generate frozen splits with Phase 0 tooling.",
            )
        )
        return results

    results.append(pass_result("subtask2b_split_exists", f"Found split file: {split_path}"))

    if train_df is None:
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Cannot validate split without {train_path}",
                hint="Ensure train_subtask2b.csv is present and readable.",
            )
        )
        return results

    try:
        payload = json.loads(split_path.read_text())
        train_idx, val_idx = _load_split_indices(payload, split_path)
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Failed to parse split indices from {split_path}: {exc}",
                hint="Ensure split JSON uses train_indices/val_indices schema.",
            )
        )
        return results

    try:
        train_idx = [int(i) for i in train_idx]
        val_idx = [int(i) for i in val_idx]
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Non-integer indices in {split_path}: {exc}",
                hint="Ensure split indices are integers.",
            )
        )
        return results

    if not train_idx or not val_idx:
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Empty train/val indices in {split_path}",
                hint="Ensure split JSON has non-empty train/val indices.",
            )
        )
        return results

    train_set = set(train_idx)
    val_set = set(val_idx)
    overlap = train_set.intersection(val_set)
    if overlap:
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Train/val overlap in {split_path} (count={len(overlap)})",
                hint="Ensure split indices are disjoint.",
            )
        )
        return results

    max_idx = max(train_set.union(val_set))
    min_idx = min(train_set.union(val_set))
    if min_idx < 0 or max_idx >= len(train_df):
        results.append(
            fail_result(
                "subtask2b_split_valid",
                f"Index out of bounds in {split_path} (n_rows={len(train_df)})",
                hint="Ensure split indices align to train_subtask2b.csv.",
            )
        )
        return results

    results.append(
        pass_result("subtask2b_split_valid", f"Split indices valid for {split_path}")
    )

    if ctx.mode == "strict":
        base_split_path = get_split_path(
            "subtask2b", "unseen_user", ctx.seed, split_key=None
        )
        user_split_path = get_split_path(
            "subtask2b", "unseen_user", ctx.seed, split_key="user_disposition_change"
        )

        def _validate_variant(
            df: pd.DataFrame,
            split_path: Path,
            split_key: str | None,
            variant_label: str,
        ) -> tuple[list[int], list[int]] | None:
            if not split_path.exists():
                results.append(
                    fail_result(
                        "subtask2b_split_in_bounds",
                        f"Missing split file: {split_path}",
                        hint=(
                            f"Run: python -m src.eval.splits --task subtask2b "
                            f"--variant {variant_label} --seed {ctx.seed}"
                        ),
                    )
                )
                return None
            try:
                payload = json.loads(split_path.read_text())
                validate_split_payload(
                    payload,
                    task="subtask2b",
                    regime="unseen_user",
                    seed=ctx.seed,
                    n_total=len(df),
                    split_key=split_key,
                )
                train_idx = [int(i) for i in payload["train_indices"]]
                val_idx = [int(i) for i in payload["val_indices"]]
                validate_unseen_user_disjoint(df, train_idx, val_idx)
            except Exception as exc:
                results.append(
                    fail_result(
                        "subtask2b_split_in_bounds",
                        f"Invalid split for {split_path}: {exc}",
                        hint=(
                            f"Regenerate: python -m src.eval.splits --task subtask2b "
                            f"--variant {variant_label} --seed {ctx.seed}"
                        ),
                    )
                )
                return None
            results.append(
                pass_result(
                    "subtask2b_split_in_bounds",
                    f"Split valid for {variant_label}: {split_path}",
                )
            )
            return train_idx, val_idx

        if train_df is None:
            results.append(
                fail_result(
                    "subtask2b_split_in_bounds",
                    f"Cannot run strict split checks without {train_path}",
                    hint="Ensure train_subtask2b.csv is present and readable.",
                )
            )
            return results

        base_indices = _validate_variant(train_df, base_split_path, None, "base")
        if user_df is None:
            results.append(
                fail_result(
                    "subtask2b_split_in_bounds",
                    f"Cannot validate user_disposition_change without {user_path}",
                    hint="Ensure train_subtask2b_user_disposition_change.csv exists.",
                )
            )
            return results
        _validate_variant(user_df, user_split_path, "user_disposition_change", "user_disposition_change")

        if base_indices is None:
            return results
        train_idx, val_idx = base_indices

        paths_checked = f"paths_checked={base_split_path}, {train_path}"
        if "user_id" not in train_df.columns:
            results.append(
                fail_result(
                    "subtask2b_unseen_user_no_overlap",
                    f"Missing user_id in {train_path}. {paths_checked}",
                    hint="Ensure train_subtask2b.csv includes user_id.",
                )
            )
            return results

        train_users = set(train_df.loc[train_idx, "user_id"].tolist())
        val_users = set(train_df.loc[val_idx, "user_id"].tolist())
        overlap_users = sorted(train_users.intersection(val_users))
        if overlap_users:
            results.append(
                fail_result(
                    "subtask2b_unseen_user_no_overlap",
                    f"Overlapping user_id across train/val (count={len(overlap_users)}, examples={overlap_users[:5]}). "
                    f"{paths_checked}",
                    hint="Regenerate split with user_id group separation.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2b_unseen_user_no_overlap",
                    "No user_id overlap between train and val.",
                )
            )

        target_pairs = [
            ("disposition_change_valence", "disposition_change_arousal"),
            ("state_change_valence", "state_change_arousal"),
        ]
        target_pair = None
        for pair in target_pairs:
            if all(col in train_df.columns for col in pair):
                target_pair = pair
                break
        if target_pair is None:
            results.append(
                fail_result(
                    "subtask2b_targets_finite",
                    f"Missing target delta columns in {train_path}. {paths_checked}",
                    hint="Expected disposition_change_* or state_change_* columns.",
                )
            )
            return results

        subset = train_df.loc[train_idx + val_idx, list(target_pair)]
        values = subset.to_numpy(dtype=float)
        non_finite = ~np.isfinite(values)
        if non_finite.any():
            counts = {
                target_pair[0]: int(np.isnan(values[:, 0]).sum()),
                target_pair[1]: int(np.isnan(values[:, 1]).sum()),
            }
            bad_rows = subset[~np.isfinite(subset).all(axis=1)].head(5)
            examples = bad_rows.index.tolist()
            results.append(
                fail_result(
                    "subtask2b_targets_finite",
                    f"Non-finite target values in {train_path}: {counts}, examples={examples}. {paths_checked}",
                    hint="Ensure target delta columns are finite on train+val indices.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2b_targets_finite",
                    f"Target columns finite for {target_pair}.",
                )
            )

        if detailed_df is not None:
            if "user_id" not in detailed_df.columns:
                results.append(
                    fail_result(
                        "subtask2b_join_keys_ok",
                        f"Missing user_id in {detailed_path}.",
                        hint="Ensure train_subtask2b_detailed.csv includes user_id.",
                    )
                )
        if user_df is not None:
            if "user_id" not in user_df.columns:
                results.append(
                    fail_result(
                        "subtask2b_join_keys_ok",
                        f"Missing user_id in {user_path}.",
                        hint="Ensure train_subtask2b_user_disposition_change.csv includes user_id.",
                    )
                )
            else:
                dup_users = user_df["user_id"][user_df["user_id"].duplicated()].unique().tolist()
                if dup_users:
                    results.append(
                        fail_result(
                            "subtask2b_join_keys_ok",
                            f"user_id not unique in {user_path} (examples={dup_users[:5]}).",
                            hint="Ensure one row per user in user disposition table.",
                        )
                    )
                else:
                    results.append(
                        pass_result(
                            "subtask2b_join_keys_ok",
                            "Join keys OK for detailed/user tables.",
                        )
                    )

        reports_dir = repo_root / "reports"
        patterns = [
            "subtask2b*pred*.parquet",
            "subtask2b*pred*.csv",
            "subtask2b*prediction*.parquet",
            "subtask2b*prediction*.csv",
        ]
        artifact_paths = []
        for pattern in patterns:
            artifact_paths.extend(reports_dir.glob(pattern))
        if not artifact_paths:
            results.append(
                pass_result(
                    "subtask2b_pred_alignment_ok",
                    "No 2B prediction artifacts found; skipping alignment checks.",
                )
            )
        else:
            for path in artifact_paths[:3]:
                try:
                    if path.suffix == ".parquet":
                        pred_df = pd.read_parquet(path)
                    else:
                        pred_df = pd.read_csv(path)
                except Exception as exc:
                    results.append(
                        fail_result(
                            "subtask2b_pred_alignment_ok",
                            f"Failed to read {path}: {exc}",
                            hint="Ensure prediction artifacts are readable.",
                        )
                    )
                    continue

                pred_cols = pred_df.columns.tolist()
                pair_candidates = [
                    ("pred_delta_valence", "pred_delta_arousal"),
                    ("valence_pred", "arousal_pred"),
                    ("delta_valence_pred", "delta_arousal_pred"),
                ]
                chosen_pair = None
                for pair in pair_candidates:
                    if all(col in pred_cols for col in pair):
                        chosen_pair = pair
                        break
                if chosen_pair is None:
                    results.append(
                        fail_result(
                            "subtask2b_pred_alignment_ok",
                            f"Missing prediction columns in {path}: {pred_cols}",
                            hint="Expected pred_delta_valence/pred_delta_arousal or similar.",
                        )
                    )
                    continue

                if "idx" in pred_df.columns:
                    pred_idx = set(pred_df["idx"].astype(int).tolist())
                    val_set = set(val_idx)
                    if pred_idx != val_set:
                        missing = sorted(val_set - pred_idx)
                        extra = sorted(pred_idx - val_set)
                        results.append(
                            fail_result(
                                "subtask2b_pred_alignment_ok",
                                f"Idx mismatch in {path}: missing={missing[:5]}, extra={extra[:5]}.",
                                hint="Ensure prediction artifact aligns to frozen val indices.",
                            )
                        )
                        continue
                else:
                    if len(pred_df) != len(val_idx):
                        results.append(
                            fail_result(
                                "subtask2b_pred_alignment_ok",
                                f"Length mismatch in {path}: rows={len(pred_df)} vs val={len(val_idx)}.",
                                hint="Ensure prediction artifact aligns to frozen val indices.",
                            )
                        )
                        continue

                vals = pred_df[list(chosen_pair)].to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    results.append(
                        fail_result(
                            "subtask2b_pred_alignment_ok",
                            f"Non-finite predictions in {path}.",
                            hint="Regenerate predictions with finite outputs.",
                        )
                    )
                    continue

                results.append(
                    pass_result(
                        "subtask2b_pred_alignment_ok",
                        f"Prediction artifact OK: {path}",
                    )
                )

    run_id = getattr(ctx, "run_id", None)
    if run_id:
        results.extend(_phase0_run_checks(ctx, run_id))
        results.extend(_phaseD_run_checks(ctx, run_id))
        results.extend(_phaseE2_audits(ctx, run_id))
        results.extend(_subtask2b_embeddings_integrity(ctx, run_id))
    return results


def _phaseE2_audits(ctx: VerifyContext, run_id: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    audits: List[dict] = []
    repo_root = ctx.repo_root
    allow_warn = bool(getattr(ctx, "allow_warn", False))

    val_preds = repo_root / "reports" / "preds" / f"subtask2b_val_user_preds__{run_id}.parquet"
    forecast_preds = repo_root / "reports" / "preds" / f"subtask2b_forecast_user_preds__{run_id}.parquet"

    if not val_preds.exists():
        audits.append(
            audit_record(
                "2b_val_preds_exists",
                "FAIL",
                "Missing dev preds parquet.",
                {"path": str(val_preds)},
            )
        )
    else:
        preds = pd.read_parquet(val_preds)
        required = {
            "user_id",
            "disposition_change_valence_true",
            "disposition_change_arousal_true",
            "disposition_change_valence_pred",
            "disposition_change_arousal_pred",
        }
        missing = required - set(preds.columns)
        if missing:
            audits.append(
                audit_record(
                    "2b_val_schema",
                    "FAIL",
                    "Missing required columns in val preds.",
                    {"missing": sorted(missing)},
                )
            )
        elif preds.empty:
            audits.append(
                audit_record(
                    "2b_val_schema",
                    "FAIL",
                    "Val preds is empty.",
                    {"rows": int(len(preds))},
                )
            )
        elif preds["user_id"].duplicated().any():
            dupes = preds["user_id"][preds["user_id"].duplicated()].unique().tolist()
            audits.append(
                audit_record(
                    "2b_val_one_row_per_user",
                    "FAIL",
                    "Expected exactly one row per user in val preds.",
                    {"duplicate_users_sample": dupes[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2b_val_one_row_per_user",
                    "PASS",
                    "Val preds has one row per user.",
                    {"rows": int(len(preds)), "users": int(preds["user_id"].nunique())},
                )
            )

    if not forecast_preds.exists():
        audits.append(
            audit_record(
                "2b_forecast_preds_exists",
                "WARN",
                "Forecast preds missing; skipping forecast audits.",
                {"path": str(forecast_preds)},
            )
        )
    else:
        fp = pd.read_parquet(forecast_preds)
        f_required = {
            "user_id",
            "disposition_change_valence_pred",
            "disposition_change_arousal_pred",
        }
        f_missing = f_required - set(fp.columns)
        if f_missing:
            audits.append(
                audit_record(
                    "2b_forecast_schema",
                    "FAIL",
                    "Missing required columns in forecast preds.",
                    {"missing": sorted(f_missing)},
                )
            )
        elif fp.empty:
            audits.append(
                audit_record(
                    "2b_forecast_schema",
                    "FAIL",
                    "Forecast preds is empty.",
                    {"rows": int(len(fp))},
                )
            )
        elif fp["user_id"].duplicated().any():
            dupes = fp["user_id"][fp["user_id"].duplicated()].unique().tolist()
            audits.append(
                audit_record(
                    "2b_forecast_one_row_per_user",
                    "FAIL",
                    "Expected exactly one row per user in forecast preds.",
                    {"duplicate_users_sample": dupes[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2b_forecast_one_row_per_user",
                    "PASS",
                    "Forecast preds has one row per user.",
                    {"rows": int(len(fp)), "users": int(fp["user_id"].nunique())},
                )
            )

    # Split immutability (user table)
    split_path = repo_root / "reports" / "splits" / f"subtask2b_user_disposition_change_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        audits.append(
            audit_record(
                "2b_split_exists",
                "FAIL",
                "Missing split file for user table.",
                {"path": str(split_path)},
            )
        )
    else:
        data = load_all_data()
        df_user_raw = data["subtask2b_user"]
        payload = json.loads(split_path.read_text())
        train_idx, val_idx = _load_split_indices(payload, split_path)
        train_users = set(df_user_raw.iloc[train_idx]["user_id"])
        val_users = set(df_user_raw.iloc[val_idx]["user_id"])
        overlap = train_users.intersection(val_users)
        if overlap:
            audits.append(
                audit_record(
                    "2b_split_user_disjoint",
                    "FAIL",
                    "Train/val user overlap in split.",
                    {"overlap_sample": list(overlap)[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2b_split_user_disjoint",
                    "PASS",
                    "Train/val users disjoint for split.",
                    {"n_train": len(train_users), "n_val": len(val_users)},
                )
            )

        if val_preds.exists():
            preds = pd.read_parquet(val_preds)
            pred_users = set(preds["user_id"])
            missing = sorted(list(val_users - pred_users))
            extra = sorted(list(pred_users - val_users))
            if missing or extra:
                audits.append(
                    audit_record(
                        "2b_val_user_coverage",
                        "FAIL",
                        "Val preds users mismatch split users.",
                        {"missing": missing[:10], "extra": extra[:10]},
                    )
                )
            else:
                audits.append(
                    audit_record(
                        "2b_val_user_coverage",
                        "PASS",
                        "Val preds users match split users.",
                        {"n_users": len(val_users)},
                    )
                )

    # Group=1 only leakage audit
    data = load_all_data()
    df_text_raw = data["subtask2b"]
    if val_preds.exists():
        preds = pd.read_parquet(val_preds)
        g1_users = set(df_text_raw.loc[df_text_raw["group"] == 1, "user_id"].tolist())
        pred_users = set(preds["user_id"])
        missing_g1 = sorted(list(pred_users - g1_users))
        if missing_g1:
            audits.append(
                audit_record(
                    "2b_group1_only",
                    "FAIL",
                    "Pred users include users without any group=1 rows.",
                    {"missing_group1_users": missing_g1[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2b_group1_only",
                    "PASS",
                    "All pred users have group=1 rows.",
                    {"n_users": len(pred_users)},
                )
            )

    # Embedding coverage for group=1 rows (val users)
    emb_path = repo_root / "data" / "processed" / "subtask2b_embeddings__deberta-v3-base__ml256.npz"
    if emb_path.exists() and split_path.exists():
        npz = np.load(emb_path, allow_pickle=False)
        if {"user_id", "text_id"}.issubset(npz.files):
            emb_pairs = set(zip(npz["user_id"], npz["text_id"]))
            g1_rows = df_text_raw[df_text_raw["group"] == 1]
            missing_pairs = list(
                set(zip(g1_rows["user_id"], g1_rows["text_id"])) - emb_pairs
            )
            if missing_pairs:
                status = "WARN" if allow_warn else "FAIL"
                audits.append(
                    audit_record(
                        "2b_embedding_coverage",
                        status,
                        "Missing embeddings for group=1 rows.",
                        {"missing_pairs_sample": missing_pairs[:10]},
                    )
                )
            else:
                audits.append(
                    audit_record(
                        "2b_embedding_coverage",
                        "PASS",
                        "All group=1 rows have embeddings.",
                        {"n_group1_rows": int(len(g1_rows))},
                    )
                )
        else:
            audits.append(
                audit_record(
                    "2b_embedding_coverage",
                    "FAIL",
                    "Embeddings file missing user_id/text_id arrays.",
                    {"path": str(emb_path)},
                )
            )
    else:
        audits.append(
            audit_record(
                "2b_embedding_coverage",
                "WARN",
                "Embeddings file missing; skipping coverage audit.",
                {"path": str(emb_path)},
            )
        )

    # Forecast coverage against marker users
    marker_path = repo_root / "data" / "raw" / "test" / "subtask2b_forecasting_user_marker.csv"
    if forecast_preds.exists() and marker_path.exists():
        marker_df = pd.read_csv(marker_path)
        if "is_forecasting_user" in marker_df.columns:
            is_fc = marker_df["is_forecasting_user"]
            if is_fc.dtype != bool:
                is_fc = is_fc.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
            marker_df = marker_df[is_fc].copy()
        marker_users = set(marker_df["user_id"].unique().tolist())
        fp = pd.read_parquet(forecast_preds)
        pred_users = set(fp["user_id"].unique().tolist())
        missing = sorted(list(marker_users - pred_users))
        extra = sorted(list(pred_users - marker_users))
        if missing or extra:
            audits.append(
                audit_record(
                    "2b_forecast_user_coverage",
                    "FAIL",
                    "Forecast preds users mismatch marker users.",
                    {"missing": missing[:10], "extra": extra[:10]},
                )
            )
        else:
            audits.append(
                audit_record(
                    "2b_forecast_user_coverage",
                    "PASS",
                    "Forecast preds users match marker users.",
                    {"n_users": len(marker_users)},
                )
            )

    for audit in audits:
        if audit["status"] == "PASS":
            results.append(pass_result(audit["name"], audit["message"]))
        elif audit["status"] == "WARN":
            results.append(warn_result(audit["name"], audit["message"]))
        else:
            results.append(fail_result(audit["name"], audit["message"]))

    write_audits_to_manifest(repo_root, run_id, "subtask2b", audits)
    return results


def _subtask2b_embeddings_integrity(ctx: VerifyContext, run_id: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root
    df_raw = load_subtask2b_df_raw().copy()
    embeddings_path = repo_root / "data" / "processed" / f"subtask2b_embeddings__{run_id}.npz"
    if not embeddings_path.exists():
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Missing embeddings file: {embeddings_path}",
                hint="Run embeddings extraction with --task subtask2b and --run_id.",
            )
        )
        return results

    metadata_path = repo_root / "reports" / "runs" / f"{run_id}.json"
    if not metadata_path.exists():
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Missing run metadata: {metadata_path}",
                hint="Embeddings extraction should write reports/runs/{run_id}.json.",
            )
        )
        return results
    metadata = json.loads(metadata_path.read_text())
    quick = bool(metadata.get("quick", False))
    quick_n = metadata.get("quick_n", None)

    npz = np.load(embeddings_path, allow_pickle=False)
    required = {"embeddings", "user_id", "text_id"}
    missing = required - set(npz.files)
    if missing:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Embeddings NPZ missing arrays: {sorted(missing)}",
                hint="Regenerate embeddings with the canonical extractor.",
            )
        )
        return results

    embeddings = npz["embeddings"]
    user_ids = npz["user_id"]
    text_ids = npz["text_id"]
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Embeddings must be float32 2D; got dtype={embeddings.dtype}, ndim={embeddings.ndim}.",
                hint="Regenerate embeddings with CLS pooling and float32 outputs.",
            )
        )
        return results
    if user_ids.ndim != 1 or text_ids.ndim != 1:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "user_id/text_id arrays must be 1D.",
                hint="Regenerate embeddings with canonical schema.",
            )
        )
        return results
    if len(user_ids) != embeddings.shape[0] or len(text_ids) != embeddings.shape[0]:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Embeddings array lengths do not match user_id/text_id.",
                hint="Regenerate embeddings with canonical schema.",
            )
        )
        return results

    expected_df = df_raw.head(int(quick_n)) if quick and quick_n else df_raw
    if len(expected_df) != embeddings.shape[0]:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Embeddings length {embeddings.shape[0]} does not match expected df_raw length {len(expected_df)}.",
                hint="Ensure df_raw is unmodified and quick mode matches quick_n.",
            )
        )
        return results

    df_user = expected_df["user_id"].to_numpy()
    df_text = expected_df["text_id"].to_numpy()
    if not (np.array_equal(user_ids, df_user) and np.array_equal(text_ids, df_text)):
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Embedding store must be row-aligned to df_raw; found ordering mismatch.",
                hint="Do not sort/shuffle before embedding extraction.",
            )
        )
        return results

    pairs = pd.DataFrame({"user_id": user_ids, "text_id": text_ids})
    if pairs.duplicated().any():
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Duplicate (user_id, text_id) pairs found in embeddings NPZ.",
                hint="Regenerate embeddings with unique row alignment to df_raw.",
            )
        )
        return results

    sha_expected = metadata.get("embeddings_sha256")
    if not sha_expected:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Missing embeddings_sha256 in run metadata.",
                hint="Ensure embeddings extraction writes sha256 into reports/runs/{run_id}.json.",
            )
        )
        return results
    sha_actual = sha256_file(embeddings_path)
    if sha_expected != sha_actual:
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "embeddings_sha256 mismatch between metadata and file.",
                hint="Regenerate embeddings or update metadata by rerunning extraction.",
            )
        )
        return results

    if metadata.get("task") != "subtask2b":
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Metadata task mismatch: {metadata.get('task')}",
                hint="Ensure embeddings run metadata has task=subtask2b.",
            )
        )
        return results
    if metadata.get("model_name") != "microsoft/deberta-v3-base":
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                f"Metadata model_name mismatch: {metadata.get('model_name')}",
                hint="Use microsoft/deberta-v3-base for embeddings.",
            )
        )
        return results
    if metadata.get("max_length") != 256 or metadata.get("pooling") != "cls":
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Metadata max_length/pooling mismatch.",
                hint="Use max_length=256 and CLS pooling for embeddings.",
            )
        )
        return results
    if metadata.get("df_raw_len") != len(expected_df):
        results.append(
            fail_result(
                "subtask2b_embeddings_integrity_ok",
                "Metadata df_raw_len does not match expected length.",
                hint="Ensure metadata df_raw_len matches extraction length.",
            )
        )
        return results

    results.append(
        pass_result(
            "subtask2b_embeddings_integrity_ok",
            f"Embeddings integrity OK for run_id={run_id}.",
        )
    )
    return results


def _phase0_run_checks(ctx: VerifyContext, run_id: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root
    df_raw = load_subtask2b_df_raw()
    split_path = repo_root / "reports" / "splits" / f"subtask2b_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        results.append(
            fail_result(
                "subtask2b_phase0_split_exists",
                f"Missing split file: {split_path}",
                hint="Do not resplit; ensure the frozen split JSON exists.",
            )
        )
        return results

    payload = json.loads(split_path.read_text())
    train_idx, val_idx = _load_split_indices(payload, split_path)
    if "n_total" in payload and int(payload["n_total"]) != len(df_raw):
        results.append(
            fail_result(
                "subtask2b_phase0_split_n_total",
                f"Split n_total={payload['n_total']} does not match df_raw len={len(df_raw)}.",
                hint="Ensure split was generated for train_subtask2b_detailed.csv.",
            )
        )
        return results

    train_users = set(df_raw.iloc[train_idx]["user_id"])
    val_users = set(df_raw.iloc[val_idx]["user_id"])
    overlap = train_users.intersection(val_users)
    if overlap:
        results.append(
            fail_result(
                "subtask2b_phase0_unseen_user_ok",
                f"User overlap between train/val (count={len(overlap)}).",
                hint="Ensure unseen-user split by user_id.",
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2b_phase0_unseen_user_ok",
            "Train/val users are disjoint for unseen-user split.",
        )
    )

    embeddings_path = getattr(
        ctx,
        "embeddings_path",
        repo_root / "data" / "processed" / "subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    try:
        emb_map_df, embeddings = load_subtask2b_embeddings_npz(embeddings_path)
        merged = merge_embeddings(df_raw, emb_map_df)
        merged.attrs["embeddings"] = embeddings
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2b_phase0_embeddings_ok",
                f"Embeddings merge failed: {exc}",
                hint="Regenerate embeddings and ensure 1:1 join on (user_id,text_id).",
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2b_phase0_embeddings_ok",
            f"Embeddings merged 1:1 for df_raw: {embeddings_path}",
        )
    )

    model_joblib = repo_root / "models" / "subtask2b_user" / "runs" / run_id / "model.joblib"
    model_pt = repo_root / "models" / "subtask2b_user" / "runs" / run_id / "model.pt"
    norm_stats = repo_root / "models" / "subtask2b_user" / "runs" / run_id / "norm_stats.json"
    if model_joblib.exists():
        results.append(
            pass_result("subtask2b_phase0_model_exists", f"Found model: {model_joblib}")
        )
    elif model_pt.exists() and norm_stats.exists():
        results.append(
            pass_result("subtask2b_phase0_model_exists", f"Found model: {model_pt}")
        )
    else:
        results.append(
            fail_result(
                "subtask2b_phase0_model_exists",
                f"Missing model checkpoint: {model_joblib} or {model_pt}",
                hint="Train a 2B model for this run_id.",
            )
        )
        return results

    preds_path = repo_root / "reports" / "preds" / f"subtask2b_val_user_preds__{run_id}.parquet"
    if not preds_path.exists():
        results.append(
            fail_result(
                "subtask2b_phase0_preds_exists",
                f"Missing preds parquet: {preds_path}",
                hint="Run the 2B predict script to generate preds.",
            )
        )
        return results

    try:
        preds = pd.read_parquet(preds_path)
    except Exception as exc:
        results.append(
            fail_result(
                "subtask2b_phase0_preds_schema",
                f"Failed to read preds parquet {preds_path}: {exc}",
                hint="Regenerate preds parquet.",
            )
        )
        return results

    def _first_present(cols: list[str]) -> str | None:
        for col in cols:
            if col in preds.columns:
                return col
        return None

    true_v_col = _first_present(
        ["disposition_change_valence_true", "dispo_change_valence_true", "delta_valence_true"]
    )
    true_a_col = _first_present(
        ["disposition_change_arousal_true", "dispo_change_arousal_true", "delta_arousal_true"]
    )
    pred_v_col = _first_present(
        ["disposition_change_valence_pred", "dispo_change_valence_pred", "delta_valence_pred"]
    )
    pred_a_col = _first_present(
        ["disposition_change_arousal_pred", "dispo_change_arousal_pred", "delta_arousal_pred"]
    )
    if not all([true_v_col, true_a_col, pred_v_col, pred_a_col]):
        results.append(
            fail_result(
                "subtask2b_phase0_preds_schema",
                f"Missing required columns in preds: {preds.columns.tolist()}",
                hint="Ensure preds parquet uses canonical columns for true/pred.",
            )
        )
        return results
    if preds["user_id"].duplicated().any():
        results.append(
            fail_result(
                "subtask2b_phase0_preds_schema",
                "Preds must contain exactly one row per user.",
                hint="Ensure per-user aggregation is correct.",
            )
        )
        return results

    val_df_raw = df_raw.iloc[val_idx]
    eligible_users = (
        val_df_raw[val_df_raw["group"] == 1]["user_id"]
        .drop_duplicates(keep="first")
        .tolist()
    )
    pred_users = preds["user_id"].tolist()
    if set(pred_users) != set(eligible_users):
        missing_users = sorted(set(eligible_users) - set(pred_users))
        extra_users = sorted(set(pred_users) - set(eligible_users))
        results.append(
            fail_result(
                "subtask2b_phase0_user_eligibility",
                f"Pred users mismatch eligible users (missing={missing_users[:5]}, extra={extra_users[:5]}).",
                hint="Preds must cover exactly users with group==1 rows in val split.",
            )
        )
        return results

    results.append(
        pass_result(
            "subtask2b_phase0_preds_schema",
            f"Preds schema and one-row-per-user OK: {preds_path}",
        )
    )

    eval_path = repo_root / "reports" / "eval_records.csv"
    if not eval_path.exists():
        results.append(
            fail_result(
                "subtask2b_phase0_eval_records",
                f"Missing eval_records.csv at {eval_path}",
                hint="Run phase0_eval for subtask2b to append a row.",
            )
        )
        return results

    eval_df = pd.read_csv(eval_path)
    match = eval_df[
        (eval_df["task"] == "subtask2b")
        & (eval_df["run_id"] == run_id)
        & (eval_df["seed"] == ctx.seed)
    ]
    if match.empty:
        results.append(
            fail_result(
                "subtask2b_phase0_eval_records",
                f"No eval_records row for run_id={run_id} seed={ctx.seed}.",
                hint="Run phase0_eval for subtask2b with --run_id.",
            )
        )
    else:
        results.append(
            pass_result(
                "subtask2b_phase0_eval_records",
                f"Found eval_records row for run_id={run_id}.",
            )
        )

    return results


def _phaseD_run_checks(ctx: VerifyContext, run_id: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root
    data = load_all_data()
    df_user_raw = data["subtask2b_user"]
    df_text_raw = data["subtask2b"]

    split_path = repo_root / "reports" / "splits" / f"subtask2b_user_disposition_change_unseen_user_seed{ctx.seed}.json"
    if not split_path.exists():
        results.append(
            fail_result(
                "subtask2b_phaseD_split_exists",
                f"Missing split file: {split_path}",
                hint="Ensure frozen user split JSON exists.",
            )
        )
        return results
    payload = json.loads(split_path.read_text())
    train_idx, val_idx = _load_split_indices(payload, split_path)
    if "n_total" in payload and int(payload["n_total"]) != len(df_user_raw):
        results.append(
            fail_result(
                "subtask2b_phaseD_split_n_total",
                f"Split n_total={payload['n_total']} does not match df_user_raw len={len(df_user_raw)}.",
                hint="Ensure split was generated for subtask2b_user_disposition_change.",
            )
        )
        return results
    train_users = set(df_user_raw.iloc[train_idx]["user_id"])
    val_users = set(df_user_raw.iloc[val_idx]["user_id"])
    overlap = train_users.intersection(val_users)
    if overlap:
        results.append(
            fail_result(
                "subtask2b_phaseD_unseen_user_ok",
                f"User overlap between train/val (count={len(overlap)}).",
                hint="Ensure unseen-user split by user_id.",
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2b_phaseD_unseen_user_ok",
            "Train/val users are disjoint for unseen-user split.",
        )
    )

    model_pt = repo_root / "models" / "subtask2b_user" / "runs" / run_id / "model.pt"
    norm_stats = repo_root / "models" / "subtask2b_user" / "runs" / run_id / "norm_stats.json"
    if not model_pt.exists() or not norm_stats.exists():
        results.append(
            fail_result(
                "subtask2b_phaseD_model_exists",
                f"Missing model.pt or norm_stats.json in {model_pt.parent}",
                hint="Train Phase D model for this run_id.",
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2b_phaseD_model_exists",
            f"Found model checkpoint: {model_pt}",
        )
    )

    preds_path = repo_root / "reports" / "preds" / f"subtask2b_val_user_preds__{run_id}.parquet"
    if not preds_path.exists():
        results.append(
            fail_result(
                "subtask2b_phaseD_preds_exists",
                f"Missing dev preds: {preds_path}",
                hint="Run predict_subtask2b_user.py --mode val.",
            )
        )
        return results
    preds = pd.read_parquet(preds_path)
    required_cols = {
        "run_id",
        "seed",
        "user_id",
        "disposition_change_valence_pred",
        "disposition_change_arousal_pred",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        results.append(
            fail_result(
                "subtask2b_phaseD_preds_schema",
                f"Missing required columns in {preds_path}: {sorted(missing)}",
                hint="Regenerate preds with canonical schema.",
            )
        )
        return results
    if preds["user_id"].duplicated().any():
        results.append(
            fail_result(
                "subtask2b_phaseD_preds_schema",
                "Preds must contain exactly one row per user.",
                hint="Ensure per-user aggregation is correct.",
            )
        )
        return results

    g1_users = set(df_text_raw.loc[df_text_raw["group"] == 1, "user_id"].tolist())
    eligible_users = set(val_users).intersection(g1_users)
    pred_users = set(preds["user_id"])
    if pred_users != eligible_users:
        missing_users = sorted(eligible_users - pred_users)
        extra_users = sorted(pred_users - eligible_users)
        results.append(
            fail_result(
                "subtask2b_phaseD_user_eligibility",
                f"Pred users mismatch eligible users (missing={missing_users[:5]}, extra={extra_users[:5]}).",
                hint="Preds must cover exactly users with group==1 rows.",
            )
        )
        return results
    results.append(
        pass_result(
            "subtask2b_phaseD_user_eligibility",
            "Pred users match eligible val users (group==1 only).",
        )
    )

    eval_path = repo_root / "reports" / "eval_records.csv"
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        match = eval_df[
            (eval_df["task"] == "subtask2b")
            & (eval_df["run_id"] == run_id)
            & (eval_df["seed"] == ctx.seed)
        ]
        if match.empty:
            results.append(
                fail_result(
                    "subtask2b_phaseD_eval_records",
                    f"No eval_records row for run_id={run_id} seed={ctx.seed}.",
                    hint="Run phase0_eval for subtask2b with --run_id.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2b_phaseD_eval_records",
                    f"Found eval_records row for run_id={run_id}.",
                )
            )

    submission_path = repo_root / "reports" / "submissions" / f"subtask2b_submission__{run_id}.csv"
    forecast_path = repo_root / "reports" / "preds" / f"subtask2b_forecast_user_preds__{run_id}.parquet"
    if forecast_path.exists():
        fdf = pd.read_parquet(forecast_path)
        f_required = {
            "run_id",
            "seed",
            "user_id",
            "disposition_change_valence_pred",
            "disposition_change_arousal_pred",
        }
        missing = f_required - set(fdf.columns)
        if missing:
            results.append(
                fail_result(
                    "subtask2b_phaseD_forecast_preds",
                    f"Forecast preds missing columns: {sorted(missing)}",
                    hint="Regenerate forecast preds with canonical schema.",
                )
            )
            return results
        if fdf["user_id"].duplicated().any():
            results.append(
                fail_result(
                    "subtask2b_phaseD_forecast_preds",
                    "Forecast preds must contain exactly one row per user.",
                    hint="Ensure forecast aggregation is correct.",
                )
            )
            return results
        results.append(
            pass_result(
                "subtask2b_phaseD_forecast_preds",
                f"Forecast preds OK: {forecast_path}",
            )
        )
    if not submission_path.exists():
        results.append(
            fail_result(
                "subtask2b_phaseD_submission",
                f"Missing submission CSV: {submission_path}",
                hint="Run submit_subtask2b.py for this run_id.",
            )
        )
    else:
        sub_df = pd.read_csv(submission_path)
        required_cols = {"user_id", "pred_dispo_change_valence", "pred_dispo_change_arousal"}
        missing = required_cols - set(sub_df.columns)
        if missing:
            results.append(
                fail_result(
                    "subtask2b_phaseD_submission",
                    f"Submission missing columns: {sorted(missing)}",
                    hint="Submission must follow canonical schema.",
                )
            )
        elif sub_df["user_id"].duplicated().any():
            results.append(
                fail_result(
                    "subtask2b_phaseD_submission",
                    "Submission must contain exactly one row per user.",
                    hint="Remove duplicate user_id rows.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask2b_phaseD_submission",
                    f"Submission CSV OK: {submission_path}",
                )
            )

    return results


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify Subtask 2B Phase-0 artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--allow_warn", type=int, default=0)
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    ctx = VerifyContext(
        repo_root=repo_root,
        mode="strict",
        tasks=["subtask2b"],
        seed=args.seed,
        run_id=args.run_id,
    )
    setattr(ctx, "embeddings_path", Path(args.embeddings_path))
    setattr(ctx, "allow_warn", bool(args.allow_warn))
    results = run_checks(ctx)
    print_results(
        f"Subtask2B Verify (seed={args.seed}, run_id={args.run_id})",
        results,
    )
    raise SystemExit(exit_code(results))


if __name__ == "__main__":
    _cli()
