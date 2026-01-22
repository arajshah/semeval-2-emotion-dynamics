from __future__ import annotations

from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd

from src.eval.splits import get_split_path, validate_split_payload, validate_unseen_user_disjoint
from src.verify.shared import CheckResult, VerifyContext, pass_result, fail_result

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

    return results
