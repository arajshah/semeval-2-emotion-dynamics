from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import csv
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.provenance import merge_run_metadata


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    message: str
    hint: Optional[str] = None
    severity: str = ""

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = "PASS" if self.passed else "FAIL"


@dataclass
class VerifyContext:
    repo_root: Path
    mode: str
    tasks: List[str]
    seed: int
    run_id: Optional[str] = None
    allow_warn: bool = False


def pass_result(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id=check_id, passed=True, message=message, severity="PASS")


def fail_result(check_id: str, message: str, hint: Optional[str] = None) -> CheckResult:
    return CheckResult(
        check_id=check_id, passed=False, message=message, hint=hint, severity="FAIL"
    )


def warn_result(check_id: str, message: str, hint: Optional[str] = None) -> CheckResult:
    return CheckResult(
        check_id=check_id, passed=True, message=message, hint=hint, severity="WARN"
    )


def audit_record(
    name: str,
    status: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"
    record: Dict[str, Any] = {
        "name": name,
        "status": status,
        "message": message,
    }
    if details is not None:
        record["details"] = details
    return record


def write_audits_to_manifest(
    repo_root: Path,
    run_id: str,
    task: str,
    audits: List[Dict[str, Any]],
) -> None:
    try:
        merge_run_metadata(
            repo_root=repo_root,
            run_id=run_id,
            updates={
                "audits": {
                    "phaseE2": {
                        "task": task,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "results": audits,
                    }
                }
            },
        )
    except Exception as exc:
        audits.append(
            audit_record(
                name="phaseE2_manifest_write",
                status="WARN",
                message="Could not update run manifest",
                details={"error": str(exc)},
            )
        )


def print_results(header: str, results: List[CheckResult]) -> None:
    print(header)
    for result in results:
        status = result.severity or ("PASS" if result.passed else "FAIL")
        print(f"{status}: {result.check_id} — {result.message}")
        if result.hint:
            print(f"hint: {result.hint}")


def exit_code(results: List[CheckResult]) -> int:
    return 0 if all(result.severity != "FAIL" for result in results) else 1


def run_checks(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(_check_repo_root(ctx))
    results.append(_check_python_import_src(ctx))
    results.extend(_check_raw_datasets(ctx))
    results.extend(_check_splits(ctx))
    results.extend(_check_reports_dirs(ctx))
    results.append(_check_reports_writable(ctx))
    results.append(_check_eval_records_schema(ctx))
    return results


def run_strict_shared_checks(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(_check_train_requires_split_path(ctx))
    results.append(_check_eval_records_schema_strict(ctx))
    results.append(_check_artifact_naming_invariants(ctx))
    return results


def _check_train_requires_split_path(ctx: VerifyContext) -> CheckResult:
    module = "src." + "train" + "_subtask1_transformer"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:
        return fail_result(
            "train_requires_split_path",
            f"Failed to run training help: {exc}",
            hint="Update training CLI to require explicit --split_path; never auto-generate splits.",
        )
    if "--split_path" in proc.stdout:
        return pass_result(
            "train_requires_split_path",
            "Training help includes --split_path.",
        )
    return fail_result(
        "train_requires_split_path",
        "Training help missing --split_path.",
        hint="Update training CLI to require explicit --split_path; never auto-generate splits.",
    )


def _check_eval_records_schema_strict(ctx: VerifyContext) -> CheckResult:
    path = ctx.repo_root / "reports" / "eval_records.csv"
    if not path.exists():
        return warn_result(
            "eval_records_strict_schema",
            f"SKIP: {path} not found.",
            hint="Initialize via python -m src.eval.phase0_eval --seed 42",
        )
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return fail_result(
            "eval_records_strict_schema",
            f"Failed to read {path}: {exc}",
            hint="Ensure eval_records.csv is readable.",
        )

    cols = set(df.columns)
    required = {
        "run_id",
        "git_commit",
        "seed",
        "timestamp",
        "config_hash",
        "task",
        "regime",
        "slice",
        "model_tag",
    }
    missing = sorted(required - cols)
    if missing:
        return fail_result(
            "eval_records_strict_schema",
            f"Missing required columns in {path}: {missing}",
            hint=f"Add missing columns to {path}.",
        )

    subtask1_required = {
        "r_composite_valence",
        "r_composite_arousal",
        "r_within_valence",
        "r_between_valence",
        "r_within_arousal",
        "r_between_arousal",
    }
    missing_s1 = sorted(subtask1_required - cols)
    if missing_s1:
        return fail_result(
            "eval_records_strict_schema",
            f"Missing Subtask 1 columns in {path}: {missing_s1}",
            hint=f"Ensure eval_records.csv includes Subtask 1 correlation columns.",
        )

    delta_val = "r_delta_valence"
    delta_aro = "r_delta_arousal"
    if delta_val in cols and delta_aro in cols:
        return pass_result("eval_records_strict_schema", f"eval_records schema OK: {path}")

    alt_val = [c for c in cols if "delta" in c and "valence" in c]
    alt_aro = [c for c in cols if "delta" in c and "arousal" in c]
    if alt_val and alt_aro:
        return pass_result("eval_records_strict_schema", f"eval_records schema OK: {path}")

    return fail_result(
        "eval_records_strict_schema",
        f"Missing delta correlation columns in {path}.",
        hint="Ensure r_delta_valence and r_delta_arousal (or current canonical names) exist.",
    )


def _check_artifact_naming_invariants(ctx: VerifyContext) -> CheckResult:
    reports_dir = ctx.repo_root / "reports"
    
    root_pred_parquets = sorted(
        list((ctx.repo_root / "reports").glob("*val_preds*.parquet"))
        + list((ctx.repo_root / "reports").glob("*val_user_agg*.parquet"))
    )
    if root_pred_parquets:
        return fail_result(
            "artifact_naming_invariants",
            f"Prediction artifacts must not be in reports/ root. Found: {root_pred_parquets}",
            hint="Move prediction artifacts under reports/preds/<task>/.",
        )

    trainlogs_dir = reports_dir / "trainlogs"
    allowed_dirs = [
        trainlogs_dir / "subtask1",
        trainlogs_dir / "subtask2a",
        trainlogs_dir / "subtask2b",
    ]

    root_trainlogs = sorted(p for p in reports_dir.glob("*trainlog*.csv") if p.is_file())
    if root_trainlogs:
        return fail_result(
            "artifact_naming_invariants",
            f"Trainlog CSVs must not be in reports/ root. Found: {root_trainlogs}",
            hint="Move them under reports/trainlogs/<task>/ (subtask1, subtask2a, subtask2b).",
        )

    if not trainlogs_dir.exists():
        return pass_result("artifact_naming_invariants", "Artifact naming invariants OK.")

    direct_trainlogs = sorted(p for p in trainlogs_dir.glob("*trainlog*.csv") if p.is_file())
    if direct_trainlogs:
        return fail_result(
            "artifact_naming_invariants",
            f"Trainlog CSVs must not be directly under reports/trainlogs/. Found: {direct_trainlogs}",
            hint="Place trainlogs under reports/trainlogs/<task>/ instead.",
        )

    allowed_prefixes = tuple(str(d.resolve()) + "/" for d in allowed_dirs)
    misplaced = []
    for p in trainlogs_dir.rglob("*trainlog*.csv"):
        if not p.is_file():
            continue
        p_abs = str(p.resolve())
        if not p_abs.startswith(allowed_prefixes):
            misplaced.append(p)

    if misplaced:
        return fail_result(
            "artifact_naming_invariants",
            f"Trainlog CSVs must live under reports/trainlogs/<task>/. Found misplaced: {sorted(misplaced)}",
            hint="Allowed locations: reports/trainlogs/subtask1/, reports/trainlogs/subtask2a/, reports/trainlogs/subtask2b/.",
        )

    return pass_result("artifact_naming_invariants", "Artifact naming invariants OK.")


def _check_repo_root(ctx: VerifyContext) -> CheckResult:
    readme_ok = (ctx.repo_root / "README.md").exists()
    src_ok = (ctx.repo_root / "src").exists()
    if readme_ok and src_ok:
        return pass_result(
            "repo_root_valid",
            f"Found README.md and src/ under {ctx.repo_root}",
        )
    return fail_result(
        "repo_root_valid",
        f"Missing README.md or src/ under {ctx.repo_root}",
        hint="Run from repo root or ensure README.md and src/ exist.",
    )


def _check_python_import_src(ctx: VerifyContext) -> CheckResult:
    original_path = list(sys.path)
    try:
        if str(ctx.repo_root) not in sys.path:
            sys.path.insert(0, str(ctx.repo_root))
        importlib.import_module("src")
        return pass_result("python_import_src", f"Imported src from {ctx.repo_root}")
    except Exception as exc:
        return fail_result(
            "python_import_src",
            f"Failed to import src from {ctx.repo_root}: {exc}",
            hint="Run from repo root and ensure your venv is active.",
        )
    finally:
        sys.path = original_path


def _check_raw_datasets(ctx: VerifyContext) -> List[CheckResult]:
    required = [
        "data/raw/train_subtask1.csv",
        "data/raw/train_subtask2a.csv",
        "data/raw/train_subtask2b.csv",
        "data/raw/train_subtask2b_detailed.csv",
        "data/raw/train_subtask2b_user_disposition_change.csv",
    ]
    results: List[CheckResult] = []
    for rel_path in required:
        path = ctx.repo_root / rel_path
        if path.exists():
            results.append(pass_result("raw_datasets_present", f"Found {path}"))
        else:
            results.append(
                fail_result(
                    "raw_datasets_present",
                    f"Missing dataset: {path}",
                )
            )
    return results


def _load_split_indices(payload: dict, split_path: Path) -> tuple[list, list]:
    candidates = [
        ("train_idx", "val_idx"),
        ("train_indices", "val_indices"),
        ("train", "val"),
    ]
    for train_key, val_key in candidates:
        if train_key in payload and val_key in payload:
            return payload[train_key], payload[val_key]
    raise ValueError(f"Unsupported split schema in {split_path}")


def _validate_indices(
    train_idx: list,
    val_idx: list,
    n_rows: int,
    split_path: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    merged = list(train_idx) + list(val_idx)
    if not all(isinstance(i, (int, np.integer)) for i in merged):
        results.append(
            fail_result(
                "split_files_exist_and_valid",
                f"Non-integer indices in {split_path}",
            )
        )
        return results

    train_set = set(int(i) for i in train_idx)
    val_set = set(int(i) for i in val_idx)
    overlap = train_set.intersection(val_set)
    if overlap:
        results.append(
            fail_result(
                "split_files_exist_and_valid",
                f"Train/val overlap in {split_path} (count={len(overlap)})",
            )
        )
        return results

    if train_set or val_set:
        max_idx = max(train_set.union(val_set))
        min_idx = min(train_set.union(val_set))
        if min_idx < 0 or max_idx >= n_rows:
            results.append(
                fail_result(
                    "split_files_exist_and_valid",
                    f"Index out of bounds in {split_path} (n_rows={n_rows})",
                )
            )
            return results

    results.append(
        pass_result(
            "split_files_exist_and_valid",
            f"Validated split indices for {split_path}",
        )
    )
    return results


def _check_splits(ctx: VerifyContext) -> List[CheckResult]:
    split_specs = [
        (
            f"reports/splits/subtask1_unseen_user_seed{ctx.seed}.json",
            "data/raw/train_subtask1.csv",
        ),
        (
            f"reports/splits/subtask2a_unseen_user_seed{ctx.seed}.json",
            "data/raw/train_subtask2a.csv",
        ),
        (
            f"reports/splits/subtask2b_unseen_user_seed{ctx.seed}.json",
            "data/raw/train_subtask2b.csv",
        ),
    ]
    results: List[CheckResult] = []
    for split_rel, data_rel in split_specs:
        split_path = ctx.repo_root / split_rel
        data_path = ctx.repo_root / data_rel
        if not split_path.exists():
            results.append(
                fail_result(
                    "split_files_exist_and_valid",
                    f"Missing split file: {split_path}",
                )
            )
            continue
        try:
            payload = json.loads(split_path.read_text())
        except Exception as exc:
            results.append(
                fail_result(
                    "split_files_exist_and_valid",
                    f"Failed to parse split JSON {split_path}: {exc}",
                )
            )
            continue

        try:
            train_idx, val_idx = _load_split_indices(payload, split_path)
        except Exception as exc:
            results.append(
                fail_result(
                    "split_files_exist_and_valid",
                    f"Unsupported split schema in {split_path}: {exc}",
                )
            )
            continue

        try:
            n_rows = len(pd.read_csv(data_path))
        except Exception as exc:
            results.append(
                fail_result(
                    "split_files_exist_and_valid",
                    f"Failed to load dataset for split validation {data_path}: {exc}",
                )
            )
            continue

        results.extend(_validate_indices(train_idx, val_idx, n_rows, split_path))
    return results


def _check_reports_writable(ctx: VerifyContext) -> CheckResult:
    reports_dir = ctx.repo_root / "reports"
    if not reports_dir.exists():
        return fail_result("reports_writable", f"Missing reports/ directory: {reports_dir}")

    tmp_path = reports_dir / ".verify_tmp"
    try:
        tmp_path.write_text("ok", encoding="utf-8")
        tmp_path.unlink()
        return pass_result("reports_writable", f"reports/ is writable: {reports_dir}")
    except Exception as exc:
        return fail_result(
            "reports_writable",
            f"reports/ not writable: {reports_dir} ({exc})",
        )


def _check_reports_dirs(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    for rel in ["reports/preds", "reports/runs"]:
        path = ctx.repo_root / rel
        if path.exists():
            results.append(pass_result("reports_dirs_exist", f"Found {path}"))
        else:
            results.append(
                fail_result(
                    "reports_dirs_exist",
                    f"Missing directory: {path}",
                    hint="Create required reports directories for run-id artifacts.",
                )
            )
    return results


def _check_eval_records_schema(ctx: VerifyContext) -> CheckResult:
    path = ctx.repo_root / "reports" / "eval_records.csv"
    if not path.exists():
        return pass_result(
            "eval_records_schema_if_present",
            f"SKIP: eval_records.csv not found; initialize with: "
            f"python -m src.eval.phase0_eval --seed {ctx.seed}",
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except Exception as exc:
        return fail_result(
            "eval_records_schema_if_present",
            f"Failed to read header from {path}: {exc}",
        )

    required = {
        "run_id",
        "git_commit",
        "seed",
        "timestamp",
        "config_hash",
        "task",
        "regime",
        "slice",
        "model_tag",
        "primary_score",
    }
    missing = sorted(required - set(header))
    if missing:
        return fail_result(
            "eval_records_schema_if_present",
            f"Missing columns in {path}: {missing}",
        )
    return pass_result(
        "eval_records_schema_if_present",
        f"eval_records.csv schema OK: {path}",
    )
