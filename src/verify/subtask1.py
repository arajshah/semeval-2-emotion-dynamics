from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.eval.analysis_tools import compute_subtask1_correlations
from src.verify.shared import CheckResult, VerifyContext, pass_result, fail_result, warn_result


def _run_help(module: str) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.returncode == 0, proc.stdout
    except Exception as exc:
        return False, str(exc)


def _help_has_required_flags(help_text: str, flags: List[str]) -> List[str]:
    missing = []
    for flag in flags:
        if flag not in help_text:
            missing.append(flag)
    return missing


def _split_required_from_help(help_text: str) -> bool:
    if "--split_path" not in help_text:
        return False
    lines = [line.strip() for line in help_text.splitlines() if "--split_path" in line]
    for line in lines:
        if "required" in line.lower():
            return True
        if "--split_path" in line and "[" not in line:
            return True
    return False


def _load_split_val_indices(split_path: Path) -> List[int]:
    payload = json.loads(split_path.read_text())
    for key in ["val_idx", "val_indices", "val"]:
        if key in payload:
            return [int(i) for i in payload[key]]
    raise ValueError(f"Unsupported split schema in {split_path}")


def _prediction_fix_hint(seed: int) -> str:
    return (
        "python -m src.predict_subtask1_transformer "
        f"--split_path reports/splits/subtask1_unseen_user_seed{seed}.json "
        "--ckpt_dir models/subtask1_transformer/best"
    )


def run_checks(ctx: VerifyContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    repo_root = ctx.repo_root

    ok_train, train_help = _run_help("src.train_subtask1_transformer")
    if not ok_train:
        results.append(
            fail_result(
                "subtask1_train_cli_contract",
                f"Failed to run train help: {train_help}",
                hint="Ensure you can run python -m src.train_subtask1_transformer --help from repo root.",
            )
        )
    else:
        required_flags = [
            "--split_path",
            "--head_type",
            "--amp",
            "--quick",
            "--resume_from_checkpoint",
        ]
        missing = _help_has_required_flags(train_help, required_flags)
        if missing:
            results.append(
                fail_result(
                    "subtask1_train_cli_contract",
                    f"Missing flags in train help: {missing}",
                    hint="Ensure CLI exposes required flags.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask1_train_cli_contract",
                    "Train CLI help contains required flags.",
                )
            )

    ok_pred, pred_help = _run_help("src.predict_subtask1_transformer")
    if not ok_pred:
        results.append(
            fail_result(
                "subtask1_predict_cli_contract",
                f"Failed to run predict help: {pred_help}",
                hint="Ensure you can run python -m src.predict_subtask1_transformer --help from repo root.",
            )
        )
    else:
        required_flags = ["--split_path"]
        missing = _help_has_required_flags(pred_help, required_flags)
        batch_ok = ("--batch_size" in pred_help) or ("--batch-size" in pred_help)
        if missing or not batch_ok:
            missing_items = list(missing)
            if not batch_ok:
                missing_items.append("--batch_size")
            results.append(
                fail_result(
                    "subtask1_predict_cli_contract",
                    f"Missing flags in predict help: {missing_items}",
                    hint="Ensure CLI exposes required flags.",
                )
            )
        else:
            results.append(
                pass_result(
                    "subtask1_predict_cli_contract",
                    "Predict CLI help contains required flags.",
                )
            )

    if ok_train and ok_pred:
        train_required = _split_required_from_help(train_help)
        pred_required = _split_required_from_help(pred_help)
        if train_required and pred_required:
            results.append(
                pass_result(
                    "subtask1_split_path_required",
                    "--split_path appears required in train/predict help.",
                )
            )
        else:
            results.append(
                warn_result(
                    "subtask1_split_path_required",
                    "Could not confirm --split_path is required from help output.",
                    hint="argparse may not mark required; ensure scripts fail fast when split_path missing.",
                )
            )

    if ctx.run_id is None:
        results.append(
            warn_result(
                "subtask1_val_preds_schema",
                "SKIP: run_id not provided; cannot validate run-id’d val preds under reports/preds/.",
                hint="Run verify with --run_id <RUN_ID> (expects reports/preds/subtask1_val_preds__<RUN_ID>.parquet).",
            )
        )
    else:
        preds_path = (
            repo_root
            / "reports"
            / "preds"
            / f"subtask1_val_preds__{ctx.run_id}.parquet"
        )
        if not preds_path.exists():
            results.append(
                warn_result(
                    "subtask1_val_preds_schema",
                    f"SKIP: {preds_path} not found.",
                    hint="Run predict for this run_id to generate val preds under reports/preds/.",
                )
            )
        else:
            try:
                preds_df = pd.read_parquet(preds_path)
                required_cols = [
                    "idx",
                    "user_id",
                    "is_words",
                    "valence_true",
                    "arousal_true",
                    "valence_pred",
                    "arousal_pred",
                ]
                missing = [c for c in required_cols if c not in preds_df.columns]
                if missing:
                    results.append(
                        fail_result(
                            "subtask1_val_preds_schema",
                            f"Missing columns in {preds_path}: {missing}",
                            hint="Regenerate predictions with predict script.",
                        )
                    )
                elif len(preds_df) == 0:
                    results.append(
                        fail_result(
                            "subtask1_val_preds_schema",
                            f"Predictions file is empty: {preds_path}",
                            hint="Regenerate predictions with predict script.",
                        )
                    )
                else:
                    results.append(
                        pass_result(
                            "subtask1_val_preds_schema",
                            f"Val preds schema OK: {preds_path}",
                        )
                    )
            except Exception as exc:
                results.append(
                    fail_result(
                        "subtask1_val_preds_schema",
                        f"Failed to read {preds_path}: {exc}",
                        hint="Regenerate predictions with predict script.",
                    )
                )

    if ctx.run_id is None:
        results.append(
            warn_result(
                "subtask1_val_user_agg_schema",
                "SKIP: run_id not provided; cannot validate run-id’d user aggregates under reports/preds/.",
                hint="Run verify with --run_id <RUN_ID> (expects reports/preds/subtask1_val_user_agg__<RUN_ID>.parquet).",
            )
        )
    else:
        user_agg_path = (
            repo_root
            / "reports"
            / "preds"
            / f"subtask1_val_user_agg__{ctx.run_id}.parquet"
        )
        if not user_agg_path.exists():
            results.append(
                warn_result(
                    "subtask1_val_user_agg_schema",
                    f"SKIP: {user_agg_path} not found.",
                    hint="Run predict for this run_id to generate user aggregates under reports/preds/.",
                )
            )
        else:
            try:
                user_df = pd.read_parquet(user_agg_path)
                missing = []
                if "user_id" not in user_df.columns:
                    missing.append("user_id")
                count_cols = {"n_rows", "count"}
                if not any(col in user_df.columns for col in count_cols):
                    missing.append("n_rows|count")
                required_mean_cols = {
                    "valence_true_mean": ["valence_true_mean", "valence_true_avg"],
                    "arousal_true_mean": ["arousal_true_mean", "arousal_true_avg"],
                    "valence_pred_mean": ["valence_pred_mean", "valence_pred_avg"],
                    "arousal_pred_mean": ["arousal_pred_mean", "arousal_pred_avg"],
                }
                for key, aliases in required_mean_cols.items():
                    if not any(alias in user_df.columns for alias in aliases):
                        missing.append(key)

                if missing:
                    results.append(
                        fail_result(
                            "subtask1_val_user_agg_schema",
                            f"Missing columns in {user_agg_path}: {missing}",
                            hint="Regenerate aggregates with predict script.",
                        )
                    )
                elif len(user_df) == 0:
                    results.append(
                        fail_result(
                            "subtask1_val_user_agg_schema",
                            f"User aggregate file is empty: {user_agg_path}",
                            hint="Regenerate aggregates with predict script.",
                        )
                    )
                else:
                    results.append(
                        pass_result(
                            "subtask1_val_user_agg_schema",
                            f"User aggregate schema OK: {user_agg_path}",
                        )
                    )
            except Exception as exc:
                results.append(
                    fail_result(
                        "subtask1_val_user_agg_schema",
                        f"Failed to read {user_agg_path}: {exc}",
                        hint="Regenerate aggregates with predict script.",
                    )
                )


    metrics_path = repo_root / "models" / "subtask1_transformer" / "metrics.json"
    if not metrics_path.exists():
        results.append(
            warn_result(
                "subtask1_metrics_json_sanity",
                f"SKIP: {metrics_path} not found.",
                hint="Train the transformer to generate metrics.json.",
            )
        )
    else:
        try:
            payload = json.loads(metrics_path.read_text())
            required_keys = {"best_run_id"}
            split_keys = {"split_path", "best_split_path"}
            missing_keys = [k for k in required_keys if k not in payload]
            if missing_keys:
                results.append(
                    fail_result(
                        "subtask1_metrics_json_sanity",
                        f"Missing keys in {metrics_path}: {missing_keys}",
                        hint="Ensure training writes best_run_id into metrics.json.",
                    )
                )
            elif not any(k in payload for k in split_keys):
                results.append(
                    fail_result(
                        "subtask1_metrics_json_sanity",
                        f"Missing split_path in {metrics_path}",
                        hint="Ensure training writes split_path into metrics.json.",
                    )
                )
            else:
                results.append(
                    pass_result(
                        "subtask1_metrics_json_sanity",
                        f"metrics.json sanity OK: {metrics_path}",
                    )
                )
        except Exception as exc:
            results.append(
                fail_result(
                    "subtask1_metrics_json_sanity",
                    f"Failed to read {metrics_path}: {exc}",
                    hint="Ensure metrics.json is valid JSON.",
                )
            )

    if ctx.mode == "strict":
        if ctx.run_id is None:
            results.append(
                warn_result(
                    "run_id_specific_checks",
                    "SKIP: run_id_specific_checks — provide --run_id to validate artifacts for a specific run.",
                )
            )
            return results

        run_id = ctx.run_id
        preds_path = repo_root / "reports" / "preds" / f"subtask1_val_preds__{run_id}.parquet"
        user_agg_path = repo_root / "reports" / "preds" / f"subtask1_val_user_agg__{run_id}.parquet"
        split_path = repo_root / "reports" / "splits" / f"subtask1_unseen_user_seed{ctx.seed}.json"
        raw_path = repo_root / "data" / "raw" / "train_subtask1.csv"
        runs_path = repo_root / "reports" / "runs" / f"{run_id}.json"
        paths_checked = (
            f"paths_checked={preds_path}, {user_agg_path}, {split_path}, {raw_path}, {runs_path}"
        )

        if not preds_path.exists():
            results.append(
                fail_result(
                    "subtask1_val_preds_exists",
                    f"Missing val preds: {preds_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results
        results.append(pass_result("subtask1_val_preds_exists", f"Found {preds_path}"))

        if not user_agg_path.exists():
            results.append(
                fail_result(
                    "subtask1_val_user_agg_exists",
                    f"Missing val user agg: {user_agg_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results
        results.append(
            pass_result("subtask1_val_user_agg_exists", f"Found {user_agg_path}")
        )

        if not split_path.exists():
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Missing split file: {split_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        try:
            val_idx = _load_split_val_indices(split_path)
        except Exception as exc:
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Failed to parse split indices from {split_path}: {exc}. {paths_checked}",
                    hint="Ensure split JSON contains val_idx/val_indices/val.",
                )
            )
            return results

        try:
            preds_df = pd.read_parquet(preds_path)
        except Exception as exc:
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Failed to read preds parquet {preds_path}: {exc}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        if "idx" not in preds_df.columns:
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Missing idx column in {preds_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        pred_idx = preds_df["idx"].astype(int).tolist()
        if len(pred_idx) != len(set(pred_idx)):
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Duplicate idx values in {preds_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        split_set = set(val_idx)
        pred_set = set(pred_idx)
        missing = sorted(split_set - pred_set)
        extra = sorted(pred_set - split_set)
        if missing:
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Missing indices in preds (count={len(missing)}, examples={missing[:5]}). {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results
        if extra:
            results.append(
                fail_result(
                    "subtask1_split_alignment",
                    f"Extra indices in preds (count={len(extra)}, examples={extra[:5]}). {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        results.append(
            pass_result(
                "subtask1_split_alignment",
                "Prediction idx aligns exactly to frozen split.",
            )
        )

        required_cols = [
            "idx",
            "user_id",
            "is_words",
            "valence_true",
            "arousal_true",
            "valence_pred",
            "arousal_pred",
        ]
        missing_cols = [c for c in required_cols if c not in preds_df.columns]
        if missing_cols:
            results.append(
                fail_result(
                    "subtask1_preds_finite_and_range",
                    f"Missing columns in {preds_path}: {missing_cols}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results

        numeric = preds_df[["valence_pred", "arousal_pred"]]
        non_finite = ~np.isfinite(numeric.to_numpy())
        if non_finite.any():
            count = int(non_finite.sum())
            results.append(
                fail_result(
                    "subtask1_preds_finite_and_range",
                    f"Non-finite prediction values (count={count}) at {preds_path}. {paths_checked}",
                    hint=_prediction_fix_hint(ctx.seed),
                )
            )
            return results
        results.append(
            pass_result(
                "subtask1_preds_finite_and_range",
                "Prediction columns are finite.",
            )
        )

        if not runs_path.exists():
            results.append(
                fail_result(
                    "subtask1_run_provenance_ok",
                    f"Missing provenance JSON: {runs_path}. {paths_checked}",
                    hint="Ensure predictions write reports/runs/{run_id}.json.",
                )
            )
            return results
        try:
            payload = json.loads(runs_path.read_text())
        except Exception as exc:
            results.append(
                fail_result(
                    "subtask1_run_provenance_ok",
                    f"Failed to read {runs_path}: {exc}. {paths_checked}",
                    hint="Ensure provenance JSON is valid.",
                )
            )
            return results

        required_keys = {
            "run_id",
            "task",
            "task_tag",
            "seed",
            "regime",
            "split_path",
            "timestamp",
            "git_commit",
            "config_hash",
            "artifacts",
        }
        missing_keys = sorted(required_keys - set(payload.keys()))
        if missing_keys:
            results.append(
                fail_result(
                    "subtask1_run_provenance_ok",
                    f"Missing keys in {runs_path}: {missing_keys}. {paths_checked}",
                    hint="Ensure write_run_metadata stores required fields.",
                )
            )
            return results

        artifacts = payload.get("artifacts", {})
        needed_paths = {str(preds_path.relative_to(repo_root)), str(user_agg_path.relative_to(repo_root))}
        artifact_values = set(str(v) for v in artifacts.values())
        if not needed_paths.issubset(artifact_values):
            results.append(
                fail_result(
                    "subtask1_run_provenance_ok",
                    f"Artifacts missing preds paths in {runs_path}. {paths_checked}",
                    hint="Ensure provenance artifacts include prediction paths.",
                )
            )
            return results

        for path_str in artifact_values:
            artifact_path = Path(path_str)
            if not artifact_path.is_absolute():
                artifact_path = repo_root / artifact_path
            if not artifact_path.exists():
                results.append(
                    fail_result(
                        "subtask1_run_provenance_ok",
                        f"Artifact path missing on disk: {artifact_path}. {paths_checked}",
                        hint="Ensure artifacts listed in provenance exist.",
                    )
                )
                return results

        results.append(
            pass_result(
                "subtask1_run_provenance_ok",
                f"Provenance JSON OK: {runs_path}",
            )
        )

        no_ambiguity_paths = [
            repo_root / "src" / "eval" / "phase0_eval.py",
            repo_root / "src" / "predict_subtask1_transformer.py",
        ]
        try:
            phase0_text = no_ambiguity_paths[0].read_text(encoding="utf-8")
            predict_text = no_ambiguity_paths[1].read_text(encoding="utf-8")
        except Exception as exc:
            results.append(
                warn_result(
                    "subtask1_no_ambiguity_policy",
                    f"SKIP: failed to read CLI sources ({exc}). {paths_checked}",
                    hint="Ensure scripts expose explicit run_id/pred_path routing.",
                )
            )
            return results

        phase0_ok = ("--run_id" in phase0_text) or ("--pred_path" in phase0_text)
        predict_ok = ("--run_id" in predict_text) or ("infer_run_id_from_metrics" in predict_text)
        if phase0_ok and predict_ok:
            results.append(
                pass_result(
                    "subtask1_no_ambiguity_policy",
                    "Explicit run targeting detected in eval/predict CLIs.",
                )
            )
        else:
            results.append(
                warn_result(
                    "subtask1_no_ambiguity_policy",
                    "Could not confirm explicit run targeting in eval/predict CLIs.",
                    hint="Ensure eval and predict require --run_id or --pred_path.",
                )
            )

    return results
