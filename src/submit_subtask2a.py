from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.provenance import merge_run_metadata, artifact_ref, get_git_snapshot, get_env_snapshot
from src.utils.diagnostics import summarize_pred_df, apply_clip_to_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Subtask 2A submission CSV.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--forecast_preds_path",
        default=None,
        help="Defaults to reports/preds/subtask2a_forecast_user_preds__{run_id}.parquet",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Defaults to reports/submissions/subtask2a_submission__{run_id}.csv",
    )
    parser.add_argument("--marker_path", default=None)
    parser.add_argument("--clip_to_theoretical_bounds", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    preds_path = (
        Path(args.forecast_preds_path)
        if args.forecast_preds_path
        else repo_root / "reports" / "preds" / f"subtask2a_forecast_user_preds__{args.run_id}.parquet"
    )
    if not preds_path.is_absolute():
        preds_path = repo_root / preds_path
    if not preds_path.exists():
        raise SystemExit(f"Forecast preds parquet not found: {preds_path}")

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "stage": "submit",
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "inputs": {"preds_forecast_user": artifact_ref(preds_path, repo_root)},
            "git": get_git_snapshot(repo_root),
            "env": get_env_snapshot(),
        },
    )

    df = pd.read_parquet(preds_path)
    required = {"user_id", "delta_valence_pred", "delta_arousal_pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Forecast preds missing columns: {sorted(missing)}")
    if df["user_id"].duplicated().any():
        raise SystemExit("Submission must contain exactly one row per user.")
    if df[["delta_valence_pred", "delta_arousal_pred"]].isna().any().any():
        raise SystemExit("Submission contains missing prediction values.")
    if not np.isfinite(df[["delta_valence_pred", "delta_arousal_pred"]].to_numpy()).all():
        raise SystemExit("Submission contains non-finite prediction values.")

    if args.marker_path:
        marker_path = Path(args.marker_path)
        if not marker_path.is_absolute():
            marker_path = repo_root / marker_path
        marker_df = pd.read_csv(marker_path)
        if "user_id" in marker_df.columns:
            expected_users = marker_df["user_id"].nunique()
            if len(df) != expected_users:
                raise SystemExit(
                    f"Row count mismatch: preds={len(df)} vs marker users={expected_users}"
                )

    out_path = (
        Path(args.out_path)
        if args.out_path
        else repo_root
        / "reports"
        / "submissions"
        / f"subtask2a_submission__{args.run_id}.csv"
    )
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diag_pre = summarize_pred_df(
        df,
        pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
        true_cols=None,
        bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
    )
    clip_report = None
    diag_post = None
    df_out = df
    if args.clip_to_theoretical_bounds:
        df_out, clip_report = apply_clip_to_bounds(
            df,
            pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
            bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
        )
        diag_post = summarize_pred_df(
            df_out,
            pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
            true_cols=None,
            bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
        )

    out_df = pd.DataFrame(
        {
            "user_id": df_out["user_id"],
            "pred_state_change_valence": df_out["delta_valence_pred"],
            "pred_state_change_arousal": df_out["delta_arousal_pred"],
        }
    )
    out_df.to_csv(out_path, index=False)
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "artifacts": {"submission_csv": artifact_ref(out_path, repo_root)},
            "counts": {"n_submission_rows": int(len(out_df))},
            "diagnostics": {
                "submission": {
                    "subtask2a": {
                        "forecast": {
                            "pre": diag_pre,
                            "post": diag_post,
                            "clip_enabled": bool(args.clip_to_theoretical_bounds),
                            "clip_report": clip_report,
                        }
                    }
                }
            },
            "config": {
                "submission": {
                    "subtask2a": {
                        "clip_to_theoretical_bounds": bool(args.clip_to_theoretical_bounds),
                        "out_path": str(out_path),
                    }
                }
            },
        },
    )
    print(f"Wrote submission to: {out_path}")


if __name__ == "__main__":
    main()
