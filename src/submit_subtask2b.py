from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.subtask2b_features import load_subtask2b_embeddings_npz, merge_embeddings
from src.utils.provenance import merge_run_metadata, artifact_ref, get_git_snapshot, get_env_snapshot
from src.utils.diagnostics import summarize_pred_df, apply_clip_to_bounds


def _build_features_for_users(
    df_with_emb: pd.DataFrame, user_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = df_with_emb.attrs.get("embeddings")
    if embeddings is None:
        raise ValueError("Embeddings array not attached to df_with_emb.attrs['embeddings'].")

    feature_rows = []
    kept_users = []
    for user_id in user_ids:
        user_df = df_with_emb[df_with_emb["user_id"] == user_id]
        group1_df = user_df[user_df["group"] == 1]
        if group1_df.empty:
            continue
        emb_indices = group1_df["emb_index"].to_numpy()
        emb_mat = embeddings[emb_indices]
        emb_mean = emb_mat.mean(axis=0)
        mean_valence = float(group1_df["valence"].mean())
        mean_arousal = float(group1_df["arousal"].mean())
        features = np.concatenate([emb_mean, np.array([mean_valence, mean_arousal])])
        feature_rows.append(features)
        kept_users.append(user_id)

    X = np.asarray(feature_rows, dtype=np.float32)
    users = np.asarray(kept_users)
    return X, users


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Subtask 2B submission CSV.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--pred_path",
        default=None,
        help="Forecast preds parquet (default: reports/preds/subtask2b_forecast_user_preds__{run_id}.parquet)",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to model joblib. Defaults to models/subtask2b_user/runs/{run_id}/model.joblib",
    )
    parser.add_argument(
        "--embeddings_path",
        default="data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    parser.add_argument(
        "--marker_path",
        default="data/raw/test/subtask2b_forecasting_user_marker.csv",
    )
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--clip_to_theoretical_bounds", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(".").resolve()

    pred_path = (
        Path(args.pred_path)
        if args.pred_path
        else repo_root / "reports" / "preds" / f"subtask2b_forecast_user_preds__{args.run_id}.parquet"
    )
    if not pred_path.is_absolute():
        pred_path = repo_root / pred_path

    marker_path = Path(args.marker_path)
    if not marker_path.is_absolute():
        marker_path = repo_root / marker_path
    if not marker_path.exists():
        raise SystemExit(f"Marker file not found: {marker_path}")

    df_marker = pd.read_csv(marker_path)

    df_marker["user_id"] = pd.to_numeric(df_marker["user_id"], errors="raise").astype(int)
    df_marker["text_id"] = pd.to_numeric(df_marker["text_id"], errors="raise").astype(int)

    if "is_forecasting_user" not in df_marker.columns:
        raise SystemExit("Marker file missing required column: is_forecasting_user")

    is_fc = df_marker["is_forecasting_user"]
    if is_fc.dtype != bool:
        is_fc = is_fc.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
    df_marker = df_marker[is_fc].copy()

    if df_marker.empty:
        raise SystemExit("No forecasting users found in marker file after filtering is_forecasting_user==True.")

    diag_pre = None
    diag_post = None
    clip_report = None
    if pred_path.exists():
        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "task": "subtask2b",
                "stage": "submit",
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "inputs": {"preds_forecast_user": artifact_ref(pred_path, repo_root)},
                "git": get_git_snapshot(repo_root),
                "env": get_env_snapshot(),
            },
        )
        df_preds = pd.read_parquet(pred_path)
        required = {"user_id", "disposition_change_valence_pred", "disposition_change_arousal_pred"}
        missing = required - set(df_preds.columns)
        if missing:
            raise SystemExit(f"Forecast preds missing columns: {sorted(missing)}")
        if df_preds["user_id"].duplicated().any():
            raise SystemExit("Forecast preds must contain exactly one row per user.")
        if df_preds[["disposition_change_valence_pred", "disposition_change_arousal_pred"]].isna().any().any():
            raise SystemExit("Forecast preds contain missing values.")
        if not np.isfinite(
            df_preds[["disposition_change_valence_pred", "disposition_change_arousal_pred"]].to_numpy()
        ).all():
            raise SystemExit("Forecast preds contain non-finite values.")
        diag_pre = summarize_pred_df(
            df_preds,
            pred_cols={
                "valence": "disposition_change_valence_pred",
                "arousal": "disposition_change_arousal_pred",
            },
            true_cols=None,
            bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
        )
        df_out = df_preds
        if args.clip_to_theoretical_bounds:
            df_out, clip_report = apply_clip_to_bounds(
                df_preds,
                pred_cols={
                    "valence": "disposition_change_valence_pred",
                    "arousal": "disposition_change_arousal_pred",
                },
                bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
            )
            diag_post = summarize_pred_df(
                df_out,
                pred_cols={
                    "valence": "disposition_change_valence_pred",
                    "arousal": "disposition_change_arousal_pred",
                },
                true_cols=None,
                bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
            )

        out_df = pd.DataFrame(
            {
                "user_id": df_out["user_id"],
                "pred_dispo_change_valence": df_out["disposition_change_valence_pred"],
                "pred_dispo_change_arousal": df_out["disposition_change_arousal_pred"],
            }
        )

        marker_users = set(df_marker["user_id"].unique().tolist())
        pred_users = set(out_df["user_id"].unique().tolist())
        missing = sorted(list(marker_users - pred_users))
        extra = sorted(list(pred_users - marker_users))
        if missing or extra:
            raise SystemExit(
                "Forecast user mismatch between marker and preds. "
                f"marker={len(marker_users)} preds={len(pred_users)} "
                f"missing={len(missing)} extra={len(extra)} "
                f"missing_sample={missing[:10]} extra_sample={extra[:10]}"
            )
    else:
        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "task": "subtask2b",
                "stage": "submit",
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "inputs": {"marker_path": artifact_ref(marker_path, repo_root)},
                "git": get_git_snapshot(repo_root),
                "env": get_env_snapshot(),
            },
        )
        model_path = Path(args.model_path) if args.model_path else (
            repo_root / "models" / "subtask2b_user" / "runs" / args.run_id / "model.joblib"
        )
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

        required = {
            "user_id",
            "text_id",
            "text",
            "group",
            "valence",
            "arousal",
            "is_forecasting_user",
        }
        missing = required - set(df_marker.columns)
        if missing:
            raise SystemExit(f"Marker missing required columns: {sorted(missing)}")

        df_marker = df_marker[df_marker["is_forecasting_user"] == True].copy()
        if df_marker.empty:
            raise SystemExit("No forecasting users found in marker file.")

        emb_map_df, embeddings = load_subtask2b_embeddings_npz(args.embeddings_path)
        merged = merge_embeddings(df_marker, emb_map_df)
        merged.attrs["embeddings"] = embeddings

        user_ids = merged["user_id"].drop_duplicates(keep="first").to_numpy()
        X, users = _build_features_for_users(merged, user_ids)
        if len(users) != len(user_ids):
            raise SystemExit("Some forecasting users have no group==1 rows.")

        model = joblib.load(model_path)
        preds = model.predict(X)

        out_df = pd.DataFrame(
            {
                "user_id": users,
                "pred_dispo_change_valence": preds[:, 0],
                "pred_dispo_change_arousal": preds[:, 1],
            }
        )

    forecast_users = df_marker["user_id"].nunique() if "user_id" in df_marker.columns else None
    if forecast_users is not None and len(out_df) != forecast_users:
        raise SystemExit(
            f"Row count mismatch: submission rows={len(out_df)} vs marker users={forecast_users}"
        )

    out_path = (
        Path(args.out_path)
        if args.out_path
        else repo_root / "reports" / "submissions" / f"subtask2b_submission__{args.run_id}.csv"
    )
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Submission already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = out_df.sort_values("user_id").reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "artifacts": {"submission_csv": artifact_ref(out_path, repo_root)},
            "counts": {"n_submission_rows": int(len(out_df))},
            "diagnostics": {
                "submission": {
                    "subtask2b": {
                        "forecast": {
                            "pre": diag_pre if pred_path.exists() else None,
                            "post": diag_post if pred_path.exists() else None,
                            "clip_enabled": bool(args.clip_to_theoretical_bounds),
                            "clip_report": clip_report if pred_path.exists() else None,
                        }
                    }
                }
            },
            "config": {
                "submission": {
                    "subtask2b": {
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
