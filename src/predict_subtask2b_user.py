from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from src.data_loader import load_all_data
from src.eval.splits import load_frozen_split
from src.models.subtask2b_user import load_checkpoint
from src.subtask2b_features import (
    build_subtask2b_user_features,
    load_subtask2b_embeddings_npz,
    apply_norm_stats,
)
from src.utils.provenance import merge_run_metadata, artifact_ref, get_git_snapshot, get_env_snapshot
from src.utils.diagnostics import summarize_pred_df
from src.utils.run_id import validate_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Subtask 2B Phase-D user model.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["val", "forecast"], default="val")
    parser.add_argument("--split_path", default=None)
    parser.add_argument(
        "--emb_path",
        default="data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz",
    )
    parser.add_argument(
        "--marker_path",
        default="data/raw/test/subtask2b_forecasting_user_marker.csv",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--pooling", default=None)
    args = parser.parse_args()

    validate_run_id(args.run_id)
    repo_root = Path(".").resolve()
    run_dir = repo_root / "models" / "subtask2b_user" / "runs" / args.run_id
    model, config, norm_stats = load_checkpoint(run_dir)
    if args.pooling and args.pooling != config.get("pooling"):
        print(
            f"WARNING: --pooling={args.pooling} differs from checkpoint "
            f"({config.get('pooling')}); using checkpoint."
        )
    pooling = config.get("pooling", "mean")
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2b",
            "stage": "predict",
            "seed": args.seed,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "inputs": {
                "embeddings": artifact_ref(Path(args.emb_path), repo_root),
                "model": artifact_ref(run_dir / "model.pt", repo_root),
            },
            "git": get_git_snapshot(repo_root),
            "env": get_env_snapshot(),
        },
    )

    data = load_all_data()
    df_text_raw = data["subtask2b"]
    df_user_raw = data["subtask2b_user"]

    emb_map_df, emb_arr = load_subtask2b_embeddings_npz(args.emb_path)
    embeddings = (emb_map_df, emb_arr)

    if args.mode == "val":
        split_path = Path(
            args.split_path
            or f"reports/splits/subtask2b_user_disposition_change_unseen_user_seed{args.seed}.json"
        )
        if not split_path.is_absolute():
            split_path = repo_root / split_path
        if not split_path.exists():
            raise SystemExit(f"Split file not found: {split_path}")
        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={"inputs": {"split": artifact_ref(split_path, repo_root)}},
        )
        train_idx, val_idx = load_frozen_split(split_path, df_user_raw)
        val_users_df = df_user_raw.iloc[val_idx].copy()
        X_emb, X_num, users, meta = build_subtask2b_user_features(
            val_users_df, df_text_raw, embeddings, pooling=pooling
        )
        X_num = apply_norm_stats(X_num, norm_stats)
        X = np.concatenate([X_emb, X_num], axis=1).astype(np.float32)
        with torch.no_grad():
            preds = model(torch.from_numpy(X)).numpy()

        label_map = df_user_raw.set_index("user_id")[
            ["disposition_change_valence", "disposition_change_arousal"]
        ]
        y_true = np.stack([label_map.loc[u].to_numpy(dtype=float) for u in users], axis=0)

        out_df = pd.DataFrame(
            {
                "run_id": args.run_id,
                "seed": int(args.seed),
                "user_id": users,
                "disposition_change_valence_true": y_true[:, 0],
                "disposition_change_arousal_true": y_true[:, 1],
                "disposition_change_valence_pred": preds[:, 0],
                "disposition_change_arousal_pred": preds[:, 1],
                "n_group1": meta["n_group1"].to_numpy(),
                "n_total": meta["n_total"].to_numpy(),
                "cut": meta["cut"].to_numpy(),
            }
        )
        if out_df["user_id"].duplicated().any():
            raise SystemExit("Preds must contain exactly one row per user.")

        out_path = (
            Path(args.output_path)
            if args.output_path
            else repo_root / "reports" / "preds" / f"subtask2b_val_user_preds__{args.run_id}.parquet"
        )
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "artifacts": {"preds_val_user": artifact_ref(out_path, repo_root)},
                "counts": {"n_val_users_pred": int(len(out_df))},
                "config": {
                    "predict": {
                        "subtask2b": {
                            "pooling": pooling,
                            "pred_kind": "val",
                            "pred_path": str(out_path),
                        }
                    }
                },
                "diagnostics": {
                    "predict": {
                        "subtask2b": {
                            "val": summarize_pred_df(
                                out_df,
                                pred_cols={
                                    "valence": "disposition_change_valence_pred",
                                    "arousal": "disposition_change_arousal_pred",
                                },
                                true_cols={
                                    "valence": "disposition_change_valence_true",
                                    "arousal": "disposition_change_arousal_true",
                                },
                                bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                            )
                        }
                    }
                },
            },
        )
        print(f"Wrote val preds to: {out_path}")
        return

    marker_path = Path(args.marker_path)
    if not marker_path.is_absolute():
        marker_path = repo_root / marker_path
    if not marker_path.exists():
        raise SystemExit(f"Marker file not found: {marker_path}")

    marker_df = pd.read_csv(marker_path)

    required_cols = {"user_id", "text_id", "group", "is_forecasting_user"}
    missing = sorted(list(required_cols - set(marker_df.columns)))
    if missing:
        raise SystemExit(
            f"Forecast marker missing required columns: {missing}. "
            f"Expected at least: {sorted(list(required_cols))}"
        )

    is_fc = marker_df["is_forecasting_user"]
    if is_fc.dtype != bool:
        is_fc = is_fc.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
    marker_df = marker_df[is_fc].copy()
    if len(marker_df) == 0:
        raise SystemExit("No forecasting rows found in marker after filtering is_forecasting_user==True.")

    try:
        marker_df["group"] = marker_df["group"].astype(int)
    except Exception as e:
        raise SystemExit(f"Could not parse 'group' as int in marker file: {e}")

    valid_groups = set(pd.Series(marker_df["group"]).dropna().unique().tolist())
    if not valid_groups.issubset({1, 2}):
        raise SystemExit(f"Invalid group values in marker file: {sorted(valid_groups)} (expected only 1/2)")

    marker_df = marker_df.copy()
    marker_df["half_group"] = marker_df["group"]

    forecast_users_all = set(marker_df["user_id"].unique().tolist())

    forecast_users_g1 = set(marker_df.loc[marker_df["half_group"] == 1, "user_id"].unique().tolist())
    missing_g1 = sorted(list(forecast_users_all - forecast_users_g1))
    if missing_g1:
        raise SystemExit(
            "Some forecasting users have no group==1 rows in marker file (cannot build leakage-safe features). "
            f"Missing count={len(missing_g1)} sample={missing_g1[:10]}"
        )

    forecast_users_df = pd.DataFrame({"user_id": sorted(list(forecast_users_all))})

    text_source = marker_df

    X_emb, X_num, users, meta = build_subtask2b_user_features(
        forecast_users_df,
        text_source,
        embeddings,
        pooling=pooling,
        enforce_group1_only=True,
    )

    users_set = set(users)
    expected_set = set(forecast_users_df["user_id"].tolist())
    missing_users = sorted(list(expected_set - users_set))
    extra_users = sorted(list(users_set - expected_set))
    if missing_users or extra_users:
        raise SystemExit(
            "Forecast user mismatch after feature building. "
            f"expected={len(expected_set)} built={len(users_set)} "
            f"missing={len(missing_users)} extra={len(extra_users)} "
            f"missing_sample={missing_users[:10]} extra_sample={extra_users[:10]}"
        )
    X_num = apply_norm_stats(X_num, norm_stats)
    X = np.concatenate([X_emb, X_num], axis=1).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()

    out_df = pd.DataFrame(
        {
            "run_id": args.run_id,
            "seed": int(args.seed),
            "user_id": users,
            "disposition_change_valence_pred": preds[:, 0],
            "disposition_change_arousal_pred": preds[:, 1],
            "n_group1": meta["n_group1"].to_numpy(),
            "n_total": meta["n_total"].to_numpy(),
            "cut": meta["cut"].to_numpy(),
        }
    )
    if out_df["user_id"].duplicated().any():
        raise SystemExit("Forecast preds must contain exactly one row per user.")

    out_path = (
        Path(args.output_path)
        if args.output_path
        else repo_root / "reports" / "preds" / f"subtask2b_forecast_user_preds__{args.run_id}.parquet"
    )
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "artifacts": {"preds_forecast_user": artifact_ref(out_path, repo_root)},
            "counts": {"n_forecast_users_pred": int(len(out_df))},
            "config": {
                "predict": {
                    "subtask2b": {
                        "pooling": pooling,
                        "pred_kind": "forecast",
                        "pred_path": str(out_path),
                    }
                }
            },
            "diagnostics": {
                "predict": {
                    "subtask2b": {
                        "forecast": summarize_pred_df(
                            out_df,
                            pred_cols={
                                "valence": "disposition_change_valence_pred",
                                "arousal": "disposition_change_arousal_pred",
                            },
                            true_cols=None,
                            bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                        )
                    }
                }
            },
        },
    )
    print(f"Wrote forecast preds to: {out_path}")


if __name__ == "__main__":
    main()
