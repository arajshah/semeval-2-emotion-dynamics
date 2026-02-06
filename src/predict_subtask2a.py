from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loader import load_all_data
from src.eval.splits import load_frozen_split
from src.sequence_models.subtask2a_sequence_dataset import (
    build_subtask2a_anchor_features,
    select_latest_eligible_anchors,
    select_forecast_anchors,
)
from src.sequence_models.baselines_subtask2a import (
    add_prev_delta_features,
    fit_linear_prev,
    predict_linear_prev,
)
from src.sequence_models.simple_sequence_model import SequenceStateRegressor
from src.utils.diagnostics import summarize_pred_df
from src.utils.provenance import merge_run_metadata, artifact_ref, get_git_snapshot, get_env_snapshot
from src.utils.run_id import validate_run_id


def _load_checkpoint(model_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint
    return {"model_state": checkpoint, "config": {}, "norm_stats": None}


def _infer_model_config(checkpoint: dict, args: argparse.Namespace) -> dict:
    config = checkpoint.get("config", {})
    return {
        "seq_len": args.seq_len or config.get("seq_len", 5),
        "k_state": args.k_state or config.get("k_state", 5),
        "hidden_dim": config.get("hidden_dim", args.hidden_dim),
        "num_layers": config.get("num_layers", args.num_layers),
        "dropout": config.get("dropout", args.dropout),
        "num_features": config.get("num_features"),
        "use_numeric_features": bool(config.get("use_numeric_features", False)),
        "use_residual_targets": bool(config.get("use_residual_targets", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Subtask 2A anchored preds.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--k_state", type=int, default=None)
    parser.add_argument("--ablate_no_history", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--forecast_marker_path", default=None)
    parser.add_argument("--forecast_cutoff_path", default=None)
    parser.add_argument("--write_forecast", type=int, default=0)
    parser.add_argument("--use_residual_preds", type=int, default=-1)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--write_baseline", type=int, default=0)
    args = parser.parse_args()

    validate_run_id(args.run_id)
    repo_root = Path(".").resolve()

    split_path = Path(
        args.split_path
        or f"reports/splits/subtask2a_unseen_user_seed{args.seed}.json"
    )
    if not split_path.is_absolute():
        split_path = repo_root / split_path

    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.is_absolute():
        embeddings_path = repo_root / embeddings_path

    write_baseline = bool(int(args.write_baseline))

    # Only resolve / require a checkpoint when NOT in baseline mode.
    model_path: Path | None = None
    if not write_baseline:
        model_path = (
            Path(args.model_path)
            if args.model_path
            else repo_root
            / "models"
            / "subtask2a_sequence"
            / "runs"
            / args.run_id
            / "model.pt"
        )
        if not model_path.exists():
            raise SystemExit(f"Model checkpoint not found: {model_path}")

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "stage": "predict",
            "seed": args.seed,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "inputs": {
                "split": artifact_ref(split_path, repo_root),
                "embeddings": artifact_ref(embeddings_path, repo_root),
                **({"model": artifact_ref(model_path, repo_root)} if model_path is not None else {}),
            },
            "git": get_git_snapshot(repo_root),
            "env": get_env_snapshot(),
        },
    )

    data = load_all_data()
    df_raw = data["subtask2a"]
    train_idx, val_idx = load_frozen_split(split_path, df_raw)
    val_idx = np.asarray(val_idx, dtype=int)

    val_df_raw = df_raw.iloc[val_idx].copy()
    val_df_raw["idx"] = val_idx
    anchors_df = select_latest_eligible_anchors(val_df_raw)
    if anchors_df.empty:
        raise SystemExit("No eligible val anchors found.")

    if write_baseline:
        df_feat = df_raw.copy().reset_index().rename(columns={"index": "idx"})
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], errors="raise")

        df_feat = add_prev_delta_features(df_feat)
        df_feat = df_feat.sort_values("idx", kind="stable").set_index("idx")

        train_feat = df_feat.loc[np.asarray(train_idx, dtype=int)]
        baseline_model = fit_linear_prev(train_feat)

        anchor_idxs = anchors_df["anchor_idx"].astype(int).drop_duplicates().to_numpy()

        # IMPORTANT: compute prev-delta features on the full VAL rows (per user history),
        # then pick the anchor rows.
        val_rows = df_feat.loc[np.asarray(val_idx, dtype=int)].reset_index()

        val_rows = predict_linear_prev(
            val_rows,
            baseline_model,
            fill_value=0.0,
            out_v_col="delta_valence_pred",
            out_a_col="delta_arousal_pred",
        )

        anchor_rows = val_rows[val_rows["idx"].isin(anchor_idxs)].copy()
        assert (
            anchor_rows["delta_valence_pred"].nunique() > 1
            or anchor_rows["delta_arousal_pred"].nunique() > 1
        ), "linear(prev) baseline degenerate: preds are (near) constant."

        out_df = pd.DataFrame(
            {
                "user_id": anchor_rows["user_id"].to_numpy(),
                "anchor_idx": anchor_rows["idx"].to_numpy(),
                "anchor_text_id": anchor_rows["text_id"].to_numpy(),
                "anchor_timestamp": anchor_rows["timestamp"].to_numpy(),
                "delta_valence_true": anchor_rows["state_change_valence"].to_numpy(dtype=float),
                "delta_arousal_true": anchor_rows["state_change_arousal"].to_numpy(dtype=float),
                "delta_valence_pred": anchor_rows["delta_valence_pred"].to_numpy(dtype=float),
                "delta_arousal_pred": anchor_rows["delta_arousal_pred"].to_numpy(dtype=float),
            }
        )
        if out_df["user_id"].duplicated().any():
            raise SystemExit("Anchored dev preds must contain exactly one row per user.")

        preds_dir = repo_root / "reports" / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

        dev_path = Path(args.out_path) if args.out_path else (
            preds_dir / f"subtask2a_val_user_preds__{args.run_id}.parquet"
        )
        if not dev_path.is_absolute():
            dev_path = repo_root / dev_path
        dev_path.parent.mkdir(parents=True, exist_ok=True)

        out_df.to_parquet(dev_path, index=False)
        print(f"Wrote BASELINE dev anchored preds to: {dev_path}")

        val_users_total = df_raw.iloc[val_idx]["user_id"].nunique()
        n_dropped = int(val_users_total - len(out_df))

        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "artifacts": {"preds_val_user": artifact_ref(dev_path, repo_root)},
                "counts": {
                    "n_val_users_pred": int(len(out_df)),
                    "n_val_users_dropped_no_eligible": n_dropped,
                },
                "config": {
                    "predict": {
                        "subtask2a": {
                            "pred_kind": "val",
                            "pred_path": str(dev_path),
                            "baseline": "linear_prev",
                        }
                    }
                },
                "diagnostics": {
                    "predict": {
                        "subtask2a": {
                            "val": summarize_pred_df(
                                out_df,
                                pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
                                true_cols={"valence": "delta_valence_true", "arousal": "delta_arousal_true"},
                                bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                            )
                        }
                    }
                },
            },
        )
        
        if args.write_forecast:
            if not args.forecast_cutoff_path and not args.forecast_marker_path:
                raise SystemExit(
                    "Provide --forecast_cutoff_path (preferred; test_subtask2.csv) or "
                    "--forecast_marker_path when --write_forecast 1."
                )

            cutoff_df = None
            if args.forecast_cutoff_path:
                cutoff_path = Path(args.forecast_cutoff_path)
                if not cutoff_path.is_absolute():
                    cutoff_path = repo_root / cutoff_path
                if not cutoff_path.exists():
                    raise SystemExit(f"Forecast cutoff file not found: {cutoff_path}")
                cutoff_df = pd.read_csv(cutoff_path)
                if "user_id" not in cutoff_df.columns or "timestamp_min" not in cutoff_df.columns:
                    raise SystemExit(
                        f"Cutoff file must have columns user_id,timestamp_min. Got: {list(cutoff_df.columns)}"
                    )

            marker_df = None
            if args.forecast_marker_path:
                marker_path = Path(args.forecast_marker_path)
                if not marker_path.is_absolute():
                    marker_path = repo_root / marker_path
                if not marker_path.exists():
                    raise SystemExit(f"Forecast marker file not found: {marker_path}")
                marker_df = pd.read_csv(marker_path)

            forecast_anchors = select_forecast_anchors(
                df_raw=df_raw,
                marker_df=(marker_df if marker_df is not None else pd.DataFrame(columns=["user_id"])),
                cutoff_df=cutoff_df,
            )
            if forecast_anchors.empty:
                raise SystemExit("No forecast anchors found (baseline).")

            forecast_anchor_idxs = (
                forecast_anchors["anchor_idx"].astype(int).drop_duplicates().to_numpy()
            )

            forecast_rows = df_feat.loc[forecast_anchor_idxs].reset_index()

            forecast_rows = predict_linear_prev(
                forecast_rows,
                baseline_model,
                fill_value=0.0,
                out_v_col="delta_valence_pred",
                out_a_col="delta_arousal_pred",
            )

            forecast_df = pd.DataFrame(
                {
                    "run_id": args.run_id,
                    "seed": int(args.seed),
                    "user_id": forecast_rows["user_id"].to_numpy(),
                    "anchor_idx": forecast_rows["idx"].to_numpy(),
                    "anchor_text_id": forecast_rows["text_id"].to_numpy(),
                    "anchor_timestamp": forecast_rows["timestamp"].to_numpy(),
                    "delta_valence_pred": forecast_rows["delta_valence_pred"].to_numpy(dtype=float),
                    "delta_arousal_pred": forecast_rows["delta_arousal_pred"].to_numpy(dtype=float),
                }
            )
            if forecast_df["user_id"].duplicated().any():
                raise SystemExit("Forecast preds must contain exactly one row per user (baseline).")
            if forecast_df[["delta_valence_pred", "delta_arousal_pred"]].isna().any().any():
                raise SystemExit("Baseline forecast contains missing prediction values.")
            if not np.isfinite(forecast_df[["delta_valence_pred", "delta_arousal_pred"]].to_numpy()).all():
                raise SystemExit("Baseline forecast contains non-finite prediction values.")

            forecast_path = preds_dir / f"subtask2a_forecast_user_preds__{args.run_id}.parquet"
            forecast_df.to_parquet(forecast_path, index=False)
            print(f"Wrote BASELINE forecast preds to: {forecast_path}")

            merge_run_metadata(
                repo_root=repo_root,
                run_id=args.run_id,
                updates={
                    "artifacts": {"preds_forecast_user": artifact_ref(forecast_path, repo_root)},
                    "counts": {"n_forecast_users_pred": int(len(forecast_df))},
                    "config": {
                        "predict": {
                            "subtask2a": {
                                "pred_kind": "forecast",
                                "pred_path": str(forecast_path),
                                "baseline": "linear_prev",
                            }
                        }
                    },
                    "diagnostics": {
                        "predict": {
                            "subtask2a": {
                                "forecast": summarize_pred_df(
                                    forecast_df,
                                    pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
                                    true_cols=None,
                                    bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                                )
                            }
                        }
                    },
                },
            )

        return

    assert model_path is not None, "Internal error: model_path must be set when not in baseline mode."
    checkpoint = _load_checkpoint(model_path)
    norm_stats = checkpoint.get("norm_stats")
    if norm_stats is None:
        raise SystemExit("Checkpoint missing norm_stats required for predictions.")

    config = _infer_model_config(checkpoint, args)
    seq_len = int(config["seq_len"])
    k_state = int(config["k_state"])
    use_numeric_features = bool(config.get("use_numeric_features", False))
    use_residual_preds = (
        bool(args.use_residual_preds)
        if int(args.use_residual_preds) >= 0
        else bool(config.get("use_residual_targets", False))
    )

    if args.seq_len is not None and int(args.seq_len) != seq_len:
        print(f"WARNING: --seq_len={args.seq_len} differs from checkpoint ({seq_len}); using checkpoint.")
    if args.k_state is not None and int(args.k_state) != k_state:
        print(f"WARNING: --k_state={args.k_state} differs from checkpoint ({k_state}); using checkpoint.")

    ckpt_ablate = bool(checkpoint.get("config", {}).get("ablate_no_history", False))
    if args.ablate_no_history is not None and bool(args.ablate_no_history) != ckpt_ablate:
        print(
            f"WARNING: --ablate_no_history={args.ablate_no_history} differs from checkpoint ({ckpt_ablate}); "
            "using checkpoint setting."
        )
    ablate_no_history = ckpt_ablate

    val_bundle = build_subtask2a_anchor_features(
        df_raw=df_raw,
        anchors_df=anchors_df[["anchor_idx"]],
        embeddings_path=embeddings_path,
        seq_len=seq_len,
        k_state=k_state,
        norm_stats=norm_stats,
        ablate_no_history=ablate_no_history,
    )

    meta = val_bundle["meta"].copy().rename(columns={"text_id": "anchor_text_id", "timestamp": "anchor_timestamp"})
    y_true = meta[["state_change_valence", "state_change_arousal"]].to_numpy(dtype=float)

    num_features = (
        int(config["num_features"])
        if config.get("num_features") is not None
        else int(val_bundle["X_num"].shape[1])
    )
    if int(val_bundle["X_num"].shape[1]) != num_features:
        raise SystemExit(
            f"num_features mismatch: bundle has {val_bundle['X_num'].shape[1]}, checkpoint has {num_features}."
        )

    model = SequenceStateRegressor(
        embedding_dim=val_bundle["X_seq"].shape[2],
        num_features=num_features,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_numeric_features=use_numeric_features,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preds_list = []
    with torch.no_grad():
        for start in range(0, len(val_bundle["X_seq"]), args.batch_size):
            end = start + args.batch_size
            x_seq = torch.from_numpy(val_bundle["X_seq"][start:end])
            lengths = torch.from_numpy(val_bundle["lengths"][start:end])
            x_num = torch.from_numpy(val_bundle["X_num"][start:end])
            outputs = model(x_seq, lengths, x_num)
            preds_list.append(outputs.numpy())

    preds = np.concatenate(preds_list, axis=0)

    if use_residual_preds:
        if "base_pred" not in val_bundle:
            raise SystemExit("Residual preds requested but base_pred not available in val bundle.")
        base_pred = val_bundle["base_pred"]
        if base_pred.shape != preds.shape:
            raise SystemExit(f"base_pred shape mismatch: expected {preds.shape}, got {base_pred.shape}.")
        preds = preds + base_pred

    out_df = pd.DataFrame(
        {
            "user_id": meta["user_id"].to_numpy(),
            "anchor_idx": meta["idx"].to_numpy(),
            "anchor_text_id": meta["anchor_text_id"].to_numpy(),
            "anchor_timestamp": meta["anchor_timestamp"].to_numpy(),
            "delta_valence_true": y_true[:, 0],
            "delta_arousal_true": y_true[:, 1],
            "delta_valence_pred": preds[:, 0],
            "delta_arousal_pred": preds[:, 1],
        }
    )
    if out_df["user_id"].duplicated().any():
        raise SystemExit("Anchored dev preds must contain exactly one row per user.")

    preds_dir = repo_root / "reports" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    dev_path = Path(args.out_path) if args.out_path else (
        preds_dir / f"subtask2a_val_user_preds__{args.run_id}.parquet"
    )
    if not dev_path.is_absolute():
        dev_path = repo_root / dev_path

    out_df.to_parquet(dev_path, index=False)
    print(f"Wrote dev anchored preds to: {dev_path}")

    val_users_total = df_raw.iloc[val_idx]["user_id"].nunique()
    n_dropped = int(val_users_total - len(out_df))

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "artifacts": {"preds_val_user": artifact_ref(dev_path, repo_root)},
            "counts": {
                "n_val_users_pred": int(len(out_df)),
                "n_val_users_dropped_no_eligible": n_dropped,
            },
            "config": {
                "predict": {
                    "subtask2a": {
                        "seq_len": seq_len,
                        "ablate_no_history": ablate_no_history,
                        "pred_kind": "val",
                        "pred_path": str(dev_path),
                        "use_residual_preds": bool(use_residual_preds),
                        "use_numeric_features": bool(use_numeric_features),
                    }
                }
            },
            "diagnostics": {
                "predict": {
                    "subtask2a": {
                        "val": summarize_pred_df(
                            out_df,
                            pred_cols={"valence": "delta_valence_pred", "arousal": "delta_arousal_pred"},
                            true_cols={"valence": "delta_valence_true", "arousal": "delta_arousal_true"},
                            bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                        )
                    }
                }
            },
        },
    )

    # Forecast writing stays exactly as you already had it, using `model`, `preds_dir`, etc.
    # Keep your existing forecast block below this point unchanged.

    if args.write_forecast:
        if not args.forecast_cutoff_path and not args.forecast_marker_path:
            raise SystemExit(
                "Provide --forecast_cutoff_path (preferred; test_subtask2.csv) or "
                "--forecast_marker_path when --write_forecast 1."
            )

        cutoff_df = None
        if args.forecast_cutoff_path:
            cutoff_path = Path(args.forecast_cutoff_path)
            if not cutoff_path.is_absolute():
                cutoff_path = repo_root / cutoff_path
            if not cutoff_path.exists():
                raise SystemExit(f"Forecast cutoff file not found: {cutoff_path}")
            cutoff_df = pd.read_csv(cutoff_path)
            if "user_id" not in cutoff_df.columns or "timestamp_min" not in cutoff_df.columns:
                raise SystemExit(
                    f"Cutoff file must have columns user_id,timestamp_min. Got: {list(cutoff_df.columns)}"
                )

        marker_df = None
        if args.forecast_marker_path:
            marker_path = Path(args.forecast_marker_path)
            if not marker_path.is_absolute():
                marker_path = repo_root / marker_path
            if not marker_path.exists():
                raise SystemExit(f"Forecast marker file not found: {marker_path}")
            marker_df = pd.read_csv(marker_path)

        forecast_anchors = select_forecast_anchors(
            df_raw=df_raw,
            marker_df=(marker_df if marker_df is not None else pd.DataFrame(columns=["user_id"])),
            cutoff_df=cutoff_df,
        )

        forecast_bundle = build_subtask2a_anchor_features(
            df_raw=df_raw,
            anchors_df=forecast_anchors[["anchor_idx"]],
            embeddings_path=embeddings_path,
            seq_len=seq_len,
            k_state=k_state,
            norm_stats=norm_stats,
            ablate_no_history=ablate_no_history,
        )
        fmeta = forecast_bundle["meta"].rename(
            columns={"text_id": "anchor_text_id", "timestamp": "anchor_timestamp"}
        )

        preds_list = []
        with torch.no_grad():
            for start in range(0, len(forecast_bundle["X_seq"]), args.batch_size):
                end = start + args.batch_size
                x_seq = torch.from_numpy(forecast_bundle["X_seq"][start:end])
                lengths = torch.from_numpy(forecast_bundle["lengths"][start:end])
                x_num = torch.from_numpy(forecast_bundle["X_num"][start:end])
                outputs = model(x_seq, lengths, x_num)
                preds_list.append(outputs.numpy())
        f_preds = np.concatenate(preds_list, axis=0)
        if use_residual_preds:
            if "base_pred" not in forecast_bundle:
                raise SystemExit("Residual preds requested but base_pred not available in forecast bundle.")
            base_pred = forecast_bundle["base_pred"]
            if base_pred.shape != f_preds.shape:
                raise SystemExit(
                    f"base_pred shape mismatch: expected {f_preds.shape}, got {base_pred.shape}."
                )
            f_preds = f_preds + base_pred

        forecast_df = pd.DataFrame(
            {
                "run_id": args.run_id,
                "seed": int(args.seed),
                "user_id": fmeta["user_id"].to_numpy(),
                "anchor_idx": fmeta["idx"].to_numpy(),
                "anchor_text_id": fmeta["anchor_text_id"].to_numpy(),
                "anchor_timestamp": fmeta["anchor_timestamp"].to_numpy(),
                "delta_valence_pred": f_preds[:, 0],
                "delta_arousal_pred": f_preds[:, 1],
            }
        )
        if forecast_df["user_id"].duplicated().any():
            raise SystemExit("Forecast preds must contain exactly one row per user.")
        forecast_path = preds_dir / f"subtask2a_forecast_user_preds__{args.run_id}.parquet"
        forecast_df.to_parquet(forecast_path, index=False)
        print(f"Wrote forecast preds to: {forecast_path}")
        merge_run_metadata(
            repo_root=repo_root,
            run_id=args.run_id,
            updates={
                "artifacts": {"preds_forecast_user": artifact_ref(forecast_path, repo_root)},
                "counts": {"n_forecast_users_pred": int(len(forecast_df))},
                "config": {
                    "predict": {
                        "subtask2a": {
                            "seq_len": seq_len,
                            "ablate_no_history": ablate_no_history,
                            "pred_kind": "forecast",
                            "pred_path": str(forecast_path),
                        }
                    }
                },
                "diagnostics": {
                    "predict": {
                        "subtask2a": {
                            "forecast": summarize_pred_df(
                                forecast_df,
                                pred_cols={
                                    "valence": "delta_valence_pred",
                                    "arousal": "delta_arousal_pred",
                                },
                                true_cols=None,
                                bounds={"valence": (-4.0, 4.0), "arousal": (-2.0, 2.0)},
                            )
                        }
                    }
                },
            },
        )


if __name__ == "__main__":
    main()
