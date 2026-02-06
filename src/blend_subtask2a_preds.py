from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _pick(df: pd.DataFrame, cols: list[str]) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {cols}. Have: {list(df.columns)}")


def _standardize_forecast(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    df = df.copy()

    uid_col = _pick(df, ["user_id"])
    df[uid_col] = df[uid_col].astype(str)

    v_col = _pick(df, ["delta_valence_pred", "pred_state_change_valence"])
    a_col = _pick(df, ["delta_arousal_pred", "pred_state_change_arousal"])

    keep_cols = ["user_id"]
    for c in ["anchor_idx", "anchor_text_id", "anchor_timestamp", "seed", "run_id"]:
        if c in df.columns:
            keep_cols.append(c)

    out = df[keep_cols + [v_col, a_col]].rename(
        columns={v_col: "delta_valence_pred", a_col: "delta_arousal_pred"}
    )

    if "anchor_timestamp" in out.columns:
        out["anchor_timestamp"] = pd.to_datetime(out["anchor_timestamp"], errors="coerce")

    if out["user_id"].duplicated().any():
        dup = out.loc[out["user_id"].duplicated(), "user_id"].iloc[0]
        raise SystemExit(
            f"{label} forecast has duplicate user_id rows (example: {dup}). Must be 1 row/user."
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend Subtask 2A forecast parquets (baseline + model) and write blended forecast parquet."
    )
    parser.add_argument("--baseline_run_id", required=True)
    parser.add_argument("--model_run_id", required=True)
    parser.add_argument("--alpha_valence", type=float, required=True)
    parser.add_argument("--alpha_arousal", type=float, required=True)

    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--preds_dir", default=None)
    parser.add_argument("--blend_run_id", default=None)
    parser.add_argument("--out_path", default=None)

    args = parser.parse_args()

    if not (0.0 <= args.alpha_valence <= 1.0):
        raise SystemExit("--alpha_valence must be in [0, 1].")
    if not (0.0 <= args.alpha_arousal <= 1.0):
        raise SystemExit("--alpha_arousal must be in [0, 1].")

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise SystemExit(f"Missing repo_root: {repo_root}")

    preds_dir = Path(args.preds_dir).resolve() if args.preds_dir else (repo_root / "reports" / "preds")
    preds_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = preds_dir / f"subtask2a_forecast_user_preds__{args.baseline_run_id}.parquet"
    model_path = preds_dir / f"subtask2a_forecast_user_preds__{args.model_run_id}.parquet"
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline forecast parquet: {baseline_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model forecast parquet: {model_path}")

    av_tag = f"{args.alpha_valence:.2f}".replace(".", "p")
    aa_tag = f"{args.alpha_arousal:.2f}".replace(".", "p")
    blend_run_id = (
        args.blend_run_id
        or f"subtask2a_blend_linprev__{args.model_run_id}__av{av_tag}_aa{aa_tag}"
    )

    out_path = (
        Path(args.out_path).resolve()
        if args.out_path
        else preds_dir / f"subtask2a_forecast_user_preds__{blend_run_id}.parquet"
    )

    df_base = _standardize_forecast(pd.read_parquet(baseline_path), label="BASELINE")
    df_model = _standardize_forecast(pd.read_parquet(model_path), label="MODEL")

    base_preds = df_base[["user_id", "delta_valence_pred", "delta_arousal_pred"]].rename(
        columns={
            "delta_valence_pred": "delta_valence_pred_base",
            "delta_arousal_pred": "delta_arousal_pred_base",
        }
    )

    merged = df_model.merge(base_preds, on="user_id", how="inner")
    if len(merged) != len(df_model) or len(merged) != len(df_base):
        raise SystemExit(
            f"Alignment mismatch during blending: merged={len(merged)} model={len(df_model)} base={len(df_base)}"
        )

    mv = merged["delta_valence_pred"].to_numpy(dtype=float)
    ma = merged["delta_arousal_pred"].to_numpy(dtype=float)
    bv = merged["delta_valence_pred_base"].to_numpy(dtype=float)
    ba = merged["delta_arousal_pred_base"].to_numpy(dtype=float)

    blend_v = args.alpha_valence * mv + (1.0 - args.alpha_valence) * bv
    blend_a = args.alpha_arousal * ma + (1.0 - args.alpha_arousal) * ba

    if not np.isfinite(blend_v).all() or not np.isfinite(blend_a).all():
        raise SystemExit("Blended forecast contains non-finite values (NaN/inf).")

    out_df = pd.DataFrame(
        {
            "run_id": blend_run_id,
            "seed": int(df_model["seed"].iloc[0]) if "seed" in df_model.columns else 42,
            "user_id": merged["user_id"].astype(str).to_numpy(),
            **({"anchor_idx": merged["anchor_idx"].to_numpy()} if "anchor_idx" in merged.columns else {}),
            **({"anchor_text_id": merged["anchor_text_id"].to_numpy()} if "anchor_text_id" in merged.columns else {}),
            **({"anchor_timestamp": merged["anchor_timestamp"].to_numpy()} if "anchor_timestamp" in merged.columns else {}),
            "delta_valence_pred": blend_v,
            "delta_arousal_pred": blend_a,
        }
    )

    if out_df["user_id"].duplicated().any():
        raise SystemExit("Blended forecast must contain exactly one row per user (duplicates found).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
