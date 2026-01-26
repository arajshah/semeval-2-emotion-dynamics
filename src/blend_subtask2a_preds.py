from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = [
    "user_id",
    "anchor_idx",
    "anchor_text_id",
    "anchor_timestamp",
    "delta_valence_pred",
    "delta_arousal_pred",
    "delta_valence_true",
    "delta_arousal_true",
]


def _parse_key_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    if not cols:
        raise ValueError("--key_cols must contain at least one column name.")
    return cols


def _check_required_cols(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _check_unique_keys(df: pd.DataFrame, key_cols: list[str], path: Path) -> None:
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        sample = df.loc[dup, key_cols].head(5).to_dict(orient="records")
        raise ValueError(f"Duplicate keys in {path} for {key_cols}: sample={sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend Subtask 2A preds toward a baseline.")
    parser.add_argument("--pred_a", required=True, type=str)
    parser.add_argument("--pred_b", required=True, type=str)
    parser.add_argument("--alpha_valence", required=True, type=float)
    parser.add_argument("--alpha_arousal", required=True, type=float)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--key_cols", type=str, default="user_id,anchor_idx")
    parser.add_argument(
        "--strict",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Error on any mismatch/duplicate keys/row drops.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.alpha_valence <= 1.0):
        raise ValueError("--alpha_valence must be in [0,1].")
    if not (0.0 <= args.alpha_arousal <= 1.0):
        raise ValueError("--alpha_arousal must be in [0,1].")

    pred_a = Path(args.pred_a)
    pred_b = Path(args.pred_b)
    out_path = Path(args.out_path)
    key_cols = _parse_key_cols(args.key_cols)

    if args.strict:
        forbidden = {
            "anchor_text_id",
            "anchor_timestamp",
            "delta_valence_true",
            "delta_arousal_true",
        }
        bad = [c for c in key_cols if c in forbidden]
        if bad:
            raise ValueError(
                f"In strict mode, --key_cols must NOT include {sorted(forbidden)}. "
                f"Found forbidden key cols: {bad}. "
                "Use stable identifiers like 'user_id,anchor_idx' (recommended)."
            )

    df_a = pd.read_parquet(pred_a)
    df_b = pd.read_parquet(pred_b)

    _check_required_cols(df_a, pred_a)
    _check_required_cols(df_b, pred_b)

    _check_unique_keys(df_a, key_cols, pred_a)
    _check_unique_keys(df_b, key_cols, pred_b)

    merged = df_a.merge(df_b, on=key_cols, how="inner", suffixes=("_a", "_b"))

    if args.strict:
        if len(merged) != len(df_a) or len(merged) != len(df_b):
            raise ValueError(
                f"Strict alignment failed: merged={len(merged)} a={len(df_a)} b={len(df_b)}"
            )
    else:
        if len(merged) != len(df_a) or len(merged) != len(df_b):
            print(
                f"WARNING: rowcount mismatch: merged={len(merged)} a={len(df_a)} b={len(df_b)}"
            )

    if args.strict:
        ts_a_dt = pd.to_datetime(merged["anchor_timestamp_a"], utc=True, errors="raise")
        ts_b_dt = pd.to_datetime(merged["anchor_timestamp_b"], utc=True, errors="raise")
        if not (ts_a_dt.view("int64") == ts_b_dt.view("int64")).all():
            raise ValueError(
                "Strict mismatch in anchor_timestamp between A and B (after UTC normalization)."
            )

        text_a = merged["anchor_text_id_a"].astype("string")
        text_b = merged["anchor_text_id_b"].astype("string")
        if not text_a.equals(text_b):
            raise ValueError("Strict mismatch in anchor_text_id between A and B.")

        for col in ["delta_valence_true", "delta_arousal_true"]:
            a = pd.to_numeric(merged[f"{col}_a"], errors="raise")
            b = pd.to_numeric(merged[f"{col}_b"], errors="raise")
            if not a.equals(b):
                raise ValueError(f"Strict mismatch in {col} between A and B.")

    anchor_timestamp_out = ts_a_dt.dt.strftime("%Y-%m-%dT%H:%M:%S%z").astype("string")

    out_df = pd.DataFrame(
        {
            "user_id": merged["user_id"].astype("string"),
            "anchor_idx": pd.to_numeric(merged["anchor_idx"], errors="raise").astype("int64"),
            "anchor_text_id": merged["anchor_text_id_a"].astype("string"),
            "anchor_timestamp": anchor_timestamp_out,
            "delta_valence_true": pd.to_numeric(merged["delta_valence_true_a"], errors="raise"),
            "delta_arousal_true": pd.to_numeric(merged["delta_arousal_true_a"], errors="raise"),
            "delta_valence_pred": args.alpha_valence * merged["delta_valence_pred_a"]
            + (1.0 - args.alpha_valence) * merged["delta_valence_pred_b"],
            "delta_arousal_pred": args.alpha_arousal * merged["delta_arousal_pred_a"]
            + (1.0 - args.alpha_arousal) * merged["delta_arousal_pred_b"],
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(
        f"Blended preds: n_rows={len(out_df)} "
        f"alpha_valence={args.alpha_valence} alpha_arousal={args.alpha_arousal} "
        f"key_cols={key_cols} out_path={out_path}"
    )


if __name__ == "__main__":
    main()

# Example usage:
# python -m src.blend_subtask2a_preds \
#   --pred_a reports/preds/subtask2a_val_user_preds__RUN_A.parquet \
#   --pred_b reports/preds/subtask2a_val_user_preds__RUN_B_LINEAR_PREV.parquet \
#   --alpha_valence 0.70 \
#   --alpha_arousal 0.70 \
#   --out_path reports/preds/subtask2a_val_user_preds__BLEND_TEST.parquet
#
# Then eval:
# python -m src.eval.phase0_eval --task subtask2a --regime unseen_user --seed 42 \
#   --run_id 'blend_test' --model_tag 'blend(alpha=0.7)' \
#   --pred_path reports/preds/subtask2a_val_user_preds__BLEND_TEST.parquet \
#   --split_path reports/splits/subtask2a_unseen_user_seed42.json
