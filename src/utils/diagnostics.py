from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


def _to_py(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def summarize_array(
    x: np.ndarray,
    *,
    bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, object]:
    x = np.asarray(x)
    nan_mask = np.isnan(x)
    inf_mask = ~np.isfinite(x)
    finite = x[np.isfinite(x)]
    summary = {
        "n": int(x.size),
        "nan_count": int(nan_mask.sum()),
        "inf_count": int((inf_mask & ~nan_mask).sum()),
    }
    if finite.size == 0:
        summary.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "p01": None,
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
                "p99": None,
                "max": None,
                "abs_p99": None,
                "abs_max": None,
            }
        )
    else:
        percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
        summary.update(
            {
                "mean": _to_py(float(finite.mean())),
                "std": _to_py(float(finite.std())),
                "min": _to_py(float(finite.min())),
                "p01": _to_py(float(percentiles[0])),
                "p05": _to_py(float(percentiles[1])),
                "p25": _to_py(float(percentiles[2])),
                "p50": _to_py(float(percentiles[3])),
                "p75": _to_py(float(percentiles[4])),
                "p95": _to_py(float(percentiles[5])),
                "p99": _to_py(float(percentiles[6])),
                "max": _to_py(float(finite.max())),
                "abs_p99": _to_py(float(np.percentile(np.abs(finite), 99))),
                "abs_max": _to_py(float(np.abs(finite).max())),
            }
        )

    if bounds is not None and finite.size > 0:
        lo, hi = bounds
        summary["frac_below"] = _to_py(float((finite < lo).mean()))
        summary["frac_above"] = _to_py(float((finite > hi).mean()))
    return summary


def summarize_pred_df(
    df: pd.DataFrame,
    *,
    pred_cols: Dict[str, str],
    true_cols: Optional[Dict[str, str]] = None,
    bounds: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {"n_rows": int(len(df))}
    if "user_id" in df.columns:
        out["n_users"] = int(df["user_id"].nunique())
    pred_summary = {}
    for target, col in pred_cols.items():
        if col in df.columns:
            pred_summary[target] = summarize_array(
                df[col].to_numpy(),
                bounds=bounds.get(target) if bounds else None,
            )
    out["pred"] = pred_summary

    if true_cols:
        true_summary = {}
        resid_summary = {}
        for target, tcol in true_cols.items():
            pcol = pred_cols.get(target)
            if tcol in df.columns and pcol in df.columns:
                true_summary[target] = summarize_array(
                    df[tcol].to_numpy(),
                    bounds=bounds.get(target) if bounds else None,
                )
                mask = ~(df[tcol].isna() | df[pcol].isna())
                resid = (df[pcol] - df[tcol])[mask].to_numpy()
                resid_summary[target] = summarize_array(resid)
        if true_summary:
            out["true"] = true_summary
        if resid_summary:
            out["resid"] = resid_summary
    return out


def apply_clip_to_bounds(
    df: pd.DataFrame,
    *,
    pred_cols: Dict[str, str],
    bounds: Dict[str, Tuple[float, float]],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    df_out = df.copy()
    report: Dict[str, object] = {}
    for target, col in pred_cols.items():
        if col not in df_out.columns:
            continue
        lo, hi = bounds[target]
        vals = df_out[col].to_numpy(dtype=float)
        before = float(((vals < lo) | (vals > hi)).mean()) if len(vals) else 0.0
        clipped = np.clip(vals, lo, hi)
        after = float(((clipped < lo) | (clipped > hi)).mean()) if len(clipped) else 0.0
        df_out[col] = clipped
        report[target] = {
            "before_out_of_bounds_frac": before,
            "after_out_of_bounds_frac": after,
            "bounds": [lo, hi],
        }
    return df_out, report
