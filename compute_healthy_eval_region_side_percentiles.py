#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_healthy_eval_region_side_percentiles.py

Compute healthy-evaluation anatomical percentile references for AE-based
FaceMoCap region-side metrics.

This revised version fixes a potential memory blow-up at the matrix-pivot step.
The previous script could be killed by the OS after processing all samples
because pandas pivot_table(dropna=False) may generate a large Cartesian product
when many metadata columns are used as the index.

Main outputs:
  - healthy_eval_region_side_percentiles.csv
  - healthy_eval_region_side_raw_values_long.csv
  - healthy_eval_region_side_matrix_raw.csv
  - healthy_eval_region_side_failures.csv

Example:
  cd /media/rodriguez/easystore/Data_FaceMoCap

  # Global pathological reference:
  python compute_healthy_eval_region_side_percentiles.py \
    --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
    --semantic_labels semantic_facial_labels.csv \
    --project_root . \
    --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv \
    --movements M1 M2 M3 M4 M5

  # Leave-one-subject-out reference for a healthy sanity check:
  python compute_healthy_eval_region_side_percentiles.py \
    --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
    --semantic_labels semantic_facial_labels.csv \
    --project_root . \
    --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv \
    --movements M1 M2 M3 M4 M5 \
    --loo_participant JE
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-8

METRIC_ORDER = [
    "trajectory_abnormality",
    "amplitude_abnormality",
    "hypokinesia",
    "hyperkinesia",
    "counter_direction_ratio",
]

METRIC_LABELS = {
    "trajectory_abnormality": "Trajectory abnormality",
    "amplitude_abnormality": "Amplitude abnormality",
    "hypokinesia": "Hypokinesia",
    "hyperkinesia": "Hyperkinesia",
    "counter_direction_ratio": "Counter-direction ratio",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_movement(value: object) -> str:
    s = str(value).strip().upper()
    m = re.search(r"M0*([1-9]\d*)", s)
    if m:
        return f"M{int(m.group(1))}"
    return s


def safe_token(value: object) -> str:
    """Compact filesystem-safe token used in output filenames."""
    s = str(value).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "unknown"


def norm_col_name(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> Optional[str]:
    norm_to_original = {norm_col_name(c): c for c in columns}
    for cand in candidates:
        n = norm_col_name(cand)
        if n in norm_to_original:
            return norm_to_original[n]
    if required:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}")
    return None


def candidate_paths(value: object, project_root: Optional[Path], base_dir: Path) -> List[Path]:
    if value is None or pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []

    p = Path(s)
    out = []
    if p.is_absolute():
        out.append(p)
    else:
        out.append(base_dir / p)
        if project_root is not None:
            out.append(project_root / p)

    uniq = []
    seen = set()
    for q in out:
        key = str(q)
        if key not in seen:
            uniq.append(q)
            seen.add(key)
    return uniq


def resolve_npz_path(row: pd.Series, project_root: Optional[Path], base_dir: Path) -> Optional[Path]:
    for col in ["npz_path_resolved", "npz_path", "npz_relpath"]:
        if col not in row.index:
            continue
        for p in candidate_paths(row.get(col), project_root, base_dir):
            if p.exists():
                return p

    # Fallback: infer NPZ name from HTML name.
    for col in ["html_path", "html_relpath"]:
        if col not in row.index:
            continue
        for h in candidate_paths(row.get(col), project_root, base_dir):
            if not h.exists():
                continue
            cand = h.with_name(h.name.replace("_four_cloud_overlay.html", "_four_cloud_reconstructions.npz"))
            if cand.exists():
                return cand

    return None


def normalize_side(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"l", "left", "gauche", "izquierda"}:
        return "L"
    if s in {"r", "right", "droite", "derecha"}:
        return "R"
    if s in {"c", "center", "centre", "midline", "median", "central"}:
        return "C"
    return s.upper() if s else "UNK"


def normalize_region(value: object) -> str:
    s = str(value).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_semantic_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    if labels.empty:
        raise ValueError(f"Semantic labels file is empty: {path}")

    marker_col = find_column(
        labels.columns,
        ["marker_index", "marker", "marker_id", "id", "index", "point_index", "landmark_index", "landmark_id"],
        required=True,
    )
    region_col = find_column(
        labels.columns,
        ["region", "facial_region", "anatomical_region", "region_name", "label", "anatomical_label"],
        required=True,
    )
    side_col = find_column(labels.columns, ["side", "laterality", "hemiface"], required=True)

    out = labels[[marker_col, region_col, side_col]].copy()
    out = out.rename(columns={marker_col: "marker_index", region_col: "region", side_col: "side"})
    out["marker_index"] = pd.to_numeric(out["marker_index"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["marker_index"]).copy()
    out["marker_index"] = out["marker_index"].astype(int)

    # Convert possible 1-based facial indices to 0-based.
    if out["marker_index"].min() == 1 and out["marker_index"].max() in {105, 108}:
        out["marker_index"] = out["marker_index"] - 1

    # Convert possible full marker indices including 3 dental markers to facial 0..104.
    if out["marker_index"].max() >= 105 and out["marker_index"].min() <= 3:
        possible = out["marker_index"] - 3
        if possible.min() >= 0 and possible.max() <= 104:
            out["marker_index"] = possible

    out = out[(out["marker_index"] >= 0) & (out["marker_index"] <= 104)].copy()
    out["side"] = out["side"].map(normalize_side)
    out["region"] = out["region"].map(normalize_region)
    out["region_side"] = out["side"] + "__" + out["region"]

    if out.empty:
        raise ValueError("No valid semantic labels remained after filtering to facial marker indices 0..104.")

    return out


def lowpass_trajectories(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    if cutoff <= 0 or fs <= 0:
        return x

    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return x

    T = x.shape[0]
    if T < 8:
        return x

    nyq = 0.5 * fs
    wn = float(cutoff) / nyq
    if not (0 < wn < 1):
        return x

    try:
        b, a = butter(4, wn, btype="low")
    except Exception:
        return x

    y = np.array(x, dtype=float, copy=True)
    t = np.arange(T)

    for i in range(y.shape[1]):
        for c in range(y.shape[2]):
            v = y[:, i, c]
            finite = np.isfinite(v)
            if finite.sum() < 4:
                continue
            if not finite.all():
                v = v.copy()
                v[~finite] = np.interp(t[~finite], t[finite], v[finite])
            try:
                y[:, i, c] = filtfilt(b, a, v, axis=0)
            except Exception:
                y[:, i, c] = v

    return y


def nanmean_axis0_safe(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    count = finite.sum(axis=0)
    s = np.where(finite, a, 0.0).sum(axis=0)
    out = np.full(a.shape[1:], np.nan, dtype=float)
    ok = count > 0
    out[ok] = s[ok] / count[ok]
    return out


def nanmax_axis0_safe(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    out = np.full(a.shape[1:], np.nan, dtype=float)
    if a.ndim == 2:
        for j in range(a.shape[1]):
            vals = a[:, j]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[j] = np.max(vals)
    else:
        # Not expected here, but keep safe.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmax(a, axis=0)
    return out


def load_npz_arrays(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    required = ["observed_face", "aev3_healthy_reconstruction"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"Missing arrays in {npz_path}: {missing}")

    observed = np.asarray(z["observed_face"], dtype=float)
    ae = np.asarray(z["aev3_healthy_reconstruction"], dtype=float)

    if observed.ndim != 3 or ae.ndim != 3:
        raise ValueError(f"Expected T×N×3 arrays in {npz_path}; got {observed.shape} and {ae.shape}.")
    if observed.shape != ae.shape:
        raise ValueError(f"Observed and AE shape mismatch in {npz_path}: {observed.shape} vs {ae.shape}.")
    if observed.shape[1] != 105:
        raise ValueError(f"Expected 105 facial markers in {npz_path}; got {observed.shape[1]}.")

    if "target_mask" in z.files:
        mask = np.asarray(z["target_mask"]).astype(bool)
        if mask.shape != observed.shape[:2]:
            raise ValueError(f"Mask shape mismatch in {npz_path}: {mask.shape} vs {observed.shape[:2]}.")
    else:
        mask = np.isfinite(observed).all(axis=-1) & np.isfinite(ae).all(axis=-1)

    finite = np.isfinite(observed).all(axis=-1) & np.isfinite(ae).all(axis=-1)
    mask = mask & finite
    return observed, ae, mask


def marker_metrics(
    observed: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    fs: float = 100.0,
    lowpass_hz: float = 8.0,
    eps: float = EPS,
) -> Dict[str, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(reference, dtype=float)
    m = np.asarray(mask).astype(bool)

    if lowpass_hz and lowpass_hz > 0:
        obs = lowpass_trajectories(obs, fs=fs, cutoff=lowpass_hz)
        ref = lowpass_trajectories(ref, fs=fs, cutoff=lowpass_hz)

    valid_counts = m.sum(axis=0).astype(float)
    valid_marker = valid_counts > 0

    # Trajectory abnormality
    d = np.linalg.norm(obs - ref, axis=-1)
    d = np.where(m, d, np.nan)
    traj = nanmean_axis0_safe(d)

    # Neutral-relative displacements
    obs0 = obs[0:1, :, :]
    ref0 = ref[0:1, :, :]
    dobs = obs - obs0
    dref = ref - ref0

    obs_amp_t = np.linalg.norm(dobs, axis=-1)
    ref_amp_t = np.linalg.norm(dref, axis=-1)
    obs_amp_t = np.where(m, obs_amp_t, np.nan)
    ref_amp_t = np.where(m, ref_amp_t, np.nan)

    obs_amp = nanmax_axis0_safe(obs_amp_t)
    ref_amp = nanmax_axis0_safe(ref_amp_t)

    log_ratio = np.log((obs_amp + eps) / (ref_amp + eps))
    amp_abn = np.abs(log_ratio)
    hypokinesia = np.maximum(0.0, -log_ratio)
    hyperkinesia = np.maximum(0.0, log_ratio)

    # Counter-direction ratio
    ref_norm = np.linalg.norm(dref, axis=-1, keepdims=True)
    uref = dref / (ref_norm + eps)
    proj = np.sum(dobs * uref, axis=-1)
    opp = np.maximum(0.0, -proj)
    obs_disp_norm = np.linalg.norm(dobs, axis=-1)

    opp = np.where(m, opp, 0.0)
    obs_disp_norm = np.where(m, obs_disp_norm, 0.0)
    cdr = opp.sum(axis=0) / (obs_disp_norm.sum(axis=0) + eps)

    out = {
        "trajectory_abnormality": traj,
        "amplitude_abnormality": amp_abn,
        "hypokinesia": hypokinesia,
        "hyperkinesia": hyperkinesia,
        "counter_direction_ratio": cdr,
    }

    for k in list(out.keys()):
        arr = np.asarray(out[k], dtype=float)
        arr[~valid_marker] = np.nan
        out[k] = arr

    return out


def aggregate_region_side(marker_values: np.ndarray, labels: pd.DataFrame) -> Dict[str, float]:
    res = {}
    for region_side, sub in labels.groupby("region_side", sort=True):
        idx = sub["marker_index"].to_numpy(dtype=int)
        idx = idx[(idx >= 0) & (idx < len(marker_values))]
        if len(idx) == 0:
            res[region_side] = np.nan
            continue
        vals = marker_values[idx]
        finite = np.isfinite(vals)
        res[region_side] = float(np.nanmean(vals)) if finite.any() else np.nan
    return res


def select_healthy_eval_rows(df: pd.DataFrame, movements: List[str]) -> pd.DataFrame:
    out = df.copy()

    if "movement" in out.columns:
        out["movement_norm"] = out["movement"].map(normalize_movement)
    elif "facial_movement" in out.columns:
        out["movement_norm"] = out["facial_movement"].map(normalize_movement)
    else:
        out["movement_norm"] = ""

    wanted_movements = {normalize_movement(m) for m in movements}
    out = out[out["movement_norm"].isin(wanted_movements)].copy()

    # Main expected case: group == eval_healthy.
    if "group" in out.columns:
        group_mask = out["group"].astype(str).str.strip().str.lower().isin(
            ["eval_healthy", "healthy_eval", "evaluation_healthy"]
        )
    else:
        group_mask = pd.Series(False, index=out.index)

    if "condition" in out.columns:
        cond_mask = out["condition"].astype(str).str.strip().str.lower().eq("healthy")
    else:
        cond_mask = pd.Series(False, index=out.index)

    selected = out[group_mask | (cond_mask & group_mask)].copy()

    # Fallback if group naming is absent/different.
    if selected.empty and "reference_split" in out.columns and "condition" in out.columns:
        selected = out[
            out["reference_split"].astype(str).str.strip().str.lower().isin(["evaluation", "eval", "healthy_eval"])
            & out["condition"].astype(str).str.strip().str.lower().eq("healthy")
        ].copy()

    # Final fallback: condition healthy only. Use with warning.
    if selected.empty and "condition" in out.columns:
        selected = out[out["condition"].astype(str).str.strip().str.lower().eq("healthy")].copy()

    return selected.reset_index(drop=True)


def compute_percentile_table(long_df: pd.DataFrame, min_n_reliable: int) -> pd.DataFrame:
    rows = []
    group_cols = ["movement", "metric", "metric_label", "side", "region", "region_side"]

    for key, g in long_df.groupby(group_cols, dropna=False, sort=True):
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(len(vals))

        if n == 0:
            stats = {k: np.nan for k in ["min", "p05", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean", "std", "iqr"]}
        else:
            stats = {
                "min": float(np.nanmin(vals)),
                "p05": float(np.nanpercentile(vals, 5)),
                "p25": float(np.nanpercentile(vals, 25)),
                "p50": float(np.nanpercentile(vals, 50)),
                "p75": float(np.nanpercentile(vals, 75)),
                "p90": float(np.nanpercentile(vals, 90)),
                "p95": float(np.nanpercentile(vals, 95)),
                "p99": float(np.nanpercentile(vals, 99)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if n >= 2 else 0.0,
                "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            }

        movement, metric, metric_label, side, region, region_side = key
        rows.append({
            "movement": movement,
            "metric": metric,
            "metric_label": metric_label,
            "side": side,
            "region": region,
            "region_side": region_side,
            "healthy_n": n,
            "reliable": int(n >= min_n_reliable),
            **stats,
            "threshold_mild_p75": stats["p75"],
            "threshold_moderate_p90": stats["p90"],
            "threshold_high_p95": stats["p95"],
            "threshold_extreme_max": stats["max"],
        })

    return pd.DataFrame(rows)


def pivot_matrix_raw_safe(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe wide matrix export.

    Avoids pivot_table(dropna=False), which can create a huge Cartesian product
    across metadata columns and get the process killed by the OS.
    """
    meta_cols = [
        "movement",
        "participant_id",
        "filename",
        "sample_id",
        "npz_path",
        "metric",
        "metric_label",
    ]
    meta_cols = [c for c in meta_cols if c in long_df.columns]

    tmp = long_df.copy()
    tmp["_row_id"] = tmp.groupby(meta_cols, dropna=False, sort=False).ngroup()

    meta = tmp[["_row_id"] + meta_cols].drop_duplicates("_row_id")
    wide = tmp.pivot(index="_row_id", columns="region_side", values="value").reset_index()
    wide.columns.name = None

    out = meta.merge(wide, on="_row_id", how="left").drop(columns=["_row_id"])
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay_index", required=True, help="overlay_index.csv generated by export_four_cloud_eval_overlays.py")
    ap.add_argument("--semantic_labels", required=True, help="semantic_facial_labels.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--project_root", default=None, help="Project root used to resolve relative NPZ paths")
    ap.add_argument("--movements", nargs="*", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--fs", type=float, default=100.0, help="Sampling frequency used for low-pass filtering")
    ap.add_argument("--lowpass_hz", type=float, default=8.0, help="Low-pass cutoff in Hz; set <=0 to disable")
    ap.add_argument("--min_n_reliable", type=int, default=10, help="Minimum healthy samples for reliable threshold")
    ap.add_argument("--exclude_participants", nargs="*", default=[], help="Healthy participant IDs to exclude from percentile computation")
    ap.add_argument("--loo_participant", default=None, help="Shortcut for leave-one-subject-out: exclude this healthy participant ID")
    ap.add_argument("--output_prefix", default=None, help="Optional filename prefix. If omitted, an automatic prefix is used.")
    ap.add_argument("--no_matrix_raw", action="store_true", help="Skip healthy_eval_region_side_matrix_raw.csv export")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    overlay_index = Path(args.overlay_index)
    semantic_labels = Path(args.semantic_labels)
    out_dir = Path(args.out_dir)
    project_root = Path(args.project_root).resolve() if args.project_root else None
    base_dir = overlay_index.parent.resolve()

    ensure_dir(out_dir)

    if not overlay_index.exists():
        raise FileNotFoundError(f"Overlay index not found: {overlay_index}")
    if not semantic_labels.exists():
        raise FileNotFoundError(f"Semantic labels not found: {semantic_labels}")

    index_df = pd.read_csv(overlay_index)
    labels = load_semantic_labels(semantic_labels)

    rows_df = select_healthy_eval_rows(index_df, args.movements)

    exclude_participants = list(args.exclude_participants or [])
    if args.loo_participant:
        exclude_participants.append(args.loo_participant)
    exclude_participants = [str(x) for x in exclude_participants if str(x).strip()]
    exclude_set = set(exclude_participants)

    n_before_exclusion = len(rows_df)
    if exclude_set:
        if "participant_id" not in rows_df.columns:
            raise ValueError("--exclude_participants/--loo_participant was provided, but overlay_index has no participant_id column.")
        rows_df = rows_df[~rows_df["participant_id"].astype(str).isin(exclude_set)].copy()

    print(f"[INFO] Healthy-evaluation rows selected before exclusion: {n_before_exclusion}")
    if exclude_set:
        print(f"[INFO] Excluded healthy participant(s): {sorted(exclude_set)}")
    print(f"[INFO] Healthy-evaluation rows retained for percentile computation: {len(rows_df)}")

    if rows_df.empty:
        raise ValueError("No healthy-evaluation rows found after exclusion. Check condition/group/movement columns and excluded participant IDs.")

    long_rows = []
    failures = []

    for n, row in rows_df.iterrows():
        movement = normalize_movement(row.get("movement", row.get("facial_movement", "")))
        sample_id = row.get("sample_id", "")
        participant_id = row.get("participant_id", "")
        filename = row.get("filename", "")

        npz_path = resolve_npz_path(row, project_root=project_root, base_dir=base_dir)
        if npz_path is None:
            failures.append({
                "movement": movement,
                "participant_id": participant_id,
                "filename": filename,
                "sample_id": sample_id,
                "error": "Could not resolve NPZ path",
            })
            print(f"[FAIL] {n+1}/{len(rows_df)} Could not resolve NPZ: {sample_id}")
            continue

        try:
            observed, ae, mask = load_npz_arrays(npz_path)
            mm = marker_metrics(
                observed=observed,
                reference=ae,
                mask=mask,
                fs=args.fs,
                lowpass_hz=args.lowpass_hz,
            )

            for metric in METRIC_ORDER:
                region_values = aggregate_region_side(mm[metric], labels)
                for region_side, value in region_values.items():
                    side, region = region_side.split("__", 1)
                    long_rows.append({
                        "movement": movement,
                        "participant_id": participant_id,
                        "filename": filename,
                        "sample_id": sample_id,
                        "npz_path": str(npz_path),
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "side": side,
                        "region": region,
                        "region_side": region_side,
                        "value": value,
                    })

            print(f"[OK] {n+1}/{len(rows_df)} {movement} {participant_id} {sample_id}")

        except Exception as exc:
            failures.append({
                "movement": movement,
                "participant_id": participant_id,
                "filename": filename,
                "sample_id": sample_id,
                "npz_path": str(npz_path),
                "error": str(exc),
            })
            print(f"[FAIL] {n+1}/{len(rows_df)} {movement} {sample_id}: {exc}")

    long_df = pd.DataFrame(long_rows)
    failures_df = pd.DataFrame(failures)

    failures_path = out_dir / "healthy_eval_region_side_failures.csv"
    failures_df.to_csv(failures_path, index=False)

    if long_df.empty:
        raise RuntimeError(f"No healthy-evaluation metric rows were computed. See {failures_path}")

    if args.output_prefix:
        prefix = safe_token(args.output_prefix)
    elif exclude_set:
        prefix = "healthy_eval_excluding_" + "_".join(safe_token(x) for x in sorted(exclude_set))
    else:
        prefix = "healthy_eval"

    long_path = out_dir / f"{prefix}_region_side_raw_values_long.csv"
    pct_path = out_dir / f"{prefix}_region_side_percentiles.csv"
    matrix_path = out_dir / f"{prefix}_region_side_matrix_raw.csv"
    manifest_path = out_dir / f"{prefix}_region_side_percentile_manifest.csv"

    # Save a small manifest so the threshold cohort is explicit.
    manifest = pd.DataFrame([{
        "output_prefix": prefix,
        "excluded_participants": ";".join(sorted(exclude_set)),
        "n_rows_before_exclusion": n_before_exclusion,
        "n_rows_after_exclusion": len(rows_df),
        "movements": ";".join(args.movements),
        "min_n_reliable": args.min_n_reliable,
        "lowpass_hz": args.lowpass_hz,
        "fs": args.fs,
    }])
    manifest.to_csv(manifest_path, index=False)

    # Save the long file before any wide matrix operation, so partial useful
    # output survives even if a later step fails.
    long_df.to_csv(long_path, index=False)

    pct_df = compute_percentile_table(long_df, min_n_reliable=args.min_n_reliable)
    pct_df.to_csv(pct_path, index=False)

    if not args.no_matrix_raw:
        matrix_raw = pivot_matrix_raw_safe(long_df)
        matrix_raw.to_csv(matrix_path, index=False)
    else:
        matrix_path = None

    print("[DONE] Exported:")
    print(f"  {long_path}")
    print(f"  {pct_path}")
    if matrix_path is not None:
        print(f"  {matrix_path}")
    print(f"  {failures_path}")
    print(f"  {manifest_path}")
    print(f"[DONE] Healthy metric rows: {len(long_df)}")
    print(f"[DONE] Percentile rows: {len(pct_df)}")
    print(f"[DONE] Failures: {len(failures_df)}")


if __name__ == "__main__":
    main()
