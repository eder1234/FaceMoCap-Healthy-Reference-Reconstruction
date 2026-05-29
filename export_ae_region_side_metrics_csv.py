#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_ae_region_side_metrics_csv.py

Export AE-based FaceMoCap anatomical metrics to CSV files for clinician-facing
heatmap generation.

This script does NOT generate figures.

Inputs expected:
  - overlay_index.csv produced by export_four_cloud_eval_overlays.py --save_npz
  - semantic_facial_labels.csv
  - optional facemocap_metadata_reference_split.csv for acquisition dates

Main outputs:
  - ae_region_side_metrics_long_selected.csv
  - ae_region_side_metrics_matrix_normalized.csv
  - ae_region_side_metrics_matrix_raw.csv
  - ae_region_side_normalization_stats.csv
  - ae_region_side_column_order.csv
  - ae_region_side_sample_manifest.csv
  - ae_region_side_failures.csv

Default behavior:
  - evaluates pathological VJ and OC samples only
  - uses movements M1-M5 only
  - keeps left, right, and center region-side groups
  - normalizes each metric by the healthy-evaluation distribution for the same
    movement, metric, region, and side
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None


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

# Default clinician-friendly 40-column order: 4 blocks × 10 columns.
# This is only an ordering file; the matrix CSV will contain all existing
# semantic region-side columns in this order.
DEFAULT_REGION_SIDE_BLOCKS = [
    [
        ("left", "front"),
        ("left", "eyebrow"),
        ("left", "upper eyelid"),
        ("left", "lower eyelid"),
        ("left", "lateral canthus"),
        ("right", "lateral canthus"),
        ("right", "lower eyelid"),
        ("right", "upper eyelid"),
        ("right", "eyebrow"),
        ("right", "front"),
    ],
    [
        ("left", "medial canthus"),
        ("left", "palpebra-malar groove"),
        ("left", "ala of nose"),
        ("left", "levator superioris nasi"),
        ("left", "levator superioris"),
        ("right", "levator superioris"),
        ("right", "levator superioris nasi"),
        ("right", "ala of nose"),
        ("right", "palpebra-malar groove"),
        ("right", "medial canthus"),
    ],
    [
        ("left", "zygomaticus minor"),
        ("left", "zygomaticus major"),
        ("left", "labial commissure"),
        ("left", "upper lip"),
        ("left", "lower lip"),
        ("right", "lower lip"),
        ("right", "upper lip"),
        ("right", "labial commissure"),
        ("right", "zygomaticus major"),
        ("right", "zygomaticus minor"),
    ],
    [
        ("left", "depressor anguli oris"),
        ("left", "mentalis"),
        ("center", "front"),
        ("center", "nasion"),
        ("center", "nasal bone"),
        ("center", "philtrum"),
        ("center", "lower lip"),
        ("center", "mentalis"),
        ("right", "mentalis"),
        ("right", "depressor anguli oris"),
    ],
]


def slugify(text: object) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_movement(value: object) -> str:
    txt = str(value).strip().upper()
    m = re.search(r"M\s*0*([1-9]\d*)", txt)
    if m:
        return f"M{int(m.group(1))}"
    # Metadata sometimes stores 1.0, 2.0, ...
    try:
        v = float(txt)
        if np.isfinite(v) and int(v) == v:
            return f"M{int(v)}"
    except Exception:
        pass
    return txt


def parse_date_from_text(*values: object) -> str:
    txt = " ".join(str(v) for v in values if pd.notna(v))
    # Prefer YYYYMMDD
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", txt)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Then YYYY-MM-DD / YYYY_MM_DD / YYYY.MM.DD
    m = re.search(r"(20\d{2})[-_.](\d{2})[-_.](\d{2})", txt)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def resolve_existing_path(
    path_value: object,
    *,
    project_root: Path,
    overlay_dir: Path,
    rel_value: Optional[object] = None,
) -> Path:
    candidates: List[Path] = []

    if pd.notna(path_value) and str(path_value).strip():
        p = Path(str(path_value))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(project_root / p)
            candidates.append(overlay_dir / p)

    if rel_value is not None and pd.notna(rel_value) and str(rel_value).strip():
        rp = Path(str(rel_value))
        candidates.append(rp)
        if not rp.is_absolute():
            candidates.append(overlay_dir / rp)
            candidates.append(project_root / rp)

    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c

    # Return the most likely path for readable failure messages.
    if unique_candidates:
        return unique_candidates[0]
    return Path(str(path_value))


def read_overlay_index(path: Path, metadata_path: Optional[Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    required = {"movement", "condition", "group", "participant_id", "sample_id", "npz_path", "npz_relpath"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"overlay_index is missing required columns: {missing}")

    df["movement"] = df["movement"].map(normalize_movement)
    df["condition_norm"] = df["condition"].astype(str).str.strip().str.lower()
    df["group_norm"] = df["group"].astype(str).str.strip().str.lower()
    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    if metadata_path is not None and metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        meta = meta.copy()
        if "complete_filepath" in meta.columns:
            keep = ["complete_filepath"]
            for col in ["acquisition_date", "facial_movement", "condition", "participant_id", "filename"]:
                if col in meta.columns:
                    keep.append(col)
            meta = meta[keep].drop_duplicates("complete_filepath")
            rename = {c: f"metadata_{c}" for c in keep if c != "complete_filepath"}
            meta = meta.rename(columns=rename)
            df = df.merge(meta, on="complete_filepath", how="left")

    if "metadata_acquisition_date" in df.columns:
        df["acquisition_date"] = df["metadata_acquisition_date"].fillna("")
    else:
        df["acquisition_date"] = ""

    missing_date = df["acquisition_date"].astype(str).str.strip().eq("") | df["acquisition_date"].isna()
    df.loc[missing_date, "acquisition_date"] = df.loc[missing_date].apply(
        lambda r: parse_date_from_text(
            r.get("complete_filepath", ""),
            r.get("filename", ""),
            r.get("sample_id", ""),
        ),
        axis=1,
    )

    return df


def read_semantic_labels(path: Path) -> pd.DataFrame:
    sem = pd.read_csv(path)
    sem = sem.copy()

    required = {"marker_id", "region", "side"}
    missing = sorted(required - set(sem.columns))
    if missing:
        raise ValueError(f"semantic labels are missing required columns: {missing}")

    sem["marker_id"] = pd.to_numeric(sem["marker_id"], errors="raise").astype(int)
    sem["region"] = sem["region"].astype(str).str.strip().str.lower()
    sem["side"] = sem["side"].astype(str).str.strip().str.lower()

    if sem["marker_id"].min() < 0 or sem["marker_id"].max() > 104:
        raise ValueError(
            f"Expected marker_id in [0, 104], got [{sem['marker_id'].min()}, {sem['marker_id'].max()}]."
        )

    return sem.sort_values("marker_id").reset_index(drop=True)


def region_side_id(side: str, region: str) -> str:
    prefix = {"left": "L", "right": "R", "center": "C"}.get(str(side).lower(), slugify(side).upper())
    return f"{prefix}__{slugify(region)}"


def region_side_label(side: str, region: str) -> str:
    prefix = {"left": "L", "right": "R", "center": "C"}.get(str(side).lower(), str(side).title())
    return f"{prefix} {str(region).title()}"


def build_region_side_order(sem: pd.DataFrame) -> pd.DataFrame:
    existing = set((r["side"], r["region"]) for _, r in sem[["side", "region"]].drop_duplicates().iterrows())

    rows = []
    order = 0
    used = set()

    for block_idx, block in enumerate(DEFAULT_REGION_SIDE_BLOCKS, start=1):
        for position_idx, (side, region) in enumerate(block, start=1):
            if (side, region) not in existing:
                continue
            rows.append(
                {
                    "region_side_id": region_side_id(side, region),
                    "display_label": region_side_label(side, region),
                    "side": side,
                    "region": region,
                    "block": block_idx,
                    "position_in_block": position_idx,
                    "global_order": order,
                }
            )
            used.add((side, region))
            order += 1

    # Append any region-side group not covered by the predefined order.
    remaining = sorted(existing - used, key=lambda x: ({"left": 0, "center": 1, "right": 2}.get(x[0], 9), x[1]))
    for side, region in remaining:
        block_idx = int(order // 10) + 1
        position_idx = int(order % 10) + 1
        rows.append(
            {
                "region_side_id": region_side_id(side, region),
                "display_label": region_side_label(side, region),
                "side": side,
                "region": region,
                "block": block_idx,
                "position_in_block": position_idx,
                "global_order": order,
            }
        )
        order += 1

    return pd.DataFrame(rows)


def lowpass_filter_sequence(
    seq: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Low-pass filter a T×N×3 sequence with linear interpolation across invalid frames.

    The original validity mask is NOT changed; metrics still use the original mask.
    """
    seq = np.asarray(seq, dtype=np.float64)
    out = seq.copy()

    if cutoff_hz <= 0:
        return out

    if butter is None or filtfilt is None:
        warnings.warn("scipy.signal is unavailable; low-pass filtering was skipped.")
        return out

    T, N, C = seq.shape
    nyq = 0.5 * float(fs_hz)
    if cutoff_hz >= nyq:
        warnings.warn(f"cutoff_hz={cutoff_hz} >= Nyquist={nyq}; low-pass filtering was skipped.")
        return out

    b, a = butter(order, cutoff_hz / nyq, btype="low")

    if mask is None:
        mask_bool = np.ones((T, N), dtype=bool)
    else:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != (T, N):
            raise ValueError(f"mask shape {mask_bool.shape} does not match sequence shape {(T, N)}")

    xgrid = np.arange(T)
    for i in range(N):
        valid_marker = mask_bool[:, i]
        for c in range(C):
            y = seq[:, i, c]
            ok = valid_marker & np.isfinite(y)
            if ok.sum() < 4:
                continue

            filled = np.interp(xgrid, xgrid[ok], y[ok])
            try:
                filtered = filtfilt(b, a, filled, method="gust")
            except TypeError:
                filtered = filtfilt(b, a, filled)
            except ValueError:
                # Sequence too short for filtfilt padding; keep interpolated signal.
                filtered = filled

            out[:, i, c] = filtered

    return out


@dataclass
class MarkerMetricResult:
    marker_table: pd.DataFrame
    region_side_table: pd.DataFrame


def compute_marker_metrics(
    observed: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    sem: pd.DataFrame,
    *,
    fs_hz: float,
    cutoff_hz: float,
    filter_order: int,
    eps: float,
    min_valid_frames: int,
) -> MarkerMetricResult:
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(target_mask).astype(bool)

    if observed.shape != reference.shape:
        raise ValueError(f"observed shape {observed.shape} != reference shape {reference.shape}")
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError(f"Expected observed shape T×N×3, got {observed.shape}")
    if mask.shape != observed.shape[:2]:
        raise ValueError(f"target_mask shape {mask.shape} != observed shape {observed.shape[:2]}")

    if observed.shape[1] != 105:
        warnings.warn(f"Expected 105 facial markers; got {observed.shape[1]}.")

    observed_f = lowpass_filter_sequence(
        observed, mask, fs_hz=fs_hz, cutoff_hz=cutoff_hz, order=filter_order
    )
    reference_f = lowpass_filter_sequence(
        reference, mask, fs_hz=fs_hz, cutoff_hz=cutoff_hz, order=filter_order
    )

    finite_obs = np.isfinite(observed_f).all(axis=-1)
    finite_ref = np.isfinite(reference_f).all(axis=-1)
    valid = mask & finite_obs & finite_ref

    T, N, _ = observed_f.shape
    d_obs = observed_f - observed_f[0:1, :, :]
    d_ref = reference_f - reference_f[0:1, :, :]

    rows = []
    sem_lookup = sem.set_index("marker_id")[["region", "side"]].to_dict("index")

    for i in range(N):
        idx = valid[:, i]
        valid_count = int(idx.sum())
        region = sem_lookup.get(i, {}).get("region", "")
        side = sem_lookup.get(i, {}).get("side", "")

        neutral_ok = bool(valid[0, i]) and np.isfinite(observed_f[0, i]).all() and np.isfinite(reference_f[0, i]).all()

        values = {m: np.nan for m in METRIC_ORDER}

        if valid_count >= min_valid_frames:
            values["trajectory_abnormality"] = float(
                np.linalg.norm(observed_f[idx, i, :] - reference_f[idx, i, :], axis=1).mean()
            )

        if valid_count >= min_valid_frames and neutral_ok:
            obs_norm = np.linalg.norm(d_obs[idx, i, :], axis=1)
            ref_norm = np.linalg.norm(d_ref[idx, i, :], axis=1)

            amp_obs = float(np.nanmax(obs_norm)) if obs_norm.size else np.nan
            amp_ref = float(np.nanmax(ref_norm)) if ref_norm.size else np.nan

            if np.isfinite(amp_obs) and np.isfinite(amp_ref):
                log_ratio = float(np.log((amp_obs + eps) / (amp_ref + eps)))
                values["amplitude_abnormality"] = abs(log_ratio)
                values["hypokinesia"] = max(0.0, -log_ratio)
                values["hyperkinesia"] = max(0.0, log_ratio)

            u_ref = d_ref[idx, i, :] / (ref_norm[:, None] + eps)
            opposite = np.maximum(0.0, -np.sum(d_obs[idx, i, :] * u_ref, axis=1))
            denom = float(np.sum(obs_norm) + eps)
            values["counter_direction_ratio"] = float(np.sum(opposite) / denom)

        for metric in METRIC_ORDER:
            rows.append(
                {
                    "marker_id": i,
                    "region": region,
                    "side": side,
                    "region_side_id": region_side_id(side, region),
                    "region_side_label": region_side_label(side, region),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "raw_value": values[metric],
                    "valid_frame_count": valid_count,
                    "neutral_valid": neutral_ok,
                }
            )

    marker_df = pd.DataFrame(rows)

    # Equal-weight aggregation over markers with finite values.
    agg = (
        marker_df.dropna(subset=["raw_value"])
        .groupby(["region", "side", "region_side_id", "region_side_label", "metric", "metric_label"], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            n_valid_markers=("marker_id", "nunique"),
            mean_valid_frame_count=("valid_frame_count", "mean"),
        )
    )

    return MarkerMetricResult(marker_table=marker_df, region_side_table=agg)


def load_npz_metrics(
    row: pd.Series,
    sem: pd.DataFrame,
    *,
    project_root: Path,
    overlay_dir: Path,
    fs_hz: float,
    cutoff_hz: float,
    filter_order: int,
    eps: float,
    min_valid_frames: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    npz_path = resolve_existing_path(
        row.get("npz_path", ""),
        project_root=project_root,
        overlay_dir=overlay_dir,
        rel_value=row.get("npz_relpath", ""),
    )

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(npz_path) as z:
        required = ["observed_face", "aev3_healthy_reconstruction", "target_mask"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {missing}; found {z.files}")

        observed = z["observed_face"]
        reference = z["aev3_healthy_reconstruction"]
        mask = z["target_mask"]

    result = compute_marker_metrics(
        observed,
        reference,
        mask,
        sem,
        fs_hz=fs_hz,
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        eps=eps,
        min_valid_frames=min_valid_frames,
    )

    return result.marker_table, result.region_side_table, npz_path


def add_sample_metadata(metric_df: pd.DataFrame, row: pd.Series, npz_path: Path) -> pd.DataFrame:
    out = metric_df.copy()
    metadata_cols = {
        "movement": row.get("movement", ""),
        "condition": row.get("condition", ""),
        "group": row.get("group", ""),
        "participant_id": row.get("participant_id", ""),
        "acquisition_date": row.get("acquisition_date", ""),
        "filename": row.get("filename", ""),
        "sample_id": row.get("sample_id", ""),
        "complete_filepath": row.get("complete_filepath", ""),
        "html_path": row.get("html_path", ""),
        "html_relpath": row.get("html_relpath", ""),
        "npz_path_resolved": str(npz_path),
        "npz_path": row.get("npz_path", ""),
        "npz_relpath": row.get("npz_relpath", ""),
        "alignment_score": row.get("alignment_score", np.nan),
        "qc_valid_marker_frame_fraction": row.get("qc_valid_marker_frame_fraction", np.nan),
    }
    for col, val in reversed(list(metadata_cols.items())):
        out.insert(0, col, val)
    return out


def compute_all_metrics(
    rows: pd.DataFrame,
    sem: pd.DataFrame,
    *,
    project_root: Path,
    overlay_dir: Path,
    fs_hz: float,
    cutoff_hz: float,
    filter_order: int,
    eps: float,
    min_valid_frames: int,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_tables = []
    marker_tables = []
    failures = []

    for idx, row in rows.iterrows():
        try:
            marker_df, region_df, npz_path = load_npz_metrics(
                row,
                sem,
                project_root=project_root,
                overlay_dir=overlay_dir,
                fs_hz=fs_hz,
                cutoff_hz=cutoff_hz,
                filter_order=filter_order,
                eps=eps,
                min_valid_frames=min_valid_frames,
            )
            region_tables.append(add_sample_metadata(region_df, row, npz_path))
            marker_tables.append(add_sample_metadata(marker_df, row, npz_path))
        except Exception as exc:
            failures.append(
                {
                    "set": label,
                    "movement": row.get("movement", ""),
                    "condition": row.get("condition", ""),
                    "group": row.get("group", ""),
                    "participant_id": row.get("participant_id", ""),
                    "acquisition_date": row.get("acquisition_date", ""),
                    "filename": row.get("filename", ""),
                    "sample_id": row.get("sample_id", ""),
                    "npz_path": row.get("npz_path", ""),
                    "npz_relpath": row.get("npz_relpath", ""),
                    "error": str(exc),
                }
            )

    region_all = pd.concat(region_tables, ignore_index=True) if region_tables else pd.DataFrame()
    marker_all = pd.concat(marker_tables, ignore_index=True) if marker_tables else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    return region_all, marker_all, failures_df


def build_normalization_stats(
    healthy_long: pd.DataFrame,
    eps: float,
    *,
    denominator_floor_fraction: float,
    denominator_floor_quantile: float,
) -> pd.DataFrame:
    """Compute healthy-reference normalization statistics.

    The first version used only (q95 - median). That is fragile for sparse
    one-sided metrics such as hypokinesia, where healthy median and q95 may both
    be zero. This revised version keeps the same clinical idea--values above
    the healthy median are abnormal--but uses a safer denominator:

        denominator = max(local robust scale, movement-metric floor, metric floor)

    The floor prevents tiny denominators from creating huge artificial
    abnormality indices.
    """
    if healthy_long.empty:
        raise ValueError("No healthy-evaluation metrics were computed; cannot normalize target samples.")

    finite = healthy_long.dropna(subset=["raw_value"]).copy()
    if finite.empty:
        raise ValueError("Healthy-evaluation metrics contain no finite raw values; cannot normalize.")

    def q(x, pct):
        return float(np.nanpercentile(np.asarray(x, dtype=float), pct))

    stats = (
        finite
        .groupby(["movement", "metric", "region", "side", "region_side_id", "region_side_label"], as_index=False)
        .agg(
            healthy_n=("raw_value", "size"),
            healthy_median=("raw_value", "median"),
            healthy_q05=("raw_value", lambda x: q(x, 5)),
            healthy_q25=("raw_value", lambda x: q(x, 25)),
            healthy_q75=("raw_value", lambda x: q(x, 75)),
            healthy_q90=("raw_value", lambda x: q(x, 90)),
            healthy_q95=("raw_value", lambda x: q(x, 95)),
            healthy_q99=("raw_value", lambda x: q(x, 99)),
            healthy_mean=("raw_value", "mean"),
            healthy_sd=("raw_value", "std"),
            healthy_min=("raw_value", "min"),
            healthy_max=("raw_value", "max"),
        )
    )

    # Local candidate scales. All are non-negative by construction after clipping.
    stats["den_q95_minus_median"] = (stats["healthy_q95"] - stats["healthy_median"]).clip(lower=0)
    stats["den_q90_minus_median"] = (stats["healthy_q90"] - stats["healthy_median"]).clip(lower=0)
    stats["den_q75_minus_median"] = (stats["healthy_q75"] - stats["healthy_median"]).clip(lower=0)
    stats["den_iqr"] = (stats["healthy_q75"] - stats["healthy_q25"]).clip(lower=0)
    stats["den_robust_iqr_half"] = 0.5 * stats["den_iqr"]
    stats["den_sd"] = stats["healthy_sd"].fillna(0).clip(lower=0)
    stats["den_max_minus_median"] = (stats["healthy_max"] - stats["healthy_median"]).clip(lower=0)

    local_candidates = [
        "den_q95_minus_median",
        "den_q90_minus_median",
        "den_q75_minus_median",
        "den_robust_iqr_half",
        "den_sd",
        "den_max_minus_median",
    ]
    stats["local_denominator_candidate"] = stats[local_candidates].max(axis=1, skipna=True)
    stats.loc[stats["local_denominator_candidate"] <= eps, "local_denominator_candidate"] = np.nan

    # Metric-level floors from the complete healthy-evaluation pool.
    metric_floors = []
    for metric, g in finite.groupby("metric"):
        vals = g["raw_value"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            floor = np.nan
            q50 = q95 = np.nan
        else:
            q50 = float(np.nanpercentile(vals, 50))
            q95 = float(np.nanpercentile(vals, 95))
            floor = denominator_floor_fraction * max(q95 - q50, 0.0)
        metric_floors.append(
            {
                "metric": metric,
                "metric_global_q50": q50,
                "metric_global_q95": q95,
                "metric_global_floor": floor,
            }
        )
    metric_floors = pd.DataFrame(metric_floors)

    # Movement-metric floors are less broad than global metric floors, but still
    # much more stable than marker-region denominators.
    movement_metric_floors = []
    for (movement, metric), g in finite.groupby(["movement", "metric"]):
        vals = g["raw_value"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            floor = np.nan
            q50 = q95 = np.nan
        else:
            q50 = float(np.nanpercentile(vals, 50))
            q95 = float(np.nanpercentile(vals, 95))
            floor = denominator_floor_fraction * max(q95 - q50, 0.0)
        movement_metric_floors.append(
            {
                "movement": movement,
                "metric": metric,
                "movement_metric_global_q50": q50,
                "movement_metric_global_q95": q95,
                "movement_metric_floor": floor,
            }
        )
    movement_metric_floors = pd.DataFrame(movement_metric_floors)

    # Positive local-denominator percentile by metric, which gives a data-driven
    # lower bound while ignoring zero-denominator groups.
    local_floor_rows = []
    q_pct = float(denominator_floor_quantile)
    for metric, g in stats.groupby("metric"):
        vals = g["local_denominator_candidate"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > eps)]
        if vals.size == 0:
            floor = np.nan
        else:
            floor = float(np.nanpercentile(vals, q_pct))
        local_floor_rows.append({"metric": metric, "metric_local_positive_den_floor": floor})
    local_floor = pd.DataFrame(local_floor_rows)

    stats = stats.merge(metric_floors, on="metric", how="left")
    stats = stats.merge(movement_metric_floors, on=["movement", "metric"], how="left")
    stats = stats.merge(local_floor, on="metric", how="left")

    floor_cols = [
        "metric_global_floor",
        "movement_metric_floor",
        "metric_local_positive_den_floor",
    ]
    stats["denominator_floor"] = stats[floor_cols].max(axis=1, skipna=True)
    stats.loc[stats["denominator_floor"] <= eps, "denominator_floor"] = np.nan

    stats["normalization_denominator_local"] = stats["local_denominator_candidate"]
    stats["normalization_denominator_safe"] = stats[
        ["normalization_denominator_local", "denominator_floor"]
    ].max(axis=1, skipna=True)
    stats.loc[stats["normalization_denominator_safe"] <= eps, "normalization_denominator_safe"] = np.nan

    # Backward-compatible column name, but now safe.
    stats["normalization_denominator"] = stats["normalization_denominator_safe"]

    return stats


def _build_healthy_distribution_lookup(healthy_long: pd.DataFrame) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Map (movement, metric, region_side_id) to sorted healthy raw values."""
    lookup: Dict[Tuple[str, str, str], np.ndarray] = {}
    finite = healthy_long.dropna(subset=["raw_value"]).copy()
    for key, g in finite.groupby(["movement", "metric", "region_side_id"]):
        vals = g["raw_value"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        lookup[key] = np.sort(vals)
    return lookup


def _percentile_rank(value: float, distribution: np.ndarray) -> float:
    if not np.isfinite(value) or distribution.size == 0:
        return np.nan
    # Percent of healthy values <= target value.
    return float(np.searchsorted(distribution, value, side="right") / distribution.size)


def apply_normalization(
    selected_long: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    healthy_long: pd.DataFrame,
    norm_clip_min: float,
    norm_clip_max: float,
    softlog_cap: float,
    min_healthy_n: int,
) -> pd.DataFrame:
    keys = ["movement", "metric", "region", "side", "region_side_id", "region_side_label"]
    out = selected_long.merge(stats, on=keys, how="left")

    out["normalization_reliable"] = (
        out["healthy_n"].fillna(0).astype(float).ge(float(min_healthy_n))
        & out["normalization_denominator_safe"].notna()
        & np.isfinite(out["normalization_denominator_safe"])
    )

    out["normalized_value_unclipped"] = (
        (out["raw_value"] - out["healthy_median"]) / out["normalization_denominator_safe"]
    )
    out.loc[out["normalization_denominator_safe"].isna(), "normalized_value_unclipped"] = np.nan

    # Backward-compatible clipped version. This is not the preferred heatmap value.
    out["normalized_value"] = out["normalized_value_unclipped"].clip(lower=norm_clip_min, upper=norm_clip_max)

    # Clinically interpretable abnormality index: only excess over healthy median.
    out["abnormality_index"] = out["normalized_value_unclipped"].clip(lower=0)

    # Softly compressed [0, 1] value for heatmap display.
    # softlog_cap defines which abnormality index maps to 1 after clipping.
    cap = max(float(softlog_cap), 1e-6)
    out["softlog_0_1"] = np.log1p(out["abnormality_index"]) / np.log1p(cap)
    out["softlog_0_1"] = out["softlog_0_1"].clip(lower=0, upper=1)

    # Reliability-masked versions for plotting. Unreliable cells are NaN, not zero.
    for col in ["normalized_value_unclipped", "normalized_value", "abnormality_index", "softlog_0_1"]:
        out[f"{col}_reliable"] = out[col].where(out["normalization_reliable"], np.nan)

    # Healthy percentile rank can be useful for debugging and clinician-facing
    # visualizations. Values near 1 mean the target exceeds most healthy controls.
    lookup = _build_healthy_distribution_lookup(healthy_long)
    percentile_vals = []
    for _, r in out.iterrows():
        key = (r.get("movement", ""), r.get("metric", ""), r.get("region_side_id", ""))
        dist = lookup.get(key, np.array([], dtype=float))
        percentile_vals.append(_percentile_rank(float(r["raw_value"]) if pd.notna(r["raw_value"]) else np.nan, dist))
    out["healthy_percentile_rank"] = percentile_vals
    out["healthy_percentile_rank_reliable"] = out["healthy_percentile_rank"].where(out["normalization_reliable"], np.nan)

    return out


def pivot_matrix(
    long_df: pd.DataFrame,
    column_order: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    id_cols = [
        "participant_id",
        "acquisition_date",
        "movement",
        "filename",
        "sample_id",
        "condition",
        "group",
        "metric",
        "metric_label",
        "html_path",
        "html_relpath",
        "npz_path",
        "npz_relpath",
        "npz_path_resolved",
        "alignment_score",
        "qc_valid_marker_frame_fraction",
    ]

    cols = column_order["region_side_id"].tolist()

    piv = long_df.pivot_table(
        index=id_cols,
        columns="region_side_id",
        values=value_col,
        aggfunc="mean",
    ).reset_index()

    # Ensure all expected columns exist and appear in the intended order.
    for c in cols:
        if c not in piv.columns:
            piv[c] = np.nan

    extra_cols = [c for c in piv.columns if c not in id_cols and c not in cols]
    return piv[id_cols + cols + extra_cols]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export AE region-side anatomical metrics as CSV files for heatmap generation."
    )

    ap.add_argument(
        "--overlay_index",
        default="align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv",
        help="Path to overlay_index.csv.",
    )
    ap.add_argument(
        "--semantic_labels",
        default="semantic_facial_labels.csv",
        help="Path to semantic_facial_labels.csv.",
    )
    ap.add_argument(
        "--metadata",
        default="facemocap_metadata_reference_split.csv",
        help="Optional metadata CSV used to retrieve acquisition dates.",
    )
    ap.add_argument(
        "--project_root",
        default=".",
        help="Project root used to resolve relative paths from overlay_index.csv.",
    )
    ap.add_argument(
        "--out_dir",
        default="align_mean_movement_refactor/ae_region_side_clinical_csv",
        help="Output directory.",
    )

    ap.add_argument("--patients", nargs="*", default=["VJ", "OC"], help="Patient IDs to export.")
    ap.add_argument("--movements", nargs="*", default=["M1", "M2", "M3", "M4", "M5"], help="Movements to export.")
    ap.add_argument("--condition", default="pathological", help="Target condition to export.")

    ap.add_argument("--fs_hz", type=float, default=100.0)
    ap.add_argument("--cutoff_hz", type=float, default=8.0)
    ap.add_argument("--filter_order", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min_valid_frames", type=int, default=10)

    ap.add_argument("--norm_clip_min", type=float, default=0.0)
    ap.add_argument("--norm_clip_max", type=float, default=1.0)
    ap.add_argument(
        "--softlog_cap",
        type=float,
        default=5.0,
        help="Abnormality index that maps to 1.0 in the softlog_0_1 heatmap matrix.",
    )
    ap.add_argument(
        "--min_healthy_n",
        type=int,
        default=10,
        help="Minimum number of healthy-evaluation samples required for reliable normalization.",
    )
    ap.add_argument(
        "--denominator_floor_fraction",
        type=float,
        default=0.25,
        help="Fraction of the global healthy metric range used as denominator floor.",
    )
    ap.add_argument(
        "--denominator_floor_quantile",
        type=float,
        default=25.0,
        help="Percentile of positive local denominators used as an additional denominator floor.",
    )

    ap.add_argument(
        "--save_marker_level",
        action="store_true",
        help="Also save marker-level metrics for selected patients. This file is larger.",
    )
    ap.add_argument(
        "--save_healthy_long",
        action="store_true",
        help="Also save healthy-evaluation long metrics used to compute normalization statistics.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    overlay_index_path = Path(args.overlay_index)
    semantic_path = Path(args.semantic_labels)
    metadata_path = Path(args.metadata) if args.metadata else None
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = overlay_index_path.resolve().parent if overlay_index_path.exists() else Path(".").resolve()

    overlay = read_overlay_index(overlay_index_path, metadata_path)
    sem = read_semantic_labels(semantic_path)
    column_order = build_region_side_order(sem)

    wanted_movements = {normalize_movement(m) for m in args.movements}
    wanted_patients = {str(p).strip() for p in args.patients}
    wanted_condition = str(args.condition).strip().lower()

    healthy_rows = overlay[
        (overlay["condition_norm"] == "healthy")
        & (overlay["movement"].isin(wanted_movements))
    ].copy()

    selected_rows = overlay[
        (overlay["condition_norm"] == wanted_condition)
        & (overlay["participant_id"].isin(wanted_patients))
        & (overlay["movement"].isin(wanted_movements))
    ].copy()

    selected_rows = selected_rows.sort_values(
        ["participant_id", "acquisition_date", "movement", "filename", "sample_id"]
    ).reset_index(drop=True)

    healthy_rows = healthy_rows.sort_values(
        ["movement", "participant_id", "filename", "sample_id"]
    ).reset_index(drop=True)

    print(f"Healthy-evaluation rows for normalization: {len(healthy_rows)}")
    print(f"Selected target rows: {len(selected_rows)}")
    print(f"Output directory: {out_dir}")

    if healthy_rows.empty:
        raise RuntimeError("No healthy-evaluation rows found for normalization.")
    if selected_rows.empty:
        raise RuntimeError("No selected pathological rows found. Check --patients, --movements, and --condition.")

    healthy_long, healthy_marker, healthy_fail = compute_all_metrics(
        healthy_rows,
        sem,
        project_root=project_root,
        overlay_dir=overlay_dir,
        fs_hz=args.fs_hz,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        eps=args.eps,
        min_valid_frames=args.min_valid_frames,
        label="healthy_eval",
    )

    selected_long_raw, selected_marker, selected_fail = compute_all_metrics(
        selected_rows,
        sem,
        project_root=project_root,
        overlay_dir=overlay_dir,
        fs_hz=args.fs_hz,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        eps=args.eps,
        min_valid_frames=args.min_valid_frames,
        label="selected_target",
    )

    stats = build_normalization_stats(
        healthy_long,
        eps=args.eps,
        denominator_floor_fraction=args.denominator_floor_fraction,
        denominator_floor_quantile=args.denominator_floor_quantile,
    )
    selected_long = apply_normalization(
        selected_long_raw,
        stats,
        healthy_long=healthy_long,
        norm_clip_min=args.norm_clip_min,
        norm_clip_max=args.norm_clip_max,
        softlog_cap=args.softlog_cap,
        min_healthy_n=args.min_healthy_n,
    )

    matrix_norm = pivot_matrix(selected_long, column_order, value_col="normalized_value")
    matrix_norm_unclipped = pivot_matrix(selected_long, column_order, value_col="normalized_value_unclipped")
    matrix_abnormality_index = pivot_matrix(selected_long, column_order, value_col="abnormality_index")
    matrix_softlog = pivot_matrix(selected_long, column_order, value_col="softlog_0_1")
    matrix_softlog_reliable = pivot_matrix(selected_long, column_order, value_col="softlog_0_1_reliable")
    matrix_abnormality_index_reliable = pivot_matrix(selected_long, column_order, value_col="abnormality_index_reliable")
    matrix_percentile = pivot_matrix(selected_long, column_order, value_col="healthy_percentile_rank")
    matrix_percentile_reliable = pivot_matrix(selected_long, column_order, value_col="healthy_percentile_rank_reliable")
    matrix_raw = pivot_matrix(selected_long, column_order, value_col="raw_value")

    # Save outputs.
    selected_long.to_csv(out_dir / "ae_region_side_metrics_long_selected.csv", index=False)
    matrix_softlog.to_csv(out_dir / "ae_region_side_metrics_matrix_softlog_0_1.csv", index=False)
    matrix_softlog_reliable.to_csv(out_dir / "ae_region_side_metrics_matrix_softlog_0_1_reliable.csv", index=False)
    matrix_abnormality_index.to_csv(out_dir / "ae_region_side_metrics_matrix_abnormality_index.csv", index=False)
    matrix_abnormality_index_reliable.to_csv(out_dir / "ae_region_side_metrics_matrix_abnormality_index_reliable.csv", index=False)
    matrix_norm_unclipped.to_csv(out_dir / "ae_region_side_metrics_matrix_normalized_unclipped.csv", index=False)
    matrix_norm.to_csv(out_dir / "ae_region_side_metrics_matrix_normalized.csv", index=False)
    matrix_percentile.to_csv(out_dir / "ae_region_side_metrics_matrix_healthy_percentile.csv", index=False)
    matrix_percentile_reliable.to_csv(out_dir / "ae_region_side_metrics_matrix_healthy_percentile_reliable.csv", index=False)
    matrix_raw.to_csv(out_dir / "ae_region_side_metrics_matrix_raw.csv", index=False)
    stats.to_csv(out_dir / "ae_region_side_normalization_stats.csv", index=False)
    column_order.to_csv(out_dir / "ae_region_side_column_order.csv", index=False)

    sample_manifest_cols = [
        "participant_id",
        "acquisition_date",
        "movement",
        "filename",
        "sample_id",
        "condition",
        "group",
        "html_path",
        "html_relpath",
        "npz_path",
        "npz_relpath",
        "alignment_score",
        "qc_valid_marker_frame_fraction",
    ]
    selected_rows[sample_manifest_cols].to_csv(out_dir / "ae_region_side_sample_manifest.csv", index=False)

    failures = pd.concat([healthy_fail, selected_fail], ignore_index=True)
    failures.to_csv(out_dir / "ae_region_side_failures.csv", index=False)

    if args.save_marker_level:
        selected_marker.to_csv(out_dir / "ae_region_side_marker_metrics_selected.csv", index=False)

    if args.save_healthy_long:
        healthy_long.to_csv(out_dir / "ae_region_side_metrics_long_healthy_eval.csv", index=False)

    print("[OK] Wrote:")
    for name in [
        "ae_region_side_metrics_long_selected.csv",
        "ae_region_side_metrics_matrix_softlog_0_1.csv",
        "ae_region_side_metrics_matrix_softlog_0_1_reliable.csv",
        "ae_region_side_metrics_matrix_abnormality_index.csv",
        "ae_region_side_metrics_matrix_abnormality_index_reliable.csv",
        "ae_region_side_metrics_matrix_normalized_unclipped.csv",
        "ae_region_side_metrics_matrix_normalized.csv",
        "ae_region_side_metrics_matrix_healthy_percentile.csv",
        "ae_region_side_metrics_matrix_healthy_percentile_reliable.csv",
        "ae_region_side_metrics_matrix_raw.csv",
        "ae_region_side_normalization_stats.csv",
        "ae_region_side_column_order.csv",
        "ae_region_side_sample_manifest.csv",
        "ae_region_side_failures.csv",
    ]:
        print(f"  - {out_dir / name}")

    print(f"[OK] Healthy failures: {len(healthy_fail)}")
    print(f"[OK] Selected target failures: {len(selected_fail)}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()
