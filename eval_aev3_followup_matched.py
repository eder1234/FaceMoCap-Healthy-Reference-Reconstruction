#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_aev3_followup_matched.py

AE-v3-only MOVEMENT-MATCHED longitudinal/follow-up evaluation for
FaceMoCap healthy-reference reconstructions.

This script consumes the NPZ files exported by:

    python export_four_cloud_eval_overlays.py --out_dir ... --save_npz

and the corresponding overlay_index.csv. It quantifies how abnormality metrics
change across acquisition dates for pathological participants with repeated
visits. It does NOT compare reference methods. Only the AE-v3 healthy
reconstruction is used. Unlike eval_aev3_followup.py, this stricter version
aggregates samples by movement first, keeps only movements shared across visits
for each participant, and then computes visit-level metrics with equal movement
weighting.

Main outputs
------------
- followup_sample_metrics.csv
    one row per valid pathological NPZ sample.
- followup_marker_metrics.csv
    one row per sample and marker.
- followup_region_metrics.csv
    one row per sample and anatomical region.
- followup_side_metrics.csv
    one row per sample and side.
- followup_region_side_metrics.csv
    one row per sample and region-side pair.
- followup_visit_movement_metrics.csv
    median metrics per participant, acquisition date, and movement.
- followup_common_movements_by_participant.csv
    movements retained for matched longitudinal comparisons.
- followup_visit_global_metrics.csv
    movement-matched visit metrics per participant and acquisition date.
- followup_visit_global_metrics_unmatched.csv
    exploratory unmatched visit metrics retained for audit only.
- followup_change_from_baseline.csv
    absolute and relative change from the first available visit.
- followup_participant_summary.csv
    first/last visit summary per participant and metric.
- followup_available_dates.csv
    audit table of available dates and sample counts.
- figures/
    longitudinal line plots and region-side heatmaps.

Example
-------
conda activate facemocap_ai
python eval_aev3_followup_matched.py \
  --overlay_root align_mean_movement_refactor/four_cloud_eval_overlays \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --metadata facemocap_metadata_reference_split.csv \
  --semantic_labels semantic_facial_labels.csv \
  --out_dir align_mean_movement_refactor/aev3_followup_eval_matched \
  --fs_hz 100 \
  --lowpass_hz 8 \
  --min_dates 2

For a stricter analysis limited to patients with more than two dates:

python eval_aev3_followup_matched.py ... --min_dates 3
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, savgol_filter
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None
    savgol_filter = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

EPS = 1e-8
AEV3_ARRAY = "aev3_healthy_reconstruction"
OBS_ARRAY = "observed_face"
MASK_ARRAY = "target_mask"

CORE_METRICS = [
    "trajectory_mae",
    "trajectory_rmse",
    "displacement_mae",
    "displacement_rmse",
    "amplitude_log_ratio_abs",
    "hypokinesia_proxy",
    "hyperkinesia_proxy",
    "counter_direction_ratio",
    "direction_cosine_mean",
    "amplitude_observed",
    "amplitude_reference",
]

LOWER_IS_BETTER = {
    "trajectory_mae": True,
    "trajectory_rmse": True,
    "displacement_mae": True,
    "displacement_rmse": True,
    "amplitude_log_ratio_abs": True,
    "hypokinesia_proxy": True,
    "hyperkinesia_proxy": True,
    "counter_direction_ratio": True,
    # direction_cosine_mean is different: higher is closer to the expected direction.
    "direction_cosine_mean": False,
}


@dataclass
class SampleRecord:
    npz_path: Path
    sample_id: str
    participant_id: str
    movement: str
    condition: str
    group: str
    filename: str
    complete_filepath: str
    acquisition_date: Optional[pd.Timestamp]
    date_source: str
    alignment_score: float = math.nan
    qc_valid_marker_frame_fraction: float = math.nan


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return math.nan
    return float(np.nanmean(arr))


def safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return math.nan
    return float(np.nanmedian(arr))


def normalize_movement(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("M"):
        m = re.search(r"M\s*0*([1-9]\d*)", text)
        if m:
            return f"M{int(m.group(1))}"
    try:
        f = float(text)
        if np.isfinite(f):
            return f"M{int(f)}"
    except Exception:
        pass
    m = re.search(r"(?:^|[^A-Z0-9])M\s*0*([1-9]\d*)(?:[^A-Z0-9]|$)", text)
    if m:
        return f"M{int(m.group(1))}"
    return text


def parse_date_from_text(text: object) -> Optional[pd.Timestamp]:
    """Infer a date from strings containing YYYY-MM-DD or YYYYMMDD.

    Handles filenames like 2023062001_DC_M1.csv by taking the first valid
    eight digits, i.e. 20230620, and ignoring the extra visit/repetition code.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text)
    # Direct ISO-like date.
    m = re.search(r"(20\d{2})[-_/\.](0[1-9]|1[0-2])[-_/\.](0[1-9]|[12]\d|3[01])", s)
    if m:
        try:
            return pd.to_datetime("-".join(m.groups()), errors="coerce")
        except Exception:
            pass
    # Compact date. Validate candidate dates rather than blindly accepting.
    for m in re.finditer(r"(20\d{6})", s):
        candidate = m.group(1)
        dt = pd.to_datetime(candidate, format="%Y%m%d", errors="coerce")
        if not pd.isna(dt):
            return dt
    return None


def read_metadata(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.copy()
    for c in ["complete_filepath", "filename", "participant_id", "condition"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "facial_movement" in df.columns:
        df["movement_norm"] = df["facial_movement"].map(normalize_movement)
    else:
        df["movement_norm"] = ""
    if "acquisition_date" in df.columns:
        df["acquisition_date_parsed"] = pd.to_datetime(df["acquisition_date"], errors="coerce")
    else:
        df["acquisition_date_parsed"] = pd.NaT
    return df


def build_metadata_lookup(metadata: pd.DataFrame) -> Dict[Tuple[str, str, str, str], pd.Timestamp]:
    """Build several exact-ish lookup keys for acquisition dates."""
    lookup: Dict[Tuple[str, str, str, str], pd.Timestamp] = {}
    if metadata.empty:
        return lookup
    for _, row in metadata.iterrows():
        dt = row.get("acquisition_date_parsed", pd.NaT)
        if pd.isna(dt):
            continue
        pid = str(row.get("participant_id", "")).strip()
        movement = normalize_movement(row.get("movement_norm", row.get("facial_movement", "")))
        filename = Path(str(row.get("filename", ""))).name
        complete = str(row.get("complete_filepath", ""))
        complete_name = Path(complete).name
        keys = [
            ("complete", pid, movement, complete),
            ("filename", pid, movement, filename),
            ("filename", pid, movement, complete_name),
        ]
        for key in keys:
            lookup[key] = pd.Timestamp(dt)
    return lookup


def resolve_npz_path(row: pd.Series, overlay_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for col in ["npz_path", "npz_relpath"]:
        if col not in row or pd.isna(row[col]) or str(row[col]).strip() == "":
            continue
        p = Path(str(row[col]))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(overlay_root / p)
            # npz_path in overlay_index may already include overlay_root as a relative prefix.
            # If overlay_root is also supplied, avoid duplicating it by also trying from cwd.
            candidates.append(Path.cwd() / p)
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    # Return the most informative candidate even if it does not exist, so the audit can report it.
    return candidates[0] if candidates else None


def acquisition_date_for_row(
    row: pd.Series,
    metadata_lookup: Dict[Tuple[str, str, str, str], pd.Timestamp],
) -> Tuple[Optional[pd.Timestamp], str]:
    pid = str(row.get("participant_id", "")).strip()
    movement = normalize_movement(row.get("movement", ""))
    filename = Path(str(row.get("filename", ""))).name
    complete = str(row.get("complete_filepath", ""))
    complete_name = Path(complete).name

    for key in [
        ("complete", pid, movement, complete),
        ("filename", pid, movement, filename),
        ("filename", pid, movement, complete_name),
    ]:
        if key in metadata_lookup:
            return metadata_lookup[key], "metadata"

    for col in ["complete_filepath", "filename", "sample_id", "html_path", "npz_path", "npz_relpath"]:
        dt = parse_date_from_text(row.get(col, ""))
        if dt is not None and not pd.isna(dt):
            return pd.Timestamp(dt), f"inferred_from_{col}"

    return None, "missing"


def load_overlay_records(
    overlay_index_path: Path,
    overlay_root: Path,
    metadata_path: Optional[Path],
    condition: str = "pathological",
) -> Tuple[List[SampleRecord], pd.DataFrame]:
    if not overlay_index_path.exists():
        raise FileNotFoundError(f"overlay_index.csv not found: {overlay_index_path}")
    overlay = pd.read_csv(overlay_index_path)
    overlay = overlay.copy()
    overlay["condition_norm"] = overlay.get("condition", "").astype(str).str.strip().str.lower()
    metadata = read_metadata(metadata_path)
    metadata_lookup = build_metadata_lookup(metadata)

    records: List[SampleRecord] = []
    audit_rows: List[Dict[str, object]] = []
    for _, row in overlay.iterrows():
        cond = str(row.get("condition", "")).strip().lower()
        if condition and cond != condition.lower():
            continue
        npz_path = resolve_npz_path(row, overlay_root)
        dt, source = acquisition_date_for_row(row, metadata_lookup)
        rec = SampleRecord(
            npz_path=npz_path if npz_path is not None else Path(""),
            sample_id=str(row.get("sample_id", "")),
            participant_id=str(row.get("participant_id", "")).strip(),
            movement=normalize_movement(row.get("movement", "")),
            condition=cond,
            group=str(row.get("group", "")),
            filename=str(row.get("filename", "")),
            complete_filepath=str(row.get("complete_filepath", "")),
            acquisition_date=dt,
            date_source=source,
            alignment_score=safe_float(row.get("alignment_score", math.nan)),
            qc_valid_marker_frame_fraction=safe_float(row.get("qc_valid_marker_frame_fraction", math.nan)),
        )
        records.append(rec)
        audit_rows.append(
            {
                "participant_id": rec.participant_id,
                "movement": rec.movement,
                "sample_id": rec.sample_id,
                "filename": rec.filename,
                "npz_path": str(rec.npz_path),
                "npz_exists": rec.npz_path.exists(),
                "acquisition_date": rec.acquisition_date.date().isoformat() if rec.acquisition_date is not None else "",
                "date_source": rec.date_source,
            }
        )
    return records, pd.DataFrame(audit_rows)


def load_semantic_labels(path: Path, n_markers: int) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"marker_id", "region", "side", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Semantic label file is missing columns: {sorted(missing)}")
    labels = labels.copy()
    labels["marker_id"] = pd.to_numeric(labels["marker_id"], errors="raise").astype(int)
    labels["region"] = labels["region"].astype(str).str.strip()
    labels["side"] = labels["side"].astype(str).str.strip().str.lower()
    labels["label"] = labels["label"].astype(str).str.strip()
    if labels["marker_id"].duplicated().any():
        dup = labels.loc[labels["marker_id"].duplicated(), "marker_id"].tolist()
        raise ValueError(f"Duplicated marker_id values in semantic labels: {dup}")
    all_ids = pd.DataFrame({"marker_id": np.arange(n_markers, dtype=int)})
    labels = all_ids.merge(labels, on="marker_id", how="left")
    labels["region"] = labels["region"].fillna("unknown")
    labels["side"] = labels["side"].fillna("unknown")
    labels["label"] = labels["label"].fillna(labels["marker_id"].map(lambda x: f"marker_{x}"))
    return labels


def interpolate_small_gaps_1d(y: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    y = y.astype(float, copy=True)
    valid = valid.astype(bool)
    n = len(y)
    if valid.sum() < 2:
        return y
    idx = np.arange(n)
    interp = np.interp(idx, idx[valid], y[valid])
    if max_gap <= 0:
        y[~valid] = interp[~valid]
        return y
    missing = ~valid
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            y[i:j] = interp[i:j]
        i = j
    return y


def smooth_sequence(
    seq: np.ndarray,
    mask: np.ndarray,
    fs_hz: float,
    lowpass_hz: float,
    filter_order: int = 2,
    savgol_window: int = 9,
    savgol_poly: int = 2,
    max_interp_gap: int = 5,
) -> np.ndarray:
    """Temporally smooth a (T,M,3) trajectory without changing invalid masks.

    Invalid values are interpolated only for filtering stability; final metrics still
    use the original validity mask. If SciPy's Butterworth filter is available and
    lowpass_hz is valid, it is used. Otherwise a Savitzky-Golay fallback is used.
    """
    out = np.asarray(seq, dtype=float).copy()
    T, M, C = out.shape
    valid_marker = mask.astype(bool) & np.isfinite(out).all(axis=-1)
    use_butter = (
        butter is not None
        and filtfilt is not None
        and fs_hz > 0
        and lowpass_hz > 0
        and lowpass_hz < fs_hz / 2.0
        and T > max(9, 3 * filter_order)
    )
    if use_butter:
        b, a = butter(filter_order, lowpass_hz / (fs_hz / 2.0), btype="low")
        padlen = min(3 * max(len(a), len(b)), T - 1)
    else:
        b = a = None
        padlen = 0

    # Ensure valid odd SG window.
    if savgol_window % 2 == 0:
        savgol_window += 1
    if savgol_window > T:
        savgol_window = T if T % 2 == 1 else T - 1
    if savgol_window <= savgol_poly:
        savgol_window = savgol_poly + 3
        if savgol_window % 2 == 0:
            savgol_window += 1
    if savgol_window > T:
        savgol_window = 0

    for m in range(M):
        valid = valid_marker[:, m]
        if valid.sum() < 3:
            continue
        for c in range(C):
            y = out[:, m, c]
            y_interp = interpolate_small_gaps_1d(y, valid, max_gap=max_interp_gap)
            if not np.isfinite(y_interp).all():
                good = np.isfinite(y_interp)
                if good.sum() < 2:
                    continue
                y_interp = np.interp(np.arange(T), np.arange(T)[good], y_interp[good])
            try:
                if use_butter:
                    yf = filtfilt(b, a, y_interp, padlen=padlen)
                elif savgol_filter is not None and savgol_window >= 5:
                    yf = savgol_filter(y_interp, window_length=savgol_window, polyorder=savgol_poly, mode="interp")
                else:
                    # Simple moving average fallback.
                    w = min(5, T)
                    kernel = np.ones(w) / float(w)
                    yf = np.convolve(y_interp, kernel, mode="same")
                out[:, m, c] = yf
            except Exception:
                # Keep original if filtering fails for a marker/coordinate.
                pass
    return out


def valid_mask_for(obs: np.ndarray, ref: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    return (
        target_mask.astype(bool)
        & np.isfinite(obs).all(axis=-1)
        & np.isfinite(ref).all(axis=-1)
    )


def trajectory_amplitude(seq: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 2:
        return math.nan
    pts = seq[valid]
    centroid = np.nanmean(pts, axis=0)
    # Robust maximum excursion relative to the valid trajectory centroid.
    return float(np.nanpercentile(np.linalg.norm(pts - centroid[None, :], axis=1), 95) * 2.0)


def first_valid_point(seq: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    idx = np.where(valid)[0]
    if idx.size == 0:
        return None
    return seq[int(idx[0])]


def displacement_from_first(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(seq, np.nan, dtype=float)
    p0 = first_valid_point(seq, valid)
    if p0 is None:
        return out
    out[valid] = seq[valid] - p0[None, :]
    return out


def path_length(seq: np.ndarray, valid: np.ndarray) -> float:
    idx = np.where(valid)[0]
    if idx.size < 2:
        return math.nan
    diffs = np.diff(seq[idx], axis=0)
    return float(np.nansum(np.linalg.norm(diffs, axis=1)))


def speed_metrics(seq: np.ndarray, valid: np.ndarray, fs_hz: float) -> Tuple[float, float]:
    idx = np.where(valid)[0]
    if idx.size < 2:
        return math.nan, math.nan
    diffs = np.diff(seq[idx], axis=0)
    dt_frames = np.diff(idx).astype(float)
    dt_frames[dt_frames <= 0] = 1.0
    speeds = np.linalg.norm(diffs, axis=1) * fs_hz / dt_frames
    return safe_mean(speeds), float(np.nanmax(speeds)) if np.isfinite(speeds).any() else math.nan


def direction_metrics(obs: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    idx = np.where(valid)[0]
    if idx.size < 2:
        return math.nan, math.nan
    obs_d = np.diff(obs[idx], axis=0)
    ref_d = np.diff(ref[idx], axis=0)
    obs_norm = np.linalg.norm(obs_d, axis=1)
    ref_norm = np.linalg.norm(ref_d, axis=1)
    good = (obs_norm > EPS) & (ref_norm > EPS) & np.isfinite(obs_norm) & np.isfinite(ref_norm)
    if good.sum() == 0:
        return math.nan, math.nan
    dot = np.sum(obs_d[good] * ref_d[good], axis=1)
    cos = dot / (obs_norm[good] * ref_norm[good] + EPS)
    cos = np.clip(cos, -1.0, 1.0)
    total_obs = np.sum(obs_norm[good]) + EPS
    # Opposite-to-reference movement weighted by observed step length.
    counter = np.sum(obs_norm[good] * np.maximum(0.0, -cos)) / total_obs
    return safe_mean(cos), float(counter)


def marker_metrics(obs: np.ndarray, ref: np.ndarray, valid: np.ndarray, fs_hz: float) -> Dict[str, float]:
    if valid.sum() < 2:
        return {k: math.nan for k in [
            "valid_fraction", "amplitude_observed", "amplitude_reference",
            "amplitude_ratio_obs_over_ref", "amplitude_log_ratio_obs_over_ref",
            "amplitude_log_ratio_abs", "hypokinesia_proxy", "hyperkinesia_proxy",
            "path_length_observed", "path_length_reference", "path_length_ratio_obs_over_ref",
            "mean_speed_observed", "mean_speed_reference", "peak_speed_observed",
            "peak_speed_reference", "trajectory_mae", "trajectory_rmse",
            "displacement_mae", "displacement_rmse", "direction_cosine_mean",
            "counter_direction_ratio",
        ]}

    obs_v = obs[valid]
    ref_v = ref[valid]
    diff = obs_v - ref_v
    dist = np.linalg.norm(diff, axis=1)

    obs_disp = displacement_from_first(obs, valid)
    ref_disp = displacement_from_first(ref, valid)
    disp_diff = obs_disp[valid] - ref_disp[valid]
    disp_dist = np.linalg.norm(disp_diff, axis=1)

    amp_o = trajectory_amplitude(obs, valid)
    amp_r = trajectory_amplitude(ref, valid)
    amp_ratio = (amp_o + EPS) / (amp_r + EPS) if np.isfinite(amp_o) and np.isfinite(amp_r) else math.nan
    log_ratio = math.log(amp_ratio) if np.isfinite(amp_ratio) and amp_ratio > 0 else math.nan

    pl_o = path_length(obs, valid)
    pl_r = path_length(ref, valid)
    pl_ratio = (pl_o + EPS) / (pl_r + EPS) if np.isfinite(pl_o) and np.isfinite(pl_r) else math.nan
    mean_speed_o, peak_speed_o = speed_metrics(obs, valid, fs_hz)
    mean_speed_r, peak_speed_r = speed_metrics(ref, valid, fs_hz)
    cos_mean, counter = direction_metrics(obs, ref, valid)

    return {
        "valid_fraction": float(valid.mean()),
        "amplitude_observed": amp_o,
        "amplitude_reference": amp_r,
        "amplitude_ratio_obs_over_ref": amp_ratio,
        "amplitude_log_ratio_obs_over_ref": log_ratio,
        "amplitude_log_ratio_abs": abs(log_ratio) if np.isfinite(log_ratio) else math.nan,
        "hypokinesia_proxy": max(0.0, -log_ratio) if np.isfinite(log_ratio) else math.nan,
        "hyperkinesia_proxy": max(0.0, log_ratio) if np.isfinite(log_ratio) else math.nan,
        "path_length_observed": pl_o,
        "path_length_reference": pl_r,
        "path_length_ratio_obs_over_ref": pl_ratio,
        "mean_speed_observed": mean_speed_o,
        "mean_speed_reference": mean_speed_r,
        "peak_speed_observed": peak_speed_o,
        "peak_speed_reference": peak_speed_r,
        "trajectory_mae": safe_mean(dist),
        "trajectory_rmse": float(np.sqrt(np.nanmean(dist ** 2))) if np.isfinite(dist).any() else math.nan,
        "displacement_mae": safe_mean(disp_dist),
        "displacement_rmse": float(np.sqrt(np.nanmean(disp_dist ** 2))) if np.isfinite(disp_dist).any() else math.nan,
        "direction_cosine_mean": cos_mean,
        "counter_direction_ratio": counter,
    }


def aggregate_metrics(marker_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_cols = [c for c in marker_df.columns if c in marker_metrics_columns()]
    for keys, g in marker_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_markers"] = int(g["marker_id"].nunique()) if "marker_id" in g.columns else len(g)
        for metric in metric_cols:
            row[metric] = safe_median(g[metric].values)
        rows.append(row)
    return pd.DataFrame(rows)


def marker_metrics_columns() -> List[str]:
    return [
        "valid_fraction",
        "amplitude_observed",
        "amplitude_reference",
        "amplitude_ratio_obs_over_ref",
        "amplitude_log_ratio_obs_over_ref",
        "amplitude_log_ratio_abs",
        "hypokinesia_proxy",
        "hyperkinesia_proxy",
        "path_length_observed",
        "path_length_reference",
        "path_length_ratio_obs_over_ref",
        "mean_speed_observed",
        "mean_speed_reference",
        "peak_speed_observed",
        "peak_speed_reference",
        "trajectory_mae",
        "trajectory_rmse",
        "displacement_mae",
        "displacement_rmse",
        "direction_cosine_mean",
        "counter_direction_ratio",
    ]


def load_npz_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    missing = [k for k in [OBS_ARRAY, AEV3_ARRAY, MASK_ARRAY] if k not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    obs = np.asarray(data[OBS_ARRAY], dtype=float)
    ref = np.asarray(data[AEV3_ARRAY], dtype=float)
    mask = np.asarray(data[MASK_ARRAY]).astype(bool)
    if obs.shape != ref.shape:
        raise ValueError(f"Shape mismatch in {path}: observed {obs.shape}, AEv3 {ref.shape}")
    if obs.ndim != 3 or obs.shape[-1] != 3:
        raise ValueError(f"Expected observed_face shape (T,M,3), got {obs.shape} in {path}")
    if mask.shape != obs.shape[:2]:
        raise ValueError(f"Expected target_mask shape {obs.shape[:2]}, got {mask.shape} in {path}")
    return obs, ref, mask


def process_sample(
    record: SampleRecord,
    labels: pd.DataFrame,
    fs_hz: float,
    lowpass_hz: float,
    filter_order: int,
    no_filter: bool,
    min_valid_fraction_marker: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    obs, ref, mask = load_npz_arrays(record.npz_path)
    if not no_filter:
        obs_f = smooth_sequence(obs, mask, fs_hz=fs_hz, lowpass_hz=lowpass_hz, filter_order=filter_order)
        ref_f = smooth_sequence(ref, mask, fs_hz=fs_hz, lowpass_hz=lowpass_hz, filter_order=filter_order)
    else:
        obs_f, ref_f = obs, ref
    valid_all = valid_mask_for(obs_f, ref_f, mask)
    T, M, _ = obs_f.shape
    if labels.shape[0] != M:
        raise ValueError(f"Semantic labels contain {labels.shape[0]} markers, but NPZ contains {M} markers")

    base = {
        "sample_id": record.sample_id,
        "participant_id": record.participant_id,
        "acquisition_date": record.acquisition_date.date().isoformat() if record.acquisition_date is not None else "",
        "movement": record.movement,
        "condition": record.condition,
        "group": record.group,
        "filename": record.filename,
        "complete_filepath": record.complete_filepath,
        "npz_path": str(record.npz_path),
        "date_source": record.date_source,
        "alignment_score": record.alignment_score,
        "qc_valid_marker_frame_fraction": record.qc_valid_marker_frame_fraction,
        "T": T,
        "n_markers_total": M,
    }

    rows: List[Dict[str, object]] = []
    for marker_id in range(M):
        valid = valid_all[:, marker_id]
        mm = marker_metrics(obs_f[:, marker_id, :], ref_f[:, marker_id, :], valid, fs_hz=fs_hz)
        if np.isfinite(mm.get("valid_fraction", math.nan)) and mm["valid_fraction"] < min_valid_fraction_marker:
            # Keep an audit row but nullify metrics except valid_fraction.
            valid_fraction = mm["valid_fraction"]
            mm = {k: math.nan for k in marker_metrics_columns()}
            mm["valid_fraction"] = valid_fraction
        label_row = labels.iloc[marker_id]
        row = dict(base)
        row.update(
            {
                "marker_id": marker_id,
                "region": label_row.get("region", "unknown"),
                "side": label_row.get("side", "unknown"),
                "label": label_row.get("label", f"marker_{marker_id}"),
            }
        )
        row.update(mm)
        rows.append(row)

    sample_valid = valid_all.astype(bool)
    sample_summary = dict(base)
    sample_summary["n_valid_marker_frames"] = int(sample_valid.sum())
    sample_summary["valid_marker_frame_fraction"] = float(sample_valid.mean())
    sample_summary["n_markers_with_any_valid_data"] = int((sample_valid.sum(axis=0) > 0).sum())
    return rows, sample_summary


def summarize_sample_from_marker_rows(marker_rows: List[Dict[str, object]]) -> Dict[str, object]:
    df = pd.DataFrame(marker_rows)
    if df.empty:
        return {}
    meta_cols = [
        "sample_id", "participant_id", "acquisition_date", "movement", "condition", "group",
        "filename", "complete_filepath", "npz_path", "date_source", "alignment_score",
        "qc_valid_marker_frame_fraction", "T", "n_markers_total",
    ]
    out = {c: df[c].iloc[0] for c in meta_cols if c in df.columns}
    out["n_markers_used"] = int(df.loc[df["valid_fraction"].fillna(0) > 0, "marker_id"].nunique())
    for metric in marker_metrics_columns():
        out[metric] = safe_median(df[metric].values)
    return out


def filter_followup_participants(sample_df: pd.DataFrame, min_dates: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    audit = []
    keep_ids = []
    for pid, g in sample_df.groupby("participant_id"):
        dates = sorted(d for d in g["acquisition_date"].dropna().unique().tolist() if str(d) != "")
        audit.append(
            {
                "participant_id": pid,
                "n_dates": len(dates),
                "dates": ";".join(map(str, dates)),
                "n_samples": len(g),
                "movements": ";".join(sorted(map(str, g["movement"].dropna().unique()))),
            }
        )
        if len(dates) >= min_dates:
            keep_ids.append(pid)
    audit_df = pd.DataFrame(audit).sort_values(["n_dates", "participant_id"], ascending=[False, True])
    out = sample_df[sample_df["participant_id"].isin(keep_ids)].copy()
    return out, audit_df


def aggregate_visits(sample_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Per participant-date-movement median across repetitions/samples.
    metric_cols = [c for c in CORE_METRICS if c in sample_df.columns]
    visit_move_rows = []
    group_cols_move = ["participant_id", "acquisition_date", "movement"]
    for keys, g in sample_df.groupby(group_cols_move, dropna=False):
        pid, date, movement = keys
        row = {
            "participant_id": pid,
            "acquisition_date": date,
            "movement": movement,
            "n_samples": len(g),
            "n_unique_samples": int(g["sample_id"].nunique()),
        }
        for metric in metric_cols:
            row[metric] = safe_median(g[metric].values)
        visit_move_rows.append(row)
    visit_movement = pd.DataFrame(visit_move_rows).sort_values(group_cols_move)

    visit_global_rows = []
    for keys, g in sample_df.groupby(["participant_id", "acquisition_date"], dropna=False):
        pid, date = keys
        row = {
            "participant_id": pid,
            "acquisition_date": date,
            "n_samples": len(g),
            "n_movements": int(g["movement"].nunique()),
            "movements": ";".join(sorted(map(str, g["movement"].dropna().unique()))),
        }
        for metric in metric_cols:
            row[metric] = safe_median(g[metric].values)
        visit_global_rows.append(row)
    visit_global = pd.DataFrame(visit_global_rows).sort_values(["participant_id", "acquisition_date"])
    return visit_movement, visit_global


def movement_aggregate(values: Sequence[float], method: str = "mean") -> float:
    """Aggregate one value per movement into a visit-level value.

    The important point is that each movement has equal weight, regardless of
    the number of repetitions/samples acquired for that movement.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if method == "median":
        return float(np.nanmedian(arr))
    return float(np.nanmean(arr))


def common_movements_by_participant(
    visit_movement: pd.DataFrame,
    min_dates: int,
    mode: str = "all_dates",
) -> pd.DataFrame:
    """Return the movement set retained for each participant.

    mode='all_dates' keeps only movements present at every retained date for the
    participant. This is the strictest and recommended setting for the report.
    mode='baseline_last' keeps movements shared between the first and last dates;
    intermediate dates are then evaluated only for those movements if available.
    """
    rows: List[Dict[str, object]] = []
    if visit_movement.empty:
        return pd.DataFrame()

    vm = visit_movement.copy()
    vm["date_dt"] = pd.to_datetime(vm["acquisition_date"], errors="coerce")
    vm = vm.dropna(subset=["date_dt"])

    for pid, g in vm.groupby("participant_id"):
        date_values = sorted(g["acquisition_date"].dropna().unique().tolist())
        if len(date_values) < min_dates:
            continue
        by_date = {
            d: set(g.loc[g["acquisition_date"] == d, "movement"].astype(str).tolist())
            for d in date_values
        }
        if mode == "baseline_last" and len(date_values) >= 2:
            common = by_date[date_values[0]].intersection(by_date[date_values[-1]])
        else:
            common = set.intersection(*by_date.values()) if by_date else set()
        rows.append(
            {
                "participant_id": pid,
                "n_dates": len(date_values),
                "dates": ";".join(map(str, date_values)),
                "mode": mode,
                "n_common_movements": len(common),
                "common_movements": ";".join(sorted(common)),
                "all_date_movements": " | ".join(f"{d}:{';'.join(sorted(ms))}" for d, ms in by_date.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_dates", "participant_id"], ascending=[False, True])


def aggregate_visits_movement_matched(
    visit_movement: pd.DataFrame,
    common_df: pd.DataFrame,
    movement_agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate movement-level metrics into visit-level metrics after matching movements."""
    if visit_movement.empty or common_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in CORE_METRICS if c in visit_movement.columns]
    common_map: Dict[str, set] = {}
    for _, row in common_df.iterrows():
        ms = [m for m in str(row.get("common_movements", "")).split(";") if m]
        if ms:
            common_map[str(row["participant_id"])] = set(ms)

    rows: List[Dict[str, object]] = []
    for keys, g in visit_movement.groupby(["participant_id", "acquisition_date"], dropna=False):
        pid, date = keys
        retained = common_map.get(str(pid), set())
        if not retained:
            continue
        gg = g[g["movement"].astype(str).isin(retained)].copy()
        if gg.empty:
            continue
        row = {
            "participant_id": pid,
            "acquisition_date": date,
            "n_samples": int(gg["n_samples"].sum()) if "n_samples" in gg.columns else math.nan,
            "n_movements": int(gg["movement"].nunique()),
            "n_common_movements": len(retained),
            "movements": ";".join(sorted(gg["movement"].astype(str).unique())),
            "common_movements": ";".join(sorted(retained)),
            "movement_aggregation": movement_agg,
        }
        for metric in metric_cols:
            row[metric] = movement_aggregate(gg[metric].values, method=movement_agg)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["participant_id", "acquisition_date"])


def aggregate_anatomy_movement_first(
    anatomy_sample_df: pd.DataFrame,
    anatomy_cols: List[str],
    common_df: pd.DataFrame,
    movement_agg: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Movement-first anatomical aggregation.

    First, samples/repetitions are summarized per participant-date-movement-anatomy.
    Then only participant-specific common movements are retained, and visit-level
    anatomical metrics are computed by equal movement weighting.
    """
    if anatomy_sample_df.empty or common_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [c for c in CORE_METRICS if c in anatomy_sample_df.columns]
    group_cols = ["participant_id", "acquisition_date", "movement"] + anatomy_cols
    movement_rows: List[Dict[str, object]] = []
    for keys, g in anatomy_sample_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_samples"] = len(g)
        if "n_markers" in g.columns:
            row["n_markers_median"] = safe_median(g["n_markers"].values)
        for metric in metric_cols:
            row[metric] = safe_median(g[metric].values)
        movement_rows.append(row)
    movement_df = pd.DataFrame(movement_rows)
    if movement_df.empty:
        return movement_df, pd.DataFrame()

    common_map: Dict[str, set] = {}
    for _, row in common_df.iterrows():
        ms = [m for m in str(row.get("common_movements", "")).split(";") if m]
        if ms:
            common_map[str(row["participant_id"])] = set(ms)

    visit_rows: List[Dict[str, object]] = []
    visit_group_cols = ["participant_id", "acquisition_date"] + anatomy_cols
    for keys, g in movement_df.groupby(visit_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pid = str(keys[0])
        retained = common_map.get(pid, set())
        if not retained:
            continue
        gg = g[g["movement"].astype(str).isin(retained)].copy()
        if gg.empty:
            continue
        row = {col: val for col, val in zip(visit_group_cols, keys)}
        row["n_samples"] = int(gg["n_samples"].sum()) if "n_samples" in gg.columns else math.nan
        row["n_movements"] = int(gg["movement"].nunique())
        row["n_common_movements"] = len(retained)
        row["movements"] = ";".join(sorted(gg["movement"].astype(str).unique()))
        row["common_movements"] = ";".join(sorted(retained))
        row["movement_aggregation"] = movement_agg
        if "n_markers_median" in gg.columns:
            row["n_markers_median"] = safe_median(gg["n_markers_median"].values)
        for metric in metric_cols:
            row[metric] = movement_aggregate(gg[metric].values, method=movement_agg)
        visit_rows.append(row)
    visit_df = pd.DataFrame(visit_rows)
    if not visit_df.empty:
        visit_df = visit_df.sort_values(visit_group_cols)
    return movement_df, visit_df


def compute_change_from_baseline(visit_df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if visit_df.empty:
        return pd.DataFrame()
    for pid, g in visit_df.sort_values("acquisition_date").groupby("participant_id"):
        g = g.sort_values("acquisition_date").reset_index(drop=True)
        baseline = g.iloc[0]
        baseline_date = baseline["acquisition_date"]
        for _, row in g.iterrows():
            out = {
                "participant_id": pid,
                "acquisition_date": row["acquisition_date"],
                "baseline_date": baseline_date,
                "days_from_baseline": (
                    pd.to_datetime(row["acquisition_date"]) - pd.to_datetime(baseline_date)
                ).days if str(row["acquisition_date"]) and str(baseline_date) else math.nan,
                "n_samples": row.get("n_samples", math.nan),
                "n_movements": row.get("n_movements", math.nan),
                "movements": row.get("movements", ""),
            }
            for metric in metric_cols:
                current = safe_float(row.get(metric, math.nan))
                base = safe_float(baseline.get(metric, math.nan))
                delta = current - base if np.isfinite(current) and np.isfinite(base) else math.nan
                rel = 100.0 * delta / (abs(base) + EPS) if np.isfinite(delta) and np.isfinite(base) else math.nan
                if metric in LOWER_IS_BETTER:
                    lower_better = LOWER_IS_BETTER[metric]
                    improvement = -delta if lower_better else delta
                    improvement_pct = -rel if lower_better else rel
                else:
                    improvement = -delta
                    improvement_pct = -rel
                out[f"{metric}_baseline"] = base
                out[f"{metric}_current"] = current
                out[f"{metric}_delta"] = delta
                out[f"{metric}_relative_delta_pct"] = rel
                out[f"{metric}_improvement"] = improvement
                out[f"{metric}_improvement_pct"] = improvement_pct
            rows.append(out)
    return pd.DataFrame(rows)


def participant_summary(change_df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    if change_df.empty:
        return pd.DataFrame()
    for pid, g in change_df.sort_values("acquisition_date").groupby("participant_id"):
        first = g.iloc[0]
        last = g.iloc[-1]
        row = {
            "participant_id": pid,
            "first_date": first["acquisition_date"],
            "last_date": last["acquisition_date"],
            "n_dates": int(g["acquisition_date"].nunique()),
            "days_followup": safe_float(last.get("days_from_baseline", math.nan)),
            "last_n_samples": last.get("n_samples", math.nan),
            "last_n_movements": last.get("n_movements", math.nan),
            "last_movements": last.get("movements", ""),
        }
        for metric in metric_cols:
            row[f"{metric}_baseline"] = last.get(f"{metric}_baseline", math.nan)
            row[f"{metric}_last"] = last.get(f"{metric}_current", math.nan)
            row[f"{metric}_delta_last_vs_baseline"] = last.get(f"{metric}_delta", math.nan)
            row[f"{metric}_improvement_last_vs_baseline"] = last.get(f"{metric}_improvement", math.nan)
            row[f"{metric}_improvement_pct_last_vs_baseline"] = last.get(f"{metric}_improvement_pct", math.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("participant_id")


def plot_visit_lines(visit_global: pd.DataFrame, metric_cols: Sequence[str], out_dir: Path) -> None:
    if plt is None or visit_global.empty:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    for metric in metric_cols:
        if metric not in visit_global.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        any_plot = False
        for pid, g in visit_global.sort_values("acquisition_date").groupby("participant_id"):
            g = g.copy()
            g["date_dt"] = pd.to_datetime(g["acquisition_date"], errors="coerce")
            g = g.dropna(subset=["date_dt", metric])
            if len(g) < 2:
                continue
            ax.plot(g["date_dt"], g[metric], marker="o", label=str(pid))
            any_plot = True
        if not any_plot:
            plt.close(fig)
            continue
        ax.set_title(f"AE-v3 follow-up: {metric}")
        ax.set_xlabel("Acquisition date")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / f"followup_global_{metric}.png", dpi=250)
        fig.savefig(fig_dir / f"followup_global_{metric}.pdf")
        plt.close(fig)


def plot_region_side_heatmaps(region_side_df: pd.DataFrame, out_dir: Path, metric: str = "trajectory_mae") -> None:
    if plt is None or region_side_df.empty or metric not in region_side_df.columns:
        return
    fig_dir = out_dir / "figures" / "region_side_heatmaps"
    ensure_dir(fig_dir)
    df = region_side_df.copy()
    df["region_side"] = df["region"].astype(str) + " | " + df["side"].astype(str)
    # Aggregate repetitions/movements per date for each region-side.
    agg = (
        df.groupby(["participant_id", "acquisition_date", "region_side"], dropna=False)[metric]
        .median()
        .reset_index()
    )
    for pid, g in agg.groupby("participant_id"):
        dates = sorted(g["acquisition_date"].dropna().unique().tolist())
        if len(dates) < 2:
            continue
        # Keep the top 20 region-side rows by maximum abnormality for readable figures.
        top_regions = (
            g.groupby("region_side")[metric]
            .max()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )
        mat = (
            g[g["region_side"].isin(top_regions)]
            .pivot_table(index="region_side", columns="acquisition_date", values=metric, aggfunc="median")
            .reindex(top_regions)
        )
        if mat.empty:
            continue
        fig_h = max(5, 0.35 * len(mat.index) + 2)
        fig_w = max(6, 0.8 * len(mat.columns) + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat.values, aspect="auto")
        ax.set_title(f"{pid}: AE-v3 region-side {metric}")
        ax.set_xlabel("Acquisition date")
        ax.set_ylabel("Region | side")
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        safe_pid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pid))
        fig.savefig(fig_dir / f"{safe_pid}_region_side_{metric}.png", dpi=250)
        fig.savefig(fig_dir / f"{safe_pid}_region_side_{metric}.pdf")
        plt.close(fig)


def write_report(
    out_dir: Path,
    audit_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    visit_global: pd.DataFrame,
    participant_df: pd.DataFrame,
    min_dates: int,
) -> None:
    lines = []
    lines.append("AE-v3 movement-matched follow-up evaluation")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"Minimum dates required: {min_dates}")
    lines.append(f"Pathological overlay rows inspected: {len(audit_df)}")
    lines.append(f"Rows with existing NPZ files: {int(audit_df['npz_exists'].sum()) if 'npz_exists' in audit_df else 'NA'}")
    lines.append(f"Rows with acquisition dates: {int((audit_df['acquisition_date'].astype(str) != '').sum()) if 'acquisition_date' in audit_df else 'NA'}")
    lines.append(f"Follow-up samples retained: {len(sample_df)}")
    lines.append(f"Follow-up participants retained: {sample_df['participant_id'].nunique() if not sample_df.empty else 0}")
    lines.append(f"Participant-date visits retained: {len(visit_global)}")
    lines.append("")
    lines.append("Available dates by participant:")
    if not audit_df.empty:
        tmp = audit_df.groupby("participant_id").agg(
            n_dates=("acquisition_date", lambda x: len(set(v for v in x if str(v) != ""))),
            n_rows=("sample_id", "count"),
        ).reset_index().sort_values(["n_dates", "participant_id"], ascending=[False, True])
        for _, row in tmp.iterrows():
            lines.append(f"  {row['participant_id']}: {int(row['n_dates'])} dates, {int(row['n_rows'])} rows")
    lines.append("")
    lines.append("Last visit versus baseline, selected metrics:")
    if participant_df.empty:
        lines.append("  No participants passed the follow-up inclusion criteria.")
    else:
        for _, row in participant_df.iterrows():
            pid = row["participant_id"]
            lines.append(f"  {pid}: {row['first_date']} -> {row['last_date']} ({row['n_dates']} dates)")
            for metric in ["trajectory_mae", "amplitude_log_ratio_abs", "hypokinesia_proxy", "hyperkinesia_proxy", "counter_direction_ratio"]:
                col = f"{metric}_improvement_pct_last_vs_baseline"
                if col in row and np.isfinite(safe_float(row[col])):
                    lines.append(f"    {metric}: improvement_pct={safe_float(row[col]):.2f}")
    (out_dir / "followup_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay_root", default="align_mean_movement_refactor/four_cloud_eval_overlays")
    ap.add_argument("--overlay_index", default=None, help="Default: <overlay_root>/overlay_index.csv")
    ap.add_argument("--metadata", default="facemocap_metadata_reference_split.csv", help="Optional metadata CSV containing acquisition_date.")
    ap.add_argument("--semantic_labels", default="semantic_facial_labels.csv")
    ap.add_argument("--out_dir", default="align_mean_movement_refactor/aev3_followup_eval_matched")
    ap.add_argument("--condition", default="pathological")
    ap.add_argument("--min_dates", type=int, default=2, help="Minimum distinct acquisition dates per participant. Use 3 for >2 dates.")
    ap.add_argument("--common_movement_mode", choices=["all_dates", "baseline_last"], default="all_dates", help="How to define the participant-specific common movement set.")
    ap.add_argument("--movement_agg", choices=["mean", "median"], default="mean", help="How to combine one value per retained movement into a visit-level score. Mean gives explicit equal movement weighting.")
    ap.add_argument("--fs_hz", type=float, default=100.0)
    ap.add_argument("--lowpass_hz", type=float, default=8.0)
    ap.add_argument("--filter_order", type=int, default=2)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--min_valid_fraction_marker", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_figures", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    overlay_root = Path(args.overlay_root)
    overlay_index = Path(args.overlay_index) if args.overlay_index else overlay_root / "overlay_index.csv"
    metadata_path = Path(args.metadata) if args.metadata and Path(args.metadata).exists() else None
    semantic_path = Path(args.semantic_labels)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records, audit_df = load_overlay_records(
        overlay_index_path=overlay_index,
        overlay_root=overlay_root,
        metadata_path=metadata_path,
        condition=args.condition,
    )
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    audit_df.to_csv(out_dir / "followup_input_audit.csv", index=False)

    # Keep only records that can actually be evaluated and have dates.
    eval_records = [r for r in records if r.npz_path.exists() and r.acquisition_date is not None]
    print(f"Pathological rows in overlay index: {len(records)}")
    print(f"Rows with existing NPZ and acquisition date: {len(eval_records)}")
    if not eval_records:
        print("No evaluable follow-up records found. Check NPZ paths and acquisition dates.")
        return

    # Determine marker count from first NPZ and load labels.
    obs0, ref0, mask0 = load_npz_arrays(eval_records[0].npz_path)
    labels = load_semantic_labels(semantic_path, obs0.shape[1])

    all_marker_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for i, rec in enumerate(eval_records, start=1):
        print(f"[{i}/{len(eval_records)}] {rec.participant_id} {rec.acquisition_date.date()} {rec.movement} {rec.sample_id}")
        try:
            marker_rows, sample_info = process_sample(
                rec,
                labels=labels,
                fs_hz=args.fs_hz,
                lowpass_hz=args.lowpass_hz,
                filter_order=args.filter_order,
                no_filter=args.no_filter,
                min_valid_fraction_marker=args.min_valid_fraction_marker,
            )
            all_marker_rows.extend(marker_rows)
            sample_summary = summarize_sample_from_marker_rows(marker_rows)
            sample_summary.update(sample_info)
            sample_rows.append(sample_summary)
        except Exception as exc:
            failures.append(
                {
                    "participant_id": rec.participant_id,
                    "acquisition_date": rec.acquisition_date.date().isoformat() if rec.acquisition_date is not None else "",
                    "movement": rec.movement,
                    "sample_id": rec.sample_id,
                    "npz_path": str(rec.npz_path),
                    "error": str(exc),
                }
            )
            print(f"  [FAIL] {exc}")

    marker_df = pd.DataFrame(all_marker_rows)
    sample_df_all = pd.DataFrame(sample_rows)
    pd.DataFrame(failures).to_csv(out_dir / "followup_failures.csv", index=False)

    if marker_df.empty or sample_df_all.empty:
        print("No metrics were computed.")
        return

    # Keep only participants with enough distinct dates.
    sample_df, available_dates = filter_followup_participants(sample_df_all, min_dates=args.min_dates)
    keep_ids = set(sample_df["participant_id"].unique()) if not sample_df.empty else set()
    marker_df = marker_df[marker_df["participant_id"].isin(keep_ids)].copy()

    sample_df_all.to_csv(out_dir / "all_pathological_sample_metrics_before_followup_filter.csv", index=False)
    available_dates.to_csv(out_dir / "followup_available_dates.csv", index=False)
    sample_df.to_csv(out_dir / "followup_sample_metrics.csv", index=False)
    marker_df.to_csv(out_dir / "followup_marker_metrics.csv", index=False)

    if sample_df.empty:
        print(f"No participants had at least {args.min_dates} acquisition dates.")
        write_report(out_dir, audit_df, sample_df, pd.DataFrame(), pd.DataFrame(), min_dates=args.min_dates)
        return

    # Anatomical sample-level aggregations retained for traceability.
    id_cols = [
        "sample_id", "participant_id", "acquisition_date", "movement", "condition", "group",
        "filename", "complete_filepath", "npz_path", "date_source",
    ]
    region_df = aggregate_metrics(marker_df, id_cols + ["region"])
    side_df = aggregate_metrics(marker_df, id_cols + ["side"])
    region_side_df = aggregate_metrics(marker_df, id_cols + ["region", "side"])
    region_df.to_csv(out_dir / "followup_region_metrics_sample_level.csv", index=False)
    side_df.to_csv(out_dir / "followup_side_metrics_sample_level.csv", index=False)
    region_side_df.to_csv(out_dir / "followup_region_side_metrics_sample_level.csv", index=False)

    # First summarize repetitions/samples per participant-date-movement.
    # Then retain only participant-specific common movements and aggregate with
    # equal movement weighting. These matched outputs are the report-ready ones.
    visit_movement, visit_global_unmatched = aggregate_visits(sample_df)
    visit_movement.to_csv(out_dir / "followup_visit_movement_metrics.csv", index=False)
    visit_global_unmatched.to_csv(out_dir / "followup_visit_global_metrics_unmatched.csv", index=False)

    common_df = common_movements_by_participant(
        visit_movement,
        min_dates=args.min_dates,
        mode=args.common_movement_mode,
    )
    common_df.to_csv(out_dir / "followup_common_movements_by_participant.csv", index=False)

    visit_global = aggregate_visits_movement_matched(
        visit_movement,
        common_df=common_df,
        movement_agg=args.movement_agg,
    )
    visit_global.to_csv(out_dir / "followup_visit_global_metrics.csv", index=False)

    # Remove participants that lost all movements after common-movement matching.
    matched_ids = set(visit_global["participant_id"].unique()) if not visit_global.empty else set()
    sample_df = sample_df[sample_df["participant_id"].isin(matched_ids)].copy()
    marker_df = marker_df[marker_df["participant_id"].isin(matched_ids)].copy()

    # Matched anatomical visit-level outputs. These are the preferred outputs for
    # regional hypokinesia/hyperkinesia interpretation.
    region_movement_df, region_visit_df = aggregate_anatomy_movement_first(
        region_df[region_df["participant_id"].isin(matched_ids)].copy(),
        anatomy_cols=["region"],
        common_df=common_df,
        movement_agg=args.movement_agg,
    )
    side_movement_df, side_visit_df = aggregate_anatomy_movement_first(
        side_df[side_df["participant_id"].isin(matched_ids)].copy(),
        anatomy_cols=["side"],
        common_df=common_df,
        movement_agg=args.movement_agg,
    )
    region_side_movement_df, region_side_visit_df = aggregate_anatomy_movement_first(
        region_side_df[region_side_df["participant_id"].isin(matched_ids)].copy(),
        anatomy_cols=["region", "side"],
        common_df=common_df,
        movement_agg=args.movement_agg,
    )
    region_movement_df.to_csv(out_dir / "followup_region_movement_metrics.csv", index=False)
    side_movement_df.to_csv(out_dir / "followup_side_movement_metrics.csv", index=False)
    region_side_movement_df.to_csv(out_dir / "followup_region_side_movement_metrics.csv", index=False)
    region_visit_df.to_csv(out_dir / "followup_region_visit_metrics.csv", index=False)
    side_visit_df.to_csv(out_dir / "followup_side_visit_metrics.csv", index=False)
    region_side_visit_df.to_csv(out_dir / "followup_region_side_metrics.csv", index=False)

    metric_cols = [m for m in CORE_METRICS if m in visit_global.columns]
    change_df = compute_change_from_baseline(visit_global, metric_cols)
    change_df.to_csv(out_dir / "followup_change_from_baseline.csv", index=False)
    participant_df = participant_summary(change_df, metric_cols)
    participant_df.to_csv(out_dir / "followup_participant_summary.csv", index=False)

    # Movement-specific baseline change remains useful as a diagnostic file.
    # It is computed from movement-level metrics, before visit-level aggregation.
    movement_change_parts = []
    if not common_df.empty:
        common_map = {
            str(row["participant_id"]): set(m for m in str(row.get("common_movements", "")).split(";") if m)
            for _, row in common_df.iterrows()
        }
        vm_common_parts = []
        for pid, g in visit_movement.groupby("participant_id"):
            retained = common_map.get(str(pid), set())
            if retained:
                vm_common_parts.append(g[g["movement"].astype(str).isin(retained)].copy())
        visit_movement_common = pd.concat(vm_common_parts, ignore_index=True) if vm_common_parts else pd.DataFrame()
    else:
        visit_movement_common = pd.DataFrame()

    if not visit_movement_common.empty:
        for movement, g in visit_movement_common.groupby("movement"):
            ch = compute_change_from_baseline(g, [m for m in CORE_METRICS if m in g.columns])
            if not ch.empty:
                ch.insert(2, "movement", movement)
                movement_change_parts.append(ch)
    movement_change = pd.concat(movement_change_parts, ignore_index=True) if movement_change_parts else pd.DataFrame()
    movement_change.to_csv(out_dir / "followup_change_from_baseline_by_movement.csv", index=False)

    if not args.no_figures:
        plot_visit_lines(visit_global, ["trajectory_mae", "amplitude_log_ratio_abs", "hypokinesia_proxy", "hyperkinesia_proxy", "counter_direction_ratio"], out_dir)
        plot_region_side_heatmaps(region_side_visit_df, out_dir, metric="trajectory_mae")
        plot_region_side_heatmaps(region_side_visit_df, out_dir, metric="hypokinesia_proxy")
        plot_region_side_heatmaps(region_side_visit_df, out_dir, metric="hyperkinesia_proxy")

    write_report(out_dir, audit_df, sample_df, visit_global, participant_df, min_dates=args.min_dates)

    # Append matched-analysis details to the report.
    report_path = out_dir / "followup_report.txt"
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n\nMovement-matched analysis details\n")
        f.write("================================\n")
        f.write(f"Common movement mode: {args.common_movement_mode}\n")
        f.write(f"Visit-level movement aggregation: {args.movement_agg}\n")
        f.write("Only the movement-matched visit-level outputs should be used for final longitudinal interpretation.\n")
        if not common_df.empty:
            f.write("\nCommon movements retained by participant:\n")
            for _, row in common_df.iterrows():
                if str(row.get("participant_id", "")) in matched_ids:
                    f.write(f"  {row['participant_id']}: {row.get('common_movements', '')} ({row.get('n_common_movements', 0)} movements)\n")

    print(f"[OK] Wrote movement-matched follow-up evaluation to: {out_dir}")
    print("Main report-ready files:")
    for name in [
        "followup_common_movements_by_participant.csv",
        "followup_visit_global_metrics.csv",
        "followup_change_from_baseline.csv",
        "followup_participant_summary.csv",
        "followup_region_side_metrics.csv",
        "followup_report.txt",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
