#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ae_dataset_v2.py

Builds v2 autoencoder-ready FaceMoCap arrays separating neutral morphology from movement displacement.

Purpose
-------
This script prepares aligned, resampled, neutral-relative facial trajectories for
masked denoising autoencoder experiments. It uses only the metadata split to
create:

  - healthy train cohort:      condition == healthy, reference_split == reference
  - healthy evaluation cohort: condition == healthy, reference_split == evaluation
  - pathological cohort:       condition == pathological, usually reference_split == evaluation

In addition to neutral-relative displacement X and mask M, v2 exports neutral facial
geometries N so full face trajectories can be reconstructed as:

    face(t) = neutral_face + displacement(t)

Output per movement
-------------------
ae_dataset/
  M1/
    X_train_healthy.npy          float32, shape (S, T, 105, 3), zero-filled where missing
    M_train_healthy.npy          uint8,   shape (S, T, 105), 1 where all xyz are finite
    N_train_healthy.npy          float32, shape (S,105,3), target neutral face
    metadata_train_healthy.csv
    X_eval_healthy.npy
    M_eval_healthy.npy
    N_eval_healthy.npy
    metadata_eval_healthy.csv
    X_pathological.npy
    M_pathological.npy
    N_pathological.npy
    metadata_pathological.csv
    reference_displacement.npy    float32, shape (100,105,3)
    reference_envelope_per_marker.csv
  dataset_manifest.json

Important
---------
X stores neutral-relative facial displacements in a common template frame:

    X[t, i] = aligned_face[t, i] - aligned_face[neutral_idx_resampled, i]
    N[i]    = aligned_face[neutral_idx_resampled, i]

Missing values are stored as zeros in X and tracked by M. Training/evaluation
must use M to ignore originally missing points.

Recommended run
---------------
python build_ae_dataset_v2.py \
  --metadata /media/rodriguez/easystore/Data_FaceMoCap/facemocap_metadata_reference_split.csv \
  --ref_root /media/rodriguez/easystore/Data_FaceMoCap/align_mean_movement_refactor/output_healthy_ref_split_1b \
  --out_dir /media/rodriguez/easystore/Data_FaceMoCap/align_mean_movement_refactor/ae_dataset_v2 \
  --root_override /media/rodriguez/easystore/Data_FaceMoCap \
  --movements M1 M2 M3 M4 M5 \
  --try_yaw_flip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# General utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def normalize_movement(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        s = value.strip()
        if s.upper().startswith("M"):
            digits = "".join(ch for ch in s[1:] if ch.isdigit())
            return f"M{int(digits)}" if digits else s.upper()
        try:
            return f"M{int(float(s))}"
        except Exception:
            return s.upper()
    try:
        return f"M{int(float(value))}"
    except Exception:
        return str(value).strip().upper()


def safe_id_from_filepath(path: str) -> str:
    p = Path(str(path))
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)[:80]
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{h}"


def resolve_path(path_value: str, root_override: Optional[str]) -> str:
    path = str(path_value)
    if os.path.exists(path):
        return path
    if not root_override:
        return path
    marker = "Data_FaceMoCap"
    if marker in path:
        rel = path.split(marker, 1)[1].lstrip("/\\")
        cand = os.path.join(root_override, rel)
        if os.path.exists(cand):
            return cand
    cand = os.path.join(root_override, path.lstrip("/\\"))
    return cand if os.path.exists(cand) else path


def infer_recording_id(row: pd.Series) -> str:
    """Infer a stable recording/session identifier from acquisition_date/path/filename."""
    # 1) acquisition_date is best if available.
    val = row.get("acquisition_date", "")
    if pd.notna(val) and str(val).strip() and str(val).strip().lower() not in {"nan", "none", "nat"}:
        return str(val).strip()

    fp = str(row.get("complete_filepath", ""))
    filename = str(row.get("filename", Path(fp).name))

    # 2) Date-like folders or filename tokens.
    tokens = re.split(r"[/\\_\-.\s]+", fp)
    date_candidates = []
    for tok in tokens:
        if re.fullmatch(r"20\d{6}", tok):
            date_candidates.append(tok)
        elif re.fullmatch(r"20\d{2}\d{2}\d{2}\d{1,2}", tok):
            date_candidates.append(tok)
    if date_candidates:
        return date_candidates[-1]

    # 3) D1/D01 style session tokens in Mathieu-like files.
    m = re.search(r"(?:^|[_\-])D0*(\d+)(?:[_\-]|$)", filename, flags=re.IGNORECASE)
    if m:
        return f"D{int(m.group(1))}"

    # 4) Parent folder fallback.
    try:
        parent = Path(fp).parent.name
        if parent:
            return parent
    except Exception:
        pass
    return "unknown_session"


# ----------------------------
# CSV loading and geometry
# ----------------------------

@dataclass
class DentalFrame:
    R: np.ndarray
    o: np.ndarray


def load_facemocap_csv(csv_path: Path, skiprows: int = 5, drop_first_cols: int = 2, n_markers_keep: int = 108) -> np.ndarray:
    df = pd.read_csv(csv_path, skiprows=skiprows, header=None)
    if df.shape[1] <= drop_first_cols:
        raise ValueError(f"CSV has too few columns after dropping metadata columns: {csv_path}")
    arr = df.iloc[:, drop_first_cols:].to_numpy(dtype=float, copy=True)
    if arr.shape[1] % 3 != 0:
        raise ValueError(f"Coordinate columns are not a multiple of 3: {csv_path}")
    n_markers = arr.shape[1] // 3
    if n_markers < n_markers_keep:
        raise ValueError(f"Expected at least {n_markers_keep} markers, got {n_markers}: {csv_path}")
    arr = arr[:, : n_markers_keep * 3]
    return arr.reshape(arr.shape[0], n_markers_keep, 3)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def fixed_rot_xyz(deg_xyz: Tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = [math.radians(d) for d in deg_xyz]
    Rx = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), rx)
    Ry = rotation_matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), ry)
    Rz = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), rz)
    return Rz @ Ry @ Rx


def compute_dental_frame(d: np.ndarray) -> Optional[DentalFrame]:
    if d.shape != (3, 3) or not np.isfinite(d).all():
        return None
    o = d.mean(axis=0)
    x = d[1] - d[0]
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return None
    x = x / nx
    v = d[2] - d[0]
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return None
    z = z / nz
    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        return None
    y = y / ny
    R = np.stack([x, y, z], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return DentalFrame(R=R, o=o)


def world_to_dental(frames_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = frames_world.shape[0]
    out = np.full_like(frames_world, np.nan, dtype=float)
    ok = np.zeros(T, dtype=bool)
    for t in range(T):
        fr = compute_dental_frame(frames_world[t, 0:3, :])
        if fr is None:
            continue
        ok[t] = True
        out[t] = (fr.R.T @ (frames_world[t] - fr.o).T).T
    return out, ok


def valid_points(P: np.ndarray) -> np.ndarray:
    return np.isfinite(P).all(axis=-1)


def choose_neutral_idx(frames: np.ndarray, face_idx: np.ndarray, neutral_first_pct: float = 0.05) -> int:
    T = frames.shape[0]
    n_first = max(1, int(math.ceil(float(neutral_first_pct) * T)))
    sub = frames[:n_first, face_idx, :]
    counts = np.isfinite(sub).all(axis=-1).sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] == 0:
        raise ValueError("No finite facial markers available for neutral frame selection.")
    return best


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t


def apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def rms(x: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.nanmean(np.square(x))))


def trimmed_indices(residuals: np.ndarray, trim_frac: float) -> np.ndarray:
    n = residuals.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(round((1.0 - trim_frac) * n)))
    order = np.argsort(residuals)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def trimmed_ransac_rigid(P: np.ndarray, Q: np.ndarray, trials: int, subset: int, trim_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = P.shape[0]
    if n < max(3, subset):
        raise ValueError(f"Not enough correspondences for RANSAC: {n}")
    best_score = float("inf")
    best = None
    idx_all = np.arange(n)
    for _ in range(trials):
        idx = rng.choice(idx_all, size=subset, replace=False)
        try:
            R, t = kabsch(P[idx], Q[idx])
        except Exception:
            continue
        res = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
        inl = trimmed_indices(res, trim_frac)
        score = rms(res[inl])
        if score < best_score:
            best_score = score
            best = (R, t, inl)
    if best is None:
        raise RuntimeError("RANSAC failed.")
    R, t, inl = best
    R, t = kabsch(P[inl], Q[inl])
    res = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
    inl = trimmed_indices(res, trim_frac)
    return R, t, inl, rms(res[inl])


def maybe_yaw_flip(P: np.ndarray, Q: np.ndarray, R: np.ndarray, t: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        res = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
        return R, t, False, rms(res)
    ax = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}[axis]
    Rflip = rotation_matrix_from_axis_angle(ax, math.pi)
    res_a = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
    score_a = rms(res_a)
    Pf = (Rflip @ P.T).T
    res_b = np.linalg.norm(apply_rt(Pf, R, t) - Q, axis=1)
    score_b = rms(res_b)
    if score_b < score_a:
        return R @ Rflip, t, True, score_b
    return R, t, False, score_a


def displacement_energy(frames: np.ndarray, neutral_idx: int, face_idx: np.ndarray) -> np.ndarray:
    P0 = frames[neutral_idx, face_idx, :]
    P = frames[:, face_idx, :]
    with np.errstate(all="ignore"):
        d = P - P0[None, :, :]
        mag = np.sqrt(np.nansum(d ** 2, axis=-1))
        return np.nanmean(mag, axis=1)


def extract_active_window(E: np.ndarray, thr_percentile: float, min_len: int, max_gap: int = 5) -> Tuple[int, int]:
    finite = np.isfinite(E)
    if finite.sum() == 0:
        return 0, len(E)
    thr = np.nanpercentile(E, thr_percentile)
    active = (E >= thr) & finite
    if not active.any():
        return 0, len(E)

    # Fill short gaps.
    idx = np.where(active)[0]
    active2 = active.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < (b - a) <= max_gap + 1:
            active2[a:b + 1] = True
    active = active2
    idx = np.where(active)[0]

    peak = int(np.nanargmax(E))
    segments = []
    s = int(idx[0])
    prev = int(idx[0])
    for ii in idx[1:]:
        ii = int(ii)
        if ii == prev + 1:
            prev = ii
        else:
            segments.append((s, prev))
            s = ii
            prev = ii
    segments.append((s, prev))

    seg = None
    for a, b in segments:
        if a <= peak <= b:
            seg = (a, b)
            break
    if seg is None:
        seg = max(segments, key=lambda ab: ab[1] - ab[0])
    start, end = seg
    if (end - start + 1) < min_len:
        extra = min_len - (end - start + 1)
        start = max(0, start - (extra // 2 + extra % 2))
        end = min(len(E) - 1, end + extra // 2)
        while (end - start + 1) < min_len and (start > 0 or end < len(E) - 1):
            if start > 0:
                start -= 1
            if (end - start + 1) >= min_len:
                break
            if end < len(E) - 1:
                end += 1
    return int(start), int(end + 1)  # end exclusive


def resample_sequence_nan_robust(frames: np.ndarray, n_frames: int) -> np.ndarray:
    T = frames.shape[0]
    if T <= 0:
        raise ValueError("Cannot resample empty sequence.")
    if T == 1:
        return np.repeat(frames, n_frames, axis=0)
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, n_frames)
    out = np.full((n_frames, frames.shape[1], 3), np.nan, dtype=float)
    for m in range(frames.shape[1]):
        for c in range(3):
            y = frames[:, m, c]
            ok = np.isfinite(y)
            if ok.sum() >= 2:
                out[:, m, c] = np.interp(t_new, t_old[ok], y[ok])
            elif ok.sum() == 1:
                out[:, m, c] = y[ok][0]
    return out


def max_missing_gap(mask_tn: np.ndarray) -> int:
    """Max consecutive missing frames over markers for mask shape T,N."""
    T, N = mask_tn.shape
    worst = 0
    for j in range(N):
        run = 0
        for t in range(T):
            if not bool(mask_tn[t, j]):
                run += 1
                worst = max(worst, run)
            else:
                run = 0
    return int(worst)


# ----------------------------
# Sample processing
# ----------------------------

def process_sample(row: pd.Series, args: argparse.Namespace, movement: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    csv_path = Path(resolve_path(str(row["complete_filepath"]), args.root_override))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    ref_path = Path(args.ref_root) / movement / "mean_healthy.npy"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference mean: {ref_path}")
    ref = np.load(ref_path)
    if ref.ndim != 3 or ref.shape[1] < args.dental_n + 1:
        raise ValueError(f"Unexpected reference shape: {ref.shape}")
    if ref.shape[1] > args.n_markers:
        ref = ref[:, :args.n_markers, :]
    if ref.shape[0] != args.n_frames:
        ref = resample_sequence_nan_robust(ref, args.n_frames)

    frames_world = load_facemocap_csv(
        csv_path,
        skiprows=args.skiprows,
        drop_first_cols=args.drop_first_cols,
        n_markers_keep=args.n_markers,
    )
    frames_dental, dental_ok = world_to_dental(frames_world)
    if dental_ok.mean() < args.min_dental_ok_frac:
        raise RuntimeError(f"Dental-frame valid fraction too low: {dental_ok.mean():.3f}")

    face_idx = np.arange(args.dental_n, args.n_markers, dtype=int)
    neutral_raw = choose_neutral_idx(frames_dental, face_idx, args.neutral_first_pct)

    Rfix = fixed_rot_xyz(tuple(args.fixed_rot_xyz))
    flat = frames_dental.reshape(-1, 3)
    frames_can = (Rfix @ flat.T).T.reshape(frames_dental.shape)

    Pref = ref[0, face_idx, :]
    Qtar = frames_can[neutral_raw, face_idx, :]
    valid = valid_points(Pref) & valid_points(Qtar)
    if int(valid.sum()) < args.min_points:
        raise RuntimeError(f"Not enough valid facial correspondences: {int(valid.sum())} < {args.min_points}")

    # Estimate transform mapping target neutral -> reference neutral.
    P = Qtar[valid]
    Q = Pref[valid]
    if args.ransac:
        R, t, inl, score = trimmed_ransac_rigid(
            P, Q,
            trials=args.ransac_trials,
            subset=args.ransac_subset,
            trim_frac=args.trim_frac,
            rng=rng,
        )
        n_inliers = int(inl.sum())
    else:
        R, t = kabsch(P, Q)
        res = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
        inl = trimmed_indices(res, args.trim_frac)
        score = rms(res[inl])
        n_inliers = int(inl.sum())

    used_flip = False
    if args.try_yaw_flip:
        R, t, used_flip, score = maybe_yaw_flip(P, Q, R, t, args.yaw_flip_axis)

    flat = frames_can.reshape(-1, 3)
    frames_aligned = apply_rt(flat, R, t).reshape(frames_can.shape)

    E = displacement_energy(frames_aligned, neutral_raw, face_idx)
    start, end = extract_active_window(E, args.energy_thr_percentile, args.min_window_len, args.max_gap)
    frames_win = frames_aligned[start:end]
    frames_res = resample_sequence_nan_robust(frames_win, args.n_frames)
    neutral_res = choose_neutral_idx(frames_res, face_idx, args.neutral_first_pct)

    face = frames_res[:, face_idx, :]
    neutral = face[neutral_res:neutral_res + 1, :, :]
    neutral_face = neutral[0].astype(np.float32)
    X = face - neutral
    M = np.isfinite(X).all(axis=-1)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    M_u8 = M.astype(np.uint8)

    valid_counts = M.sum(axis=1)
    qc = {
        "sample_id": safe_id_from_filepath(str(row["complete_filepath"])),
        "resolved_csv_path": str(csv_path),
        "movement": movement,
        "participant_id": row.get("participant_id", ""),
        "condition": row.get("condition", ""),
        "reference_split": row.get(args.reference_split_col, ""),
        "acquisition_date": row.get("acquisition_date", ""),
        "recording_id": infer_recording_id(row),
        "filename": row.get("filename", csv_path.name),
        "complete_filepath": row.get("complete_filepath", ""),
        "n_frames_raw": int(frames_world.shape[0]),
        "target_window_raw_start": int(start),
        "target_window_raw_end": int(end),
        "target_window_len": int(end - start),
        "neutral_idx_raw": int(neutral_raw),
        "neutral_idx_resampled": int(neutral_res),
        "dental_ok_frac": float(dental_ok.mean()),
        "n_valid_neutral_correspondences": int(valid.sum()),
        "alignment_score": float(score),
        "alignment_used_yaw_flip": bool(used_flip),
        "alignment_n_inliers": int(n_inliers),
        "qc_valid_marker_frame_fraction": float(M.mean()),
        "qc_mean_valid_facial_markers_per_frame": float(valid_counts.mean()),
        "qc_min_valid_facial_markers_per_frame": int(valid_counts.min()) if len(valid_counts) else 0,
        "qc_max_missing_gap_frames": int(max_missing_gap(M)),
    }
    return X, M_u8, neutral_face, qc


# ----------------------------
# Metadata and main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--ref_root", required=True, help="Reference root containing M*/mean_healthy.npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--root_override", default=None)
    ap.add_argument("--movements", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--reference_split_col", default="reference_split")
    ap.add_argument("--reference_label", default="reference")
    ap.add_argument("--evaluation_label", default="evaluation")
    ap.add_argument("--healthy_label", default="healthy")
    ap.add_argument("--pathological_label", default="pathological")

    ap.add_argument("--skiprows", type=int, default=5)
    ap.add_argument("--drop_first_cols", type=int, default=2)
    ap.add_argument("--n_markers", type=int, default=108)
    ap.add_argument("--dental_n", type=int, default=3)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--neutral_first_pct", type=float, default=0.05)
    ap.add_argument("--energy_thr_percentile", type=float, default=5.0)
    ap.add_argument("--min_window_len", type=int, default=70)
    ap.add_argument("--max_gap", type=int, default=5)

    ap.add_argument("--fixed_rot_xyz", nargs=3, type=float, default=[90.0, 90.0, 90.0])
    ap.add_argument("--min_points", type=int, default=60)
    ap.add_argument("--min_dental_ok_frac", type=float, default=0.50)
    ap.add_argument("--ransac", action="store_true", default=True)
    ap.add_argument("--no_ransac", action="store_false", dest="ransac")
    ap.add_argument("--ransac_trials", type=int, default=1200)
    ap.add_argument("--ransac_subset", type=int, default=4)
    ap.add_argument("--trim_frac", type=float, default=0.10)
    ap.add_argument("--try_yaw_flip", action="store_true")
    ap.add_argument("--yaw_flip_axis", choices=["X", "Y", "Z"], default="Z")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit_per_group", type=int, default=0, help="Debug only; 0 means no limit.")
    return ap.parse_args()


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    required = {"complete_filepath", "participant_id", "facial_movement", "condition", "single_movement", "valid_for_processing", args.reference_split_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    out = df.copy()
    out["mov_norm"] = out["facial_movement"].apply(normalize_movement)
    wanted = {m.upper() for m in args.movements}
    out = out[out["mov_norm"].str.upper().isin(wanted)].copy()
    out = out[pd.to_numeric(out["single_movement"], errors="coerce").fillna(0).astype(int) == 1].copy()
    out = out[pd.to_numeric(out["valid_for_processing"], errors="coerce").fillna(0).astype(int) == 1].copy()
    cond = out["condition"].astype(str).str.strip().str.lower()
    split = out[args.reference_split_col].astype(str).str.strip().str.lower()

    train_healthy = out[(cond == args.healthy_label.lower()) & (split == args.reference_label.lower())].copy()
    eval_healthy = out[(cond == args.healthy_label.lower()) & (split == args.evaluation_label.lower())].copy()
    pathological = out[(cond == args.pathological_label.lower())].copy()
    # Prefer evaluation split for pathology when present, but keep all if this would be empty.
    pathological_eval = pathological[split.loc[pathological.index] == args.evaluation_label.lower()].copy()
    if not pathological_eval.empty:
        pathological = pathological_eval

    groups = {
        "train_healthy": train_healthy,
        "eval_healthy": eval_healthy,
        "pathological": pathological,
    }
    if args.limit_per_group and args.limit_per_group > 0:
        groups = {k: v.head(args.limit_per_group).copy() for k, v in groups.items()}
    return groups


def save_group_arrays(mov_dir: Path, group_name: str, Xs: List[np.ndarray], Ms: List[np.ndarray], Ns: List[np.ndarray], rows: List[Dict[str, object]], n_frames: int = 100, n_face: int = 105) -> None:
    if Xs:
        X = np.stack(Xs, axis=0).astype(np.float32)
        M = np.stack(Ms, axis=0).astype(np.uint8)
        N = np.stack(Ns, axis=0).astype(np.float32)
    else:
        X = np.zeros((0, n_frames, n_face, 3), dtype=np.float32)
        M = np.zeros((0, n_frames, n_face), dtype=np.uint8)
        N = np.zeros((0, n_face, 3), dtype=np.float32)
    np.save(mov_dir / f"X_{group_name}.npy", X)
    np.save(mov_dir / f"M_{group_name}.npy", M)
    np.save(mov_dir / f"N_{group_name}.npy", N)
    pd.DataFrame(rows).to_csv(mov_dir / f"metadata_{group_name}.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    write_json(out_dir / "build_config.json", vars(args))

    df = pd.read_csv(args.metadata)
    groups = select_rows(df, args)
    rng = np.random.default_rng(args.seed)
    manifest = {"movements": {}, "groups": list(groups.keys()), "config": vars(args)}

    failures = []
    for movement in [m.upper() for m in args.movements]:
        mov_dir = out_dir / movement
        ensure_dir(mov_dir)
        manifest["movements"][movement] = {}
        print(f"\nMovement {movement}")
        # Store movement-level healthy reference displacement and envelope used by the projector.
        ref_path = Path(args.ref_root) / movement / "mean_healthy.npy"
        if ref_path.exists():
            ref = np.load(ref_path)
            if ref.shape[0] != args.n_frames:
                ref = resample_sequence_nan_robust(ref[:, :args.n_markers, :], args.n_frames)
            face_idx = np.arange(args.dental_n, args.n_markers, dtype=int)
            ref_face = ref[:, face_idx, :]
            ref_disp = (ref_face - ref_face[0:1]).astype(np.float32)
            np.save(mov_dir / "reference_displacement.npy", ref_disp)
        env_path = Path(args.ref_root) / movement / "reference_envelope_per_marker.csv"
        if env_path.exists():
            try:
                pd.read_csv(env_path).to_csv(mov_dir / "reference_envelope_per_marker.csv", index=False)
            except Exception:
                pass
        for group_name, gdf in groups.items():
            gmov = gdf[gdf["mov_norm"].str.upper() == movement].copy()
            print(f"  {group_name}: {len(gmov)} candidate rows")
            Xs: List[np.ndarray] = []
            Ms: List[np.ndarray] = []
            Ns: List[np.ndarray] = []
            meta_rows: List[Dict[str, object]] = []
            for idx, (_, row) in enumerate(gmov.iterrows(), start=1):
                try:
                    X, M, N, meta = process_sample(row, args, movement, rng)
                    meta["ae_group"] = group_name
                    Xs.append(X)
                    Ms.append(M)
                    Ns.append(N)
                    meta_rows.append(meta)
                except Exception as exc:
                    err = {
                        "movement": movement,
                        "group": group_name,
                        "participant_id": row.get("participant_id", ""),
                        "complete_filepath": row.get("complete_filepath", ""),
                        "error": str(exc),
                    }
                    failures.append(err)
                    print(f"    [FAIL] {err['participant_id']} {Path(str(err['complete_filepath'])).name}: {exc}")
            save_group_arrays(mov_dir, group_name, Xs, Ms, Ns, meta_rows, n_frames=args.n_frames, n_face=args.n_markers - args.dental_n)
            manifest["movements"][movement][group_name] = {
                "n_candidates": int(len(gmov)),
                "n_ok": int(len(meta_rows)),
                "X": str(mov_dir / f"X_{group_name}.npy"),
                "M": str(mov_dir / f"M_{group_name}.npy"),
                "N": str(mov_dir / f"N_{group_name}.npy"),
                "metadata": str(mov_dir / f"metadata_{group_name}.csv"),
            }
            print(f"    wrote {len(meta_rows)} samples")

    pd.DataFrame(failures).to_csv(out_dir / "build_failures.csv", index=False)
    manifest["n_failures"] = int(len(failures))
    write_json(out_dir / "dataset_manifest.json", manifest)
    print(f"\n[OK] Wrote AE v2 dataset: {out_dir}")
    print(f"[OK] Failures: {len(failures)}")


if __name__ == "__main__":
    main()
