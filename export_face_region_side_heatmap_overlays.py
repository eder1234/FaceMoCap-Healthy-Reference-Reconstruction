#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_face_region_side_heatmap_overlays.py

Create one schematic face + region-side categorical anomaly overlay PNG per
pathological sample, and a companion target-vs-AE HTML animation next to it.

This version visualizes anomaly classes with respect to the healthy-evaluation
anatomical distributions, using:

  1) raw AE region-side metric matrix for target/pathological samples
     e.g. ae_region_side_metrics_matrix_raw.csv

  2) healthy-evaluation percentile table
     e.g. healthy_eval_region_side_percentiles.csv

For each sample × metric × region-side, the pathological raw value is compared
against the healthy-evaluation percentiles for the same:

    movement × metric × region-side

Anomaly classes:
    NaN = unavailable / unreliable
    0   = within healthy range or low elevation: value <= p75
    1   = mild elevation:                    p75 < value <= p90
    2   = moderate elevation:                p90 < value <= p95
    3   = high elevation:                    p95 < value <= healthy max
    4   = outside healthy-evaluation range:  value > healthy max

Outputs:
  - One PNG per sample:
      *_face_region_side_anomaly_overlay.png

  - One HTML per sample:
      *_target_vs_ae_overlay.html
    containing only observed target and AE healthy reconstruction, with
    controls to show/hide each cloud.

  - face_region_side_anomaly_overlay_index.csv
  - face_region_side_anomaly_overlay_failures.csv
  - ae_region_side_anomaly_class_matrix.csv
  - ae_region_side_anomaly_class_long.csv

Example:
  cd /media/rodriguez/easystore/Data_FaceMoCap

  python export_face_region_side_heatmap_overlays.py \
    --matrix_csv align_mean_movement_refactor/ae_region_side_clinical_csv/ae_region_side_metrics_matrix_raw.csv \
    --healthy_percentiles align_mean_movement_refactor/ae_region_side_clinical_csv/healthy_eval_region_side_percentiles.csv \
    --template_image neutral_face.png \
    --out_dir align_mean_movement_refactor/ae_region_side_clinical_face_overlays_percentile \
    --project_root . \
    --patients VJ OC \
    --movements M1 M2 M3 M4 M5

Notes:
  - Anatomical side convention follows frontal-face visualization:
      subject RIGHT is shown on image-left;
      subject LEFT is shown on image-right.
  - NaN/unreliable cells are shown as gray.
  - The HTML generation requires plotly. If plotly is unavailable, PNGs still
    export and the failure CSV reports the HTML issue.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# -------------------------------------------------------------------------
# Metric configuration
# -------------------------------------------------------------------------

METRIC_ORDER = [
    "trajectory_abnormality",
    "amplitude_abnormality",
    "hypokinesia",
    "hyperkinesia",
    "counter_direction_ratio",
]

METRIC_LABELS_SHORT = {
    "trajectory_abnormality": "Trj",
    "amplitude_abnormality": "Amp",
    "hypokinesia": "Hypo",
    "hyperkinesia": "Hyper",
    "counter_direction_ratio": "Ctr",
}

METRIC_ALIASES = {
    "trajectory": "trajectory_abnormality",
    "trajectory_mae": "trajectory_abnormality",
    "trajectory_abnormality": "trajectory_abnormality",
    "amplitude": "amplitude_abnormality",
    "absolute_amplitude_abnormality": "amplitude_abnormality",
    "amplitude_abnormality": "amplitude_abnormality",
    "hypokinesia": "hypokinesia",
    "hypokinesia_proxy": "hypokinesia",
    "hyperkinesia": "hyperkinesia",
    "hyperkinesia_like": "hyperkinesia",
    "hyperkinesia_like_proxy": "hyperkinesia",
    "counter_direction": "counter_direction_ratio",
    "counter_direction_ratio": "counter_direction_ratio",
}

META_COLUMNS = {
    "participant_id",
    "acquisition_date",
    "movement",
    "filename",
    "sample_id",
    "condition",
    "group",
    "metric",
    "metric_label",
    "metric_norm",
    "html_path",
    "html_relpath",
    "npz_path",
    "npz_relpath",
    "npz_path_resolved",
    "alignment_score",
    "qc_valid_marker_frame_fraction",
}


# -------------------------------------------------------------------------
# Region anchors.
#
# Coordinates are in pixels for the user-provided neutral_face.png template
# with approximate size 532×655. They are automatically scaled if another
# image size is provided.
#
# IMPORTANT SIDE CONVENTION:
#   anatomical R -> image-left
#   anatomical L -> image-right
# -------------------------------------------------------------------------

REFERENCE_W = 532.0
REFERENCE_H = 655.0

# Values: region-side column, display label, x, y, label_position
ANCHORS_REF = [
    # Forehead / brow / eyelids
    ("R__front", "R front", 155, 205, "below"),
    ("L__front", "L front", 380, 205, "below"),
    ("C__front", "C front", 266, 170, "below"),

    ("R__eyebrow", "R brow", 150, 288, "above"),
    ("L__eyebrow", "L brow", 380, 288, "above"),

    ("R__upper_eyelid", "R upper eyelid", 165, 320, "above"),
    ("L__upper_eyelid", "L upper eyelid", 370, 320, "above"),
    ("R__lower_eyelid", "R lower eyelid", 165, 371, "below"),
    ("L__lower_eyelid", "L lower eyelid", 370, 371, "below"),

    ("R__lateral_canthus", "R lateral canthus", 103, 348, "below"),
    ("L__lateral_canthus", "L lateral canthus", 427, 348, "below"),
    ("R__medial_canthus", "R medial canthus", 211, 333, "below"),
    ("L__medial_canthus", "L medial canthus", 318, 333, "below"),

    # Midface / nose
    ("C__nasion", "Nasion", 266, 398, "below"),
    ("C__nasal_bone", "Nasal bone", 266, 430, "below"),
    ("R__ala_of_nose", "R ala nose", 238, 438, "below"),
    ("L__ala_of_nose", "L ala nose", 295, 438, "below"),

    ("R__levator_superioris_nasi", "R lev sup nasi", 204, 430, "below"),
    ("L__levator_superioris_nasi", "L lev sup nasi", 330, 430, "below"),
    ("R__levator_superioris", "R lev sup", 192, 470, "below"),
    ("L__levator_superioris", "L lev sup", 342, 470, "below"),

    ("R__palpebra_malar_groove", "R palp-malar", 165, 405, "below"),
    ("L__palpebra_malar_groove", "L palp-malar", 368, 405, "below"),

    # Cheek / zygomatic
    ("R__zygomaticus_minor", "R zyg minor", 142, 455, "below"),
    ("L__zygomaticus_minor", "L zyg minor", 392, 455, "below"),
    ("R__zygomaticus_major", "R zyg major", 137, 485, "below"),
    ("L__zygomaticus_major", "L zyg major", 397, 485, "below"),

    # Mouth / lower face
    ("C__philtrum", "Philtrum", 266, 482, "above"),
    ("C__lower_lip", "C lower lip", 266, 552, "below"),

    ("R__labial_commissure", "R labial comm.", 222, 523, "below"),
    ("L__labial_commissure", "L labial comm.", 310, 523, "below"),

    ("R__upper_lip", "R upper lip", 235, 512, "above"),
    ("L__upper_lip", "L upper lip", 298, 512, "above"),
    ("R__lower_lip", "R lower lip", 235, 552, "below"),
    ("L__lower_lip", "L lower lip", 298, 552, "below"),

    ("R__depressor_anguli_oris", "R DAO", 218, 570, "below"),
    ("L__depressor_anguli_oris", "L DAO", 315, 570, "below"),

    ("R__mentalis", "R mentalis", 230, 610, "below"),
    ("L__mentalis", "L mentalis", 303, 610, "below"),
    ("C__mentalis", "C mentalis", 266, 628, "below"),
]

# Extra offsets to reduce overlap. Values are in reference-image pixels and
# are scaled with the template image.
OFFSETS_REF = {
    "R__upper_eyelid": (0, -7),
    "L__upper_eyelid": (0, -7),
    "R__lower_eyelid": (0, 16),
    "L__lower_eyelid": (0, 16),

    "R__medial_canthus": (0, 28),
    "L__medial_canthus": (0, 28),
    "R__lateral_canthus": (-8, 0),
    "L__lateral_canthus": (8, 0),

    "C__nasion": (0, 6),
    "C__nasal_bone": (0, 8),

    "R__ala_of_nose": (-18, 0),
    "L__ala_of_nose": (18, 0),
    "R__levator_superioris_nasi": (-8, -4),
    "L__levator_superioris_nasi": (8, -4),
    "R__levator_superioris": (-12, 2),
    "L__levator_superioris": (12, 2),
    "R__palpebra_malar_groove": (-15, -4),
    "L__palpebra_malar_groove": (15, -4),

    "C__philtrum": (0, -18),
    "R__upper_lip": (-10, -8),
    "L__upper_lip": (10, -8),
    "R__lower_lip": (-10, 16),
    "L__lower_lip": (10, 16),
    "R__depressor_anguli_oris": (-10, 18),
    "L__depressor_anguli_oris": (10, 18),
    "R__mentalis": (-14, 18),
    "L__mentalis": (14, 18),
    "C__mentalis": (0, 16),

    "R__zygomaticus_minor": (-10, -8),
    "L__zygomaticus_minor": (10, -8),
    "R__zygomaticus_major": (-12, 12),
    "L__zygomaticus_major": (12, 12),
}


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object, max_len: int = 160) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:max_len].strip("_") or "sample"


def normalize_movement(value: object) -> str:
    s = str(value).strip().upper()
    m = re.search(r"M0*([1-9]\d*)", s)
    if m:
        return f"M{int(m.group(1))}"
    return s


def normalize_metric_name(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return METRIC_ALIASES.get(s, s)


def pick_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    candidates.extend([
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def scale_anchor(x: float, y: float, sx: float, sy: float) -> Tuple[int, int]:
    return int(round(x * sx)), int(round(y * sy))


def scaled_offset(col: str, sx: float, sy: float) -> Tuple[int, int]:
    dx, dy = OFFSETS_REF.get(col, (0, 0))
    return int(round(dx * sx)), int(round(dy * sy))


def infer_region_columns(df: pd.DataFrame) -> List[str]:
    region_cols = []
    for c in df.columns:
        if c in META_COLUMNS:
            continue
        if re.match(r"^[LRC]__", str(c)):
            region_cols.append(c)
    return region_cols


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        raise ValueError("Input matrix CSV must contain a 'metric' column.")
    if "sample_id" not in df.columns:
        raise ValueError("Input matrix CSV must contain a 'sample_id' column.")
    df = df.copy()
    df["metric_norm"] = df["metric"].map(normalize_metric_name)
    if "movement" in df.columns:
        df["movement_norm"] = df["movement"].map(normalize_movement)
    else:
        df["movement_norm"] = ""
    return df


def load_percentiles(path: Path) -> pd.DataFrame:
    pct = pd.read_csv(path)
    required = ["movement", "metric", "region_side", "healthy_n", "reliable", "p75", "p90", "p95", "max"]
    missing = [c for c in required if c not in pct.columns]
    if missing:
        raise ValueError(f"Healthy percentile CSV is missing required columns: {missing}")

    pct = pct.copy()
    pct["metric_norm"] = pct["metric"].map(normalize_metric_name)
    pct["movement_norm"] = pct["movement"].map(normalize_movement)
    pct["region_side"] = pct["region_side"].astype(str)
    return pct


def build_percentile_lookup(pct: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    lookup = {}
    for _, row in pct.iterrows():
        key = (str(row["movement_norm"]), str(row["metric_norm"]), str(row["region_side"]))
        lookup[key] = {
            "healthy_n": int(row["healthy_n"]) if pd.notna(row["healthy_n"]) else 0,
            "reliable": bool(int(row["reliable"])) if pd.notna(row["reliable"]) else False,
            "p75": float(row["p75"]) if pd.notna(row["p75"]) else np.nan,
            "p90": float(row["p90"]) if pd.notna(row["p90"]) else np.nan,
            "p95": float(row["p95"]) if pd.notna(row["p95"]) else np.nan,
            "max": float(row["max"]) if pd.notna(row["max"]) else np.nan,
        }
    return lookup


def filter_rows(
    df: pd.DataFrame,
    patients: Optional[List[str]],
    movements: Optional[List[str]],
    conditions: Optional[List[str]],
) -> pd.DataFrame:
    out = df.copy()

    if patients:
        wanted = {str(p) for p in patients}
        if "participant_id" in out.columns:
            out = out[out["participant_id"].astype(str).isin(wanted)].copy()

    if movements:
        wanted = {normalize_movement(m) for m in movements}
        if "movement_norm" in out.columns:
            out = out[out["movement_norm"].isin(wanted)].copy()

    if conditions:
        wanted = {str(c).strip().lower() for c in conditions}
        if "condition" in out.columns:
            out = out[out["condition"].astype(str).str.strip().str.lower().isin(wanted)].copy()

    return out


def sample_group_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "participant_id",
        "acquisition_date",
        "movement",
        "movement_norm",
        "filename",
        "sample_id",
        "condition",
        "group",
        "html_path",
        "html_relpath",
        "npz_path",
        "npz_relpath",
        "npz_path_resolved",
        "alignment_score",
        "qc_valid_marker_frame_fraction",
    ]
    return [c for c in candidates if c in df.columns]


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

    seen = set()
    uniq = []
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


# -------------------------------------------------------------------------
# Anomaly classification
# -------------------------------------------------------------------------

def classify_value(value: object, stats: Optional[Dict[str, float]]) -> float:
    """
    Return anomaly class:
        NaN unavailable/unreliable
        0   value <= p75
        1   p75 < value <= p90
        2   p90 < value <= p95
        3   p95 < value <= max
        4   value > max
    """
    if value is None or pd.isna(value):
        return np.nan
    if stats is None:
        return np.nan
    if not stats.get("reliable", False):
        return np.nan

    v = float(value)
    p75 = stats.get("p75", np.nan)
    p90 = stats.get("p90", np.nan)
    p95 = stats.get("p95", np.nan)
    vmax = stats.get("max", np.nan)

    if not np.isfinite(v) or not np.isfinite(p75) or not np.isfinite(p90) or not np.isfinite(p95) or not np.isfinite(vmax):
        return np.nan

    # Enforce monotonic thresholds defensively.
    t75 = p75
    t90 = max(p90, t75)
    t95 = max(p95, t90)
    tmax = max(vmax, t95)

    if v <= t75:
        return 0.0
    if v <= t90:
        return 1.0
    if v <= t95:
        return 2.0
    if v <= tmax:
        return 3.0
    return 4.0


def make_anomaly_long_and_matrix(
    df: pd.DataFrame,
    percentile_lookup: Dict[Tuple[str, str, str], Dict[str, float]],
    region_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []

    meta_cols = sample_group_columns(df)
    for _, row in df.iterrows():
        movement = str(row.get("movement_norm", normalize_movement(row.get("movement", ""))))
        metric = str(row.get("metric_norm", normalize_metric_name(row.get("metric", ""))))

        base = {c: row.get(c, "") for c in meta_cols}
        base.update({
            "movement_norm": movement,
            "metric": metric,
            "metric_label": METRIC_LABELS_SHORT.get(metric, metric),
        })

        for col in region_cols:
            stats = percentile_lookup.get((movement, metric, col))
            raw_value = row.get(col, np.nan)
            anomaly_class = classify_value(raw_value, stats)
            long_rows.append({
                **base,
                "region_side": col,
                "raw_value": raw_value,
                "anomaly_class": anomaly_class,
                "healthy_n": stats.get("healthy_n", np.nan) if stats else np.nan,
                "healthy_reliable": int(stats.get("reliable", False)) if stats else 0,
                "healthy_p75": stats.get("p75", np.nan) if stats else np.nan,
                "healthy_p90": stats.get("p90", np.nan) if stats else np.nan,
                "healthy_p95": stats.get("p95", np.nan) if stats else np.nan,
                "healthy_max": stats.get("max", np.nan) if stats else np.nan,
            })

    long_df = pd.DataFrame(long_rows)

    index_cols = [
        c for c in [
            "participant_id", "acquisition_date", "movement", "movement_norm",
            "filename", "sample_id", "condition", "group",
            "html_path", "html_relpath", "npz_path", "npz_relpath", "npz_path_resolved",
            "metric", "metric_label"
        ]
        if c in long_df.columns
    ]

    # Safe wide-matrix export.
    #
    # IMPORTANT:
    # Do not use pivot_table(..., dropna=False) with many metadata columns.
    # In recent pandas versions, that can create a huge Cartesian product of
    # index levels and allocate impossible amounts of memory.
    #
    # Instead, assign one compact row id to each real sample × metric row,
    # pivot only on that row id, then merge the metadata back.
    if long_df.empty:
        return long_df, pd.DataFrame()

    tmp = long_df.copy()
    tmp["_row_id"] = tmp.groupby(index_cols, dropna=False, sort=False).ngroup()

    meta = tmp[["_row_id"] + index_cols].drop_duplicates("_row_id")
    matrix = tmp.pivot(index="_row_id", columns="region_side", values="anomaly_class").reset_index()
    matrix.columns.name = None

    matrix = meta.merge(matrix, on="_row_id", how="left").drop(columns=["_row_id"])

    return long_df, matrix


# -------------------------------------------------------------------------
# Drawing functions for PNG
# -------------------------------------------------------------------------

def anomaly_color(cls: float, cmap) -> Tuple[int, int, int, int]:
    """
    Class to color. NaN = gray. Classes 0..4 use discrete points in inferno.
    """
    if pd.isna(cls):
        return (220, 220, 220, 230)

    c = int(round(float(cls)))
    c = max(0, min(4, c))
    # Avoid too-black class 0 but keep it subdued.
    positions = {
        0: 0.08,
        1: 0.32,
        2: 0.55,
        3: 0.78,
        4: 0.98,
    }
    rgba = cmap(positions[c])
    return tuple(int(255 * x) for x in rgba[:3]) + (235,)


def draw_mini_heatmap(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    vals: Iterable[float],
    label: str,
    cmap,
    font_tiny,
    cell_w: int,
    cell_h: int,
    gap: int,
    label_pos: str = "below",
    draw_labels: bool = True,
    cell_outline=(20, 20, 20, 170),
) -> None:
    vals = list(vals)
    n = len(vals)
    total_w = n * cell_w + (n - 1) * gap
    total_h = cell_h
    x0 = int(cx - total_w / 2)
    y0 = int(cy - total_h / 2)

    pad = 2
    draw.rounded_rectangle(
        [x0 - pad, y0 - pad, x0 + total_w + pad, y0 + total_h + pad],
        radius=3,
        fill=(255, 255, 255, 175),
        outline=(35, 35, 35, 145),
        width=1,
    )

    for j, v in enumerate(vals):
        x = x0 + j * (cell_w + gap)
        color = anomaly_color(v, cmap)
        draw.rectangle(
            [x, y0, x + cell_w, y0 + cell_h],
            fill=color,
            outline=cell_outline,
        )

    if draw_labels:
        bbox = draw.textbbox((0, 0), label, font=font_tiny)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if label_pos == "above":
            ty = y0 - th - 3
        else:
            ty = y0 + cell_h + 3
        draw.text((int(cx - tw / 2), ty), label, fill=(0, 0, 0, 255), font=font_tiny)


def draw_class_legend(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    cmap,
    font_small,
) -> None:
    labels = [
        "0 <=p75",
        "1 p75-p90",
        "2 p90-p95",
        "3 >p95",
        "4 >max",
        "NA",
    ]
    colors = [anomaly_color(i, cmap) for i in range(5)] + [(220, 220, 220, 230)]
    x = x0
    for lab, col in zip(labels, colors):
        draw.rectangle([x, y0, x + 13, y0 + 13], fill=col, outline=(0, 0, 0, 255))
        draw.text((x + 17, y0 - 1), lab, fill=(0, 0, 0, 255), font=font_small)
        x += 78 if lab != "NA" else 45


def extract_sample_classes(sample_df: pd.DataFrame, region_cols: List[str]) -> Dict[str, List[float]]:
    values = {c: [np.nan] * len(METRIC_ORDER) for c in region_cols}
    metric_to_row = {}
    for _, row in sample_df.iterrows():
        m = normalize_metric_name(row.get("metric", ""))
        metric_to_row[m] = row

    for k, metric in enumerate(METRIC_ORDER):
        row = metric_to_row.get(metric)
        if row is None:
            continue
        for c in region_cols:
            try:
                values[c][k] = float(row[c]) if pd.notna(row[c]) else np.nan
            except Exception:
                values[c][k] = np.nan
    return values


def create_anomaly_png_for_sample(
    template_image: Path,
    sample_df: pd.DataFrame,
    region_cols: List[str],
    out_png: Path,
    cmap_name: str = "inferno",
    draw_labels: bool = True,
    cell_w: int = 10,
    cell_h: int = 10,
    gap: int = 1,
) -> None:
    img = Image.open(template_image).convert("RGBA")
    W, H = img.size

    canvas_pad_bottom = 75
    canvas = Image.new("RGBA", (W, H + canvas_pad_bottom), (255, 255, 255, 255))
    canvas.alpha_composite(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    font_title = pick_font(13, bold=True)
    font_small = pick_font(10, bold=False)
    font_tiny = pick_font(8, bold=False)

    cmap = mpl.colormaps[cmap_name]
    sx, sy = W / REFERENCE_W, H / REFERENCE_H

    first = sample_df.iloc[0]
    patient = str(first.get("participant_id", ""))
    date = str(first.get("acquisition_date", ""))
    movement = str(first.get("movement", ""))
    filename = str(first.get("filename", ""))

    title = f"{patient} | {date} | {movement} | {filename}"
    subtitle = "Anomaly class vs healthy-evaluation percentiles; subject right appears on image-left"

    draw.text((18, 14), title, fill=(0, 0, 0, 255), font=font_title)
    draw.text((18, 33), subtitle, fill=(0, 0, 0, 255), font=font_small)
    draw.text((18, 50), "Each rectangle: Trj | Amp | Hypo | Hyper | Ctr", fill=(0, 0, 0, 255), font=font_small)

    values = extract_sample_classes(sample_df, region_cols)

    available_cols = set(region_cols)
    for col, label, x_ref, y_ref, label_pos in ANCHORS_REF:
        if col not in available_cols:
            continue
        x, y = scale_anchor(x_ref, y_ref, sx, sy)
        dx, dy = scaled_offset(col, sx, sy)
        draw_mini_heatmap(
            draw=draw,
            cx=x + dx,
            cy=y + dy,
            vals=values.get(col, [np.nan] * len(METRIC_ORDER)),
            label=label,
            cmap=cmap,
            font_tiny=font_tiny,
            cell_w=cell_w,
            cell_h=cell_h,
            gap=gap,
            label_pos=label_pos,
            draw_labels=draw_labels,
        )

    # Categorical legend
    legend_x = int(round(18 * sx))
    legend_y = H + 28
    draw_class_legend(draw, legend_x, legend_y, cmap, font_small)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(220, 220))


# -------------------------------------------------------------------------
# Target-vs-AE HTML generation
# -------------------------------------------------------------------------

def finite_axis_ranges(seqs: Iterable[np.ndarray], masks: Optional[Iterable[Optional[np.ndarray]]] = None, pad_frac: float = 0.08):
    seqs = list(seqs)
    if masks is None:
        masks = [None] * len(seqs)
    else:
        masks = list(masks)

    pts = []
    for seq, mask in zip(seqs, masks):
        ok = np.isfinite(seq).all(axis=-1)
        if mask is not None:
            ok = ok & mask.astype(bool)
        if ok.any():
            pts.append(seq[ok])
    if not pts:
        return [-1, 1], [-1, 1], [-1, 1]

    P = np.concatenate(pts, axis=0)
    mn = np.nanmin(P, axis=0)
    mx = np.nanmax(P, axis=0)
    span = float(np.nanmax(mx - mn))
    pad = pad_frac * span if np.isfinite(span) and span > 0 else 1.0
    return (
        [float(mn[0] - pad), float(mx[0] + pad)],
        [float(mn[1] - pad), float(mx[1] + pad)],
        [float(mn[2] - pad), float(mx[2] + pad)],
    )


def visible_points(seq: np.ndarray, frame_idx: int, valid_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = seq[frame_idx]
    ok = np.isfinite(pts).all(axis=1)
    if valid_mask is not None:
        ok = ok & valid_mask[frame_idx].astype(bool)
    idx = np.where(ok)[0]
    return pts[ok], idx


def trace_for(seq: np.ndarray, frame_idx: int, valid_mask: Optional[np.ndarray], name: str, color: str, size: int, opacity: float):
    pts, idx = visible_points(seq, frame_idx, valid_mask)
    return go.Scatter3d(
        x=pts[:, 0] if pts.size else [],
        y=pts[:, 1] if pts.size else [],
        z=pts[:, 2] if pts.size else [],
        mode="markers",
        name=name,
        marker=dict(size=size, color=color, opacity=opacity),
        text=[f"marker {int(i)}" for i in idx],
        hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    )


def load_npz_arrays(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    z = np.load(npz_path)
    required = ["observed_face", "aev3_healthy_reconstruction"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"Missing required arrays in {npz_path}: {missing}")

    observed = z["observed_face"]
    ae = z["aev3_healthy_reconstruction"]
    mask = z["target_mask"] if "target_mask" in z.files else None

    if observed.ndim != 3 or ae.ndim != 3:
        raise ValueError(f"Expected observed and AE arrays with shape T×N×3 in {npz_path}")
    if observed.shape != ae.shape:
        raise ValueError(f"Observed and AE shape mismatch in {npz_path}: {observed.shape} vs {ae.shape}")
    if mask is not None and mask.shape[:2] != observed.shape[:2]:
        raise ValueError(f"Mask shape mismatch in {npz_path}: {mask.shape} vs {observed.shape[:2]}")

    return observed, ae, mask


def lowpass_trajectories_for_html(x: np.ndarray, fs: float = 100.0, cutoff: float = 8.0) -> np.ndarray:
    """
    Apply the same 8 Hz low-pass filtering used for anatomical metric computation.

    NaNs are linearly interpolated before filtering. Invalid marker-frame pairs
    are still hidden later through the target mask, so interpolation is only used
    to make filtering numerically stable.
    """
    if cutoff <= 0 or fs <= 0:
        return x

    try:
        from scipy.signal import butter, filtfilt
    except Exception as exc:
        raise RuntimeError(
            "scipy is required to generate filtered HTML trajectories. "
            "Install scipy or set lowpass_hz <= 0."
        ) from exc

    T = x.shape[0]
    if T < 8:
        return x

    nyq = 0.5 * fs
    wn = float(cutoff) / nyq
    if not (0 < wn < 1):
        return x

    b, a = butter(4, wn, btype="low")

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


def make_target_ae_html_from_npz(
    npz_path: Path,
    out_html: Path,
    title: str,
    hide_invalid_predictions: bool = True,
    frame_duration: int = 60,
    fs: float = 100.0,
    lowpass_hz: float = 8.0,
) -> None:
    """
    Create target-vs-AE HTML animation.

    Important:
    The displayed trajectories are low-pass filtered at 8 Hz by default, matching
    the anatomical metrics used in the face-overlay PNG.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is not installed. Install plotly or use --copy_existing_html.")

    observed, ae, mask = load_npz_arrays(npz_path)

    # Apply the same temporal filtering used before anatomical metric computation.
    if lowpass_hz is not None and lowpass_hz > 0:
        observed_display = lowpass_trajectories_for_html(observed, fs=fs, cutoff=lowpass_hz)
        ae_display = lowpass_trajectories_for_html(ae, fs=fs, cutoff=lowpass_hz)
        filter_note = f"Displayed trajectories are low-pass filtered at {lowpass_hz:g} Hz."
    else:
        observed_display = observed
        ae_display = ae
        filter_note = "Displayed trajectories are unfiltered."

    T = int(observed_display.shape[0])
    seqs = [observed_display, ae_display]
    masks = [mask, mask if hide_invalid_predictions else None]
    xr, yr, zr = finite_axis_ranges(seqs, masks)

    def traces_at(frame_idx: int):
        return [
            trace_for(
                observed_display,
                frame_idx,
                mask,
                "Observed target",
                "#d62728",
                4,
                0.86,
            ),
            trace_for(
                ae_display,
                frame_idx,
                mask if hide_invalid_predictions else None,
                "AE healthy reconstruction",
                "#1f77b4",
                4,
                0.76,
            ),
        ]

    frames = [
        go.Frame(data=traces_at(t), name=str(t), traces=[0, 1])
        for t in range(T)
    ]

    steps = [
        dict(
            method="animate",
            args=[
                [str(t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(t),
        )
        for t in range(T)
    ]

    fig = go.Figure(data=traces_at(0), frames=frames)

    mask_note = (
        "AE points are hidden where the target marker/frame was invalid."
        if hide_invalid_predictions
        else "All finite AE points are shown."
    )

    note = f"{filter_note} {mask_note}"

    fig.update_layout(
        title=f"{title}<br><sup>{note}</sup>",
        scene=dict(
            xaxis=dict(range=xr, title="x"),
            yaxis=dict(range=yr, title="y"),
            zaxis=dict(range=zr, title="z"),
            aspectmode="data",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=-0.03,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Frame: "),
                pad=dict(t=45),
                steps=steps,
            )
        ],
        margin=dict(l=0, r=0, t=95, b=0),
        height=760,
    )

    plot_div = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="target_ae_plot",
    )

    controls = "\n".join([
        "<label><input type='checkbox' class='cloud-toggle' data-trace='0' checked> "
        "<span style='color:#d62728'>Observed target</span></label>",
        "<label><input type='checkbox' class='cloud-toggle' data-trace='1' checked> "
        "<span style='color:#1f77b4'>AE healthy reconstruction</span></label>",
    ])

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 10px; }}
    .controls label {{ cursor: pointer; white-space: nowrap; }}
    .controls button {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div class="controls">
    {controls}
    <button type="button" id="show_all">Show all</button>
    <button type="button" id="target_only">Target only</button>
    <button type="button" id="ae_only">AE only</button>
  </div>
  {plot_div}
  <script>
    const plotId = "target_ae_plot";

    function setTrace(traceIndex, isVisible) {{
      Plotly.restyle(plotId, {{visible: isVisible ? true : "legendonly"}}, [traceIndex]);
    }}

    function setBox(idx, state) {{
      const box = document.querySelector(".cloud-toggle[data-trace='" + idx + "']");
      if (box) box.checked = state;
      setTrace(idx, state);
    }}

    document.querySelectorAll(".cloud-toggle").forEach(function(box) {{
      box.addEventListener("change", function() {{
        setTrace(Number(box.dataset.trace), box.checked);
      }});
    }});

    document.getElementById("show_all").addEventListener("click", function() {{
      setBox(0, true);
      setBox(1, true);
    }});

    document.getElementById("target_only").addEventListener("click", function() {{
      setBox(0, true);
      setBox(1, false);
    }});

    document.getElementById("ae_only").addEventListener("click", function() {{
      setBox(0, false);
      setBox(1, true);
    }});
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")

def copy_existing_html(row: pd.Series, project_root: Optional[Path], base_dir: Path, out_html: Path) -> Optional[Path]:
    for col in ["html_path", "html_relpath"]:
        if col not in row.index:
            continue
        for src in candidate_paths(row.get(col), project_root, base_dir):
            if src.exists():
                out_html.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_html)
                return out_html
    return None


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_csv", required=True, help="Raw matrix CSV, preferably ae_region_side_metrics_matrix_raw.csv")
    ap.add_argument("--healthy_percentiles", required=True, help="healthy_eval_region_side_percentiles.csv")
    ap.add_argument("--template_image", required=True, help="Neutral schematic face image, e.g. neutral_face.png")
    ap.add_argument("--out_dir", required=True, help="Output directory for PNG and HTML overlays")
    ap.add_argument("--project_root", default=None, help="Project root used to resolve relative NPZ/HTML paths")
    ap.add_argument("--patients", nargs="*", default=["VJ", "OC"], help="Patient IDs to export")
    ap.add_argument("--movements", nargs="*", default=["M1", "M2", "M3", "M4", "M5"], help="Movements to export")
    ap.add_argument("--conditions", nargs="*", default=["pathological"], help="Conditions to export")
    ap.add_argument("--cmap", default="inferno", help="Matplotlib colormap name")
    ap.add_argument("--no_labels", action="store_true", help="Do not draw region labels")
    ap.add_argument("--cell_w", type=int, default=10, help="Mini-heatmap cell width")
    ap.add_argument("--cell_h", type=int, default=10, help="Mini-heatmap cell height")
    ap.add_argument("--gap", type=int, default=1, help="Gap between mini-heatmap cells")
    ap.add_argument("--no_html", action="store_true", help="Only export PNG overlays; skip companion HTML")
    ap.add_argument("--copy_existing_html", action="store_true", help="Copy existing four-cloud HTML instead of creating target-vs-AE HTML")
    ap.add_argument("--show_invalid_predictions", action="store_true", help="Show finite AE points even where target mask is invalid")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    matrix_csv = Path(args.matrix_csv)
    healthy_percentiles = Path(args.healthy_percentiles)
    template_image = Path(args.template_image)
    out_dir = Path(args.out_dir)
    project_root = Path(args.project_root).resolve() if args.project_root else None
    base_dir = matrix_csv.parent.resolve()

    ensure_dir(out_dir)

    if not matrix_csv.exists():
        raise FileNotFoundError(f"Matrix CSV not found: {matrix_csv}")
    if not healthy_percentiles.exists():
        raise FileNotFoundError(f"Healthy percentile CSV not found: {healthy_percentiles}")
    if not template_image.exists():
        raise FileNotFoundError(f"Template image not found: {template_image}")

    raw_df = load_matrix(matrix_csv)
    raw_df = filter_rows(
        raw_df,
        patients=args.patients,
        movements=args.movements,
        conditions=args.conditions,
    )

    region_cols = infer_region_columns(raw_df)
    if not region_cols:
        raise ValueError("No region-side columns found. Expected columns like L__front, R__eyebrow, C__nasion.")

    pct_df = load_percentiles(healthy_percentiles)
    percentile_lookup = build_percentile_lookup(pct_df)

    # Convert raw values into anomaly classes using healthy percentiles.
    anomaly_long, anomaly_matrix = make_anomaly_long_and_matrix(
        raw_df,
        percentile_lookup=percentile_lookup,
        region_cols=region_cols,
    )

    class_long_path = out_dir / "ae_region_side_anomaly_class_long.csv"
    class_matrix_path = out_dir / "ae_region_side_anomaly_class_matrix.csv"
    anomaly_long.to_csv(class_long_path, index=False)
    anomaly_matrix.to_csv(class_matrix_path, index=False)

    anchor_cols = {a[0] for a in ANCHORS_REF}
    missing_anchors = sorted(set(region_cols) - anchor_cols)
    unused_anchors = sorted(anchor_cols - set(region_cols))

    print(f"[INFO] Raw rows after filtering: {len(raw_df)}")
    print(f"[INFO] Anomaly matrix rows: {len(anomaly_matrix)}")
    print(f"[INFO] Region-side columns: {len(region_cols)}")
    print(f"[INFO] Wrote class long CSV: {class_long_path}")
    print(f"[INFO] Wrote class matrix CSV: {class_matrix_path}")

    if missing_anchors:
        print("[WARN] Region-side columns with no visual anchor and therefore not drawn:")
        for c in missing_anchors:
            print(f"  - {c}")
    if unused_anchors:
        print("[INFO] Anchors not present in matrix CSV:")
        for c in unused_anchors:
            print(f"  - {c}")

    group_cols = sample_group_columns(anomaly_matrix)
    key_cols = [c for c in ["participant_id", "acquisition_date", "movement", "filename", "sample_id"] if c in anomaly_matrix.columns]
    if not key_cols:
        key_cols = ["sample_id"]

    index_rows = []
    failures = []

    grouped = anomaly_matrix.groupby(key_cols, dropna=False, sort=True)
    print(f"[INFO] Samples to export: {len(grouped)}")

    for sample_key, sample_df in grouped:
        png_ok = False
        html_ok = False
        png_error = ""
        html_error = ""
        out_png = None
        out_html = None
        npz_path = None

        try:
            sample_df = sample_df.copy()
            first = sample_df.iloc[0]

            patient = safe_name(first.get("participant_id", "unknown"))
            date = safe_name(first.get("acquisition_date", "unknown_date"))
            movement = safe_name(first.get("movement", "unknown_movement"))
            sample_id = safe_name(first.get("sample_id", "unknown_sample"))

            sample_out_dir = out_dir / patient / date / movement
            out_png = sample_out_dir / f"{sample_id}_face_region_side_anomaly_overlay.png"
            out_html = sample_out_dir / f"{sample_id}_target_vs_ae_overlay.html"

            # PNG
            try:
                create_anomaly_png_for_sample(
                    template_image=template_image,
                    sample_df=sample_df,
                    region_cols=region_cols,
                    out_png=out_png,
                    cmap_name=args.cmap,
                    draw_labels=not args.no_labels,
                    cell_w=args.cell_w,
                    cell_h=args.cell_h,
                    gap=args.gap,
                )
                png_ok = True
                print(f"[OK PNG] {out_png}")
            except Exception as exc:
                png_error = str(exc)
                print(f"[FAIL PNG] {sample_key}: {exc}")

            # HTML
            if not args.no_html:
                try:
                    if args.copy_existing_html:
                        copied = copy_existing_html(first, project_root, base_dir, out_html)
                        if copied is None:
                            raise FileNotFoundError("Could not resolve existing html_path/html_relpath")
                        html_ok = True
                    else:
                        npz_path = resolve_npz_path(first, project_root, base_dir)
                        if npz_path is None:
                            raise FileNotFoundError("Could not resolve NPZ path from npz_path_resolved/npz_path/npz_relpath/html_path")
                        title = f"{first.get('participant_id', '')} | {first.get('acquisition_date', '')} | {first.get('movement', '')} | {first.get('filename', '')}"
                        make_target_ae_html_from_npz(
                            npz_path=npz_path,
                            out_html=out_html,
                            title=title,
                            hide_invalid_predictions=not args.show_invalid_predictions,
                        )
                        html_ok = True
                    print(f"[OK HTML] {out_html}")
                except Exception as exc:
                    html_error = str(exc)
                    print(f"[FAIL HTML] {sample_key}: {exc}")

            row = {c: first.get(c, "") for c in group_cols}
            row.update({
                "out_png": str(out_png) if out_png is not None else "",
                "out_png_relpath": str(out_png.relative_to(out_dir)) if out_png is not None and out_png.exists() else "",
                "out_html": str(out_html) if out_html is not None else "",
                "out_html_relpath": str(out_html.relative_to(out_dir)) if out_html is not None and out_html.exists() else "",
                "resolved_npz_path": str(npz_path) if npz_path is not None else "",
                "png_ok": int(png_ok),
                "html_ok": int(html_ok),
                "png_error": png_error,
                "html_error": html_error,
            })
            index_rows.append(row)

            if not png_ok or ((not args.no_html) and not html_ok):
                failures.append(row)

        except Exception as exc:
            try:
                first = sample_df.iloc[0]
                sample_id = first.get("sample_id", "")
            except Exception:
                sample_id = str(sample_key)
            failures.append({
                "sample_key": str(sample_key),
                "sample_id": sample_id,
                "error": str(exc),
            })
            print(f"[FAIL SAMPLE] {sample_key}: {exc}")

    pd.DataFrame(index_rows).to_csv(out_dir / "face_region_side_anomaly_overlay_index.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "face_region_side_anomaly_overlay_failures.csv", index=False)

    print(f"[DONE] Exported index rows: {len(index_rows)}")
    print(f"[DONE] Failures/partial failures: {len(failures)}")
    print(f"[DONE] Output: {out_dir}")
    print(f"[DONE] Index: {out_dir / 'face_region_side_anomaly_overlay_index.csv'}")


if __name__ == "__main__":
    main()
