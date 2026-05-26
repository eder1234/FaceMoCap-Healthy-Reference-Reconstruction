#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_four_cloud_eval_overlays.py

Generate one HTML animation per evaluation sample, overlaying four facial
point-cloud trajectories:

  1) observed target
  2) healthy mean trajectory
  3) target morphology + healthy mean velocity
  4) AE-v3 healthy reconstruction

Samples come from facemocap_metadata_reference_split.csv with:
  reference_split == evaluation
  valid_for_processing == 1
  single_movement == 1

The heavy lifting is delegated to HealthyReferenceReconstructor.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from healthy_reference_reconstructor import HealthyReferenceReconstructor, HealthyReconstructions


CLOUDS = [
    ("observed_face", "Observed target", "#d62728", 4, 0.82),
    ("healthy_mean_trajectory", "Healthy mean trajectory", "#2ca02c", 4, 0.62),
    ("healthy_mean_velocity_face", "Healthy mean velocity on target", "#9467bd", 4, 0.68),
    ("aev3_healthy_reconstruction", "AE-v3 healthy reconstruction", "#1f77b4", 4, 0.72),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_movement(value: object) -> str:
    return HealthyReferenceReconstructor.normalize_movement(value)


def infer_movement_from_row(row: pd.Series) -> str:
    movement = normalize_movement(row.get("facial_movement", ""))
    if movement.startswith("M") and any(ch.isdigit() for ch in movement):
        return movement
    text = " ".join(str(row.get(k, "")) for k in ["filename", "complete_filepath"])
    match = re.search(r"(?:^|[^A-Za-z0-9])M0*([1-9]\d*)(?:[^A-Za-z0-9]|$)", text, flags=re.IGNORECASE)
    if match:
        return f"M{int(match.group(1))}"
    return ""


def safe_name(value: object, max_len: int = 150) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:max_len]


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def finite_axis_ranges(seqs: Iterable[np.ndarray], masks: Optional[Iterable[Optional[np.ndarray]]] = None, pad_frac: float = 0.08):
    pts = []
    if masks is None:
        masks = [None] * len(list(seqs))
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


def traces_at(result: HealthyReconstructions, frame_idx: int, hide_invalid_predictions: bool):
    traces = []
    for attr, label, color, size, opacity in CLOUDS:
        seq = getattr(result, attr)
        mask = result.target_mask if (attr == "observed_face" or hide_invalid_predictions) else None
        traces.append(trace_for(seq, frame_idx, mask, label, color, size, opacity))
    return traces


def make_overlay_html(
    result: HealthyReconstructions,
    out_html: Path,
    title: str,
    hide_invalid_predictions: bool = True,
    frame_duration: int = 60,
) -> None:
    T = int(result.observed_face.shape[0])
    seqs = [getattr(result, attr) for attr, *_ in CLOUDS]
    masks = [result.target_mask if (attr == "observed_face" or hide_invalid_predictions) else None for attr, *_ in CLOUDS]
    xr, yr, zr = finite_axis_ranges(seqs, masks)

    frames = [go.Frame(data=traces_at(result, t, hide_invalid_predictions), name=str(t), traces=[0, 1, 2, 3]) for t in range(T)]
    steps = [
        dict(
            method="animate",
            args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=str(t),
        )
        for t in range(T)
    ]
    fig = go.Figure(data=traces_at(result, 0, hide_invalid_predictions), frames=frames)
    note = "Predicted/reference clouds are hidden where the target marker/frame was invalid." if hide_invalid_predictions else "All finite predicted/reference points are shown."
    fig.update_layout(
        title=f"{title}<br><sup>{note}</sup>",
        scene=dict(
            xaxis=dict(range=xr, title="x"),
            yaxis=dict(range=yr, title="y"),
            zaxis=dict(range=zr, title="z"),
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
                        args=[None, dict(frame=dict(duration=frame_duration, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[dict(active=0, currentvalue=dict(prefix="Frame: "), pad=dict(t=45), steps=steps)],
        margin=dict(l=0, r=0, t=95, b=0),
        height=760,
    )

    plot_div = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="four_cloud_plot")
    controls = "\n".join(
        f"<label><input type='checkbox' class='cloud-toggle' data-trace='{i}' checked> "
        f"<span style='color:{html.escape(color)}'>{html.escape(label)}</span></label>"
        for i, (_, label, color, _, _) in enumerate(CLOUDS)
    )
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
    <button type="button" id="hide_refs">Target only</button>
  </div>
  {plot_div}
  <script>
    const plotId = "four_cloud_plot";
    function setTrace(traceIndex, isVisible) {{
      Plotly.restyle(plotId, {{visible: isVisible ? true : "legendonly"}}, [traceIndex]);
    }}
    document.querySelectorAll(".cloud-toggle").forEach(function(box) {{
      box.addEventListener("change", function() {{
        setTrace(Number(box.dataset.trace), box.checked);
      }});
    }});
    document.getElementById("show_all").addEventListener("click", function() {{
      document.querySelectorAll(".cloud-toggle").forEach(function(box) {{
        box.checked = true;
        setTrace(Number(box.dataset.trace), true);
      }});
    }});
    document.getElementById("hide_refs").addEventListener("click", function() {{
      document.querySelectorAll(".cloud-toggle").forEach(function(box) {{
        const idx = Number(box.dataset.trace);
        box.checked = idx === 0;
        setTrace(idx, idx === 0);
      }});
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(page, encoding="utf-8")


def select_evaluation_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.metadata)
    df = df.copy()
    df["movement_norm"] = df.apply(infer_movement_from_row, axis=1)
    df["condition_norm"] = df["condition"].astype(str).str.strip().str.lower()
    df["split_norm"] = df[args.reference_split_col].astype(str).str.strip().str.lower()
    ok = (
        (df["split_norm"] == args.evaluation_label.lower())
        & (pd.to_numeric(df["valid_for_processing"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df["single_movement"], errors="coerce").fillna(0).astype(int) == 1)
    )
    out = df[ok].copy()
    out = out[out["movement_norm"].astype(str).str.match(r"^M[0-9]+$")].copy()
    if args.movements:
        wanted = {normalize_movement(m) for m in args.movements}
        out = out[out["movement_norm"].isin(wanted)].copy()
    if args.conditions:
        wanted_cond = {str(c).strip().lower() for c in args.conditions}
        out = out[out["condition_norm"].isin(wanted_cond)].copy()
    out = out.sort_values(["movement_norm", "condition_norm", "participant_id", "filename"]).reset_index(drop=True)
    if args.limit and args.limit > 0:
        out = out.head(args.limit).copy()
    return out


def write_index(rows: List[Dict[str, object]], out_dir: Path) -> None:
    index_csv = out_dir / "overlay_index.csv"
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    links = []
    for r in rows:
        label = " | ".join(
            html.escape(str(r.get(k, "")))
            for k in ["movement", "condition", "participant_id", "filename", "sample_id"]
        )
        links.append(f"<li><a href='{html.escape(str(r['html_relpath']))}' target='_blank'>{label}</a></li>")
    body = "\n".join(links)
    (out_dir / "index.html").write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Four-cloud evaluation overlays</title>
<style>body{{font-family:Arial,sans-serif;margin:28px}} li{{margin:6px 0}}</style></head>
<body>
<h1>Four-cloud evaluation overlays</h1>
<p>Each page overlays observed target, healthy mean trajectory, healthy mean velocity on target, and AE-v3 healthy reconstruction.</p>
<p>Index CSV: <a href="overlay_index.csv">overlay_index.csv</a></p>
<ol>{body}</ol>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="facemocap_metadata_reference_split.csv")
    ap.add_argument("--out_dir", default="align_mean_movement_refactor/four_cloud_eval_overlays")
    ap.add_argument("--reference_split_col", default="reference_split")
    ap.add_argument("--evaluation_label", default="evaluation")
    ap.add_argument("--movements", nargs="*", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--conditions", nargs="*", default=None, help="Optional condition filter, e.g. healthy pathological.")
    ap.add_argument("--limit", type=int, default=0, help="Debug limit; 0 exports all selected samples.")
    ap.add_argument("--show_invalid_predictions", action="store_true")
    ap.add_argument("--force_reference_cache", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    rows_df = select_evaluation_rows(args)
    print(f"Selected evaluation samples: {len(rows_df)}")

    reconstructor = HealthyReferenceReconstructor()
    movements = sorted(rows_df["movement_norm"].unique().tolist())
    reconstructor.build_reference_cache(movements, force=args.force_reference_cache)

    index_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for i, row in rows_df.iterrows():
        movement = str(row["movement_norm"])
        condition = str(row["condition_norm"])
        csv_path = Path(str(row["complete_filepath"]))
        group = "eval_healthy" if condition == "healthy" else "pathological"
        print(f"[{i + 1}/{len(rows_df)}] {movement} {group} {row.get('participant_id', '')} {csv_path.name}")
        try:
            result = reconstructor.reconstruct_from_csv(csv_path, movement)
            out_sample_dir = out_dir / movement / group
            ensure_dir(out_sample_dir)
            sample_id = safe_name(result.sample_id)
            out_html = out_sample_dir / f"{sample_id}_four_cloud_overlay.html"
            title = f"{movement} | {group} | {row.get('participant_id', '')} | {csv_path.name} | {result.sample_id}"
            make_overlay_html(
                result,
                out_html,
                title=title,
                hide_invalid_predictions=not args.show_invalid_predictions,
            )
            index_rows.append(
                {
                    "movement": movement,
                    "condition": condition,
                    "group": group,
                    "participant_id": row.get("participant_id", ""),
                    "filename": row.get("filename", csv_path.name),
                    "complete_filepath": str(csv_path),
                    "sample_id": result.sample_id,
                    "html_path": str(out_html),
                    "html_relpath": relpath(out_html, out_dir),
                    "alignment_score": result.qc.get("alignment_score", np.nan),
                    "qc_valid_marker_frame_fraction": result.qc.get("qc_valid_marker_frame_fraction", np.nan),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "movement": movement,
                    "condition": condition,
                    "participant_id": row.get("participant_id", ""),
                    "filename": row.get("filename", csv_path.name),
                    "complete_filepath": str(csv_path),
                    "error": str(exc),
                }
            )
            print(f"  [FAIL] {exc}")

    write_index(index_rows, out_dir)
    pd.DataFrame(failures).to_csv(out_dir / "overlay_failures.csv", index=False)
    print(f"[OK] Wrote {len(index_rows)} overlays under {out_dir}")
    print(f"[OK] Failures: {len(failures)}")


if __name__ == "__main__":
    main()
