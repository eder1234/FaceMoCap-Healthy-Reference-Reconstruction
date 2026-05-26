#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
healthy_reference_reconstructor.py

Reusable healthy-reference reconstruction utilities for FaceMoCap movement
comparison.

This module accepts a raw FaceMoCap CSV path and movement label, reuses the
canonical raw-CSV preprocessing from build_ae_dataset_v2.py, and returns:

  - observed target face, facial markers only, shape (100,105,3)
  - healthy mean trajectory, facial markers only, independent of target morphology
  - target-morphology trajectory driven by mean healthy velocity
  - AE-v3 healthy-twin reconstruction

The reference cohort is the metadata "reference" split, represented in the
existing ae_dataset_v2 outputs as X_train_healthy.npy.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import build_ae_dataset_v2 as ae_prep
import predict_healthy_twin_v3 as twin_v3


def _load_facemocap_csv_numpy(
    csv_path: Path,
    skiprows: int = 5,
    drop_first_cols: int = 2,
    n_markers_keep: int = 108,
) -> np.ndarray:
    """Numpy fallback matching build_ae_dataset_v2.load_facemocap_csv."""
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=skiprows, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] <= drop_first_cols:
        raise ValueError(f"CSV has too few columns after dropping metadata columns: {csv_path}")
    arr = arr[:, drop_first_cols:]
    if arr.shape[1] % 3 != 0:
        raise ValueError(f"Coordinate columns are not a multiple of 3: {csv_path}")
    n_markers = arr.shape[1] // 3
    if n_markers < n_markers_keep:
        raise ValueError(f"Expected at least {n_markers_keep} markers, got {n_markers}: {csv_path}")
    arr = arr[:, : n_markers_keep * 3]
    return arr.reshape(arr.shape[0], n_markers_keep, 3)


def _read_envelope_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _marker_ref_arrays_from_records(records: list[dict[str, str]], n_markers: int) -> Dict[str, np.ndarray]:
    out = {k: np.full(n_markers, np.nan, dtype=np.float32) for k in ["amp_low", "amp_high", "amp_scale", "rmse_high", "rmse_scale"]}
    for row in records:
        raw_id = row.get("facial_marker_id", row.get("marker_id", ""))
        try:
            j = int(float(raw_id))
        except Exception:
            continue
        if 0 <= j < n_markers:
            for key in out:
                try:
                    out[key][j] = float(row.get(key, np.nan))
                except Exception:
                    out[key][j] = np.nan
    out["amp_scale"] = np.where(np.isfinite(out["amp_scale"]) & (out["amp_scale"] > 1e-9), out["amp_scale"], 1.0)
    out["rmse_scale"] = np.where(np.isfinite(out["rmse_scale"]) & (out["rmse_scale"] > 1e-9), out["rmse_scale"], 1.0)
    return out


@dataclass
class ProcessedTarget:
    """Canonical aligned target representation."""

    movement: str
    csv_path: Path
    observed_face: np.ndarray
    displacement: np.ndarray
    mask: np.ndarray
    neutral_face: np.ndarray
    qc: Dict[str, object]


@dataclass
class HealthyReconstructions:
    """Four facial point-cloud trajectories used by later visualization/metrics."""

    movement: str
    sample_id: str
    observed_face: np.ndarray
    healthy_mean_trajectory: np.ndarray
    healthy_mean_velocity_face: np.ndarray
    aev3_healthy_reconstruction: np.ndarray
    target_displacement: np.ndarray
    target_neutral_face: np.ndarray
    target_mask: np.ndarray
    mean_velocity: np.ndarray
    mean_velocity_displacement: np.ndarray
    aev3_projected_neutral: np.ndarray
    aev3_projected_displacement: np.ndarray
    aev3_abnormal_mask: np.ndarray
    aev3_abnormal_severity: np.ndarray
    qc: Dict[str, object]


class HealthyReferenceReconstructor:
    """
    Generate the three healthy comparator trajectories for one raw target CSV.

    Parameters default to the existing project outputs. The class deliberately
    reuses build_ae_dataset_v2.process_sample for alignment/windowing/resampling.
    """

    def __init__(
        self,
        ae_dataset_dir: str | Path = "align_mean_movement_refactor/ae_dataset_v2",
        mean_ref_root: str | Path = "align_mean_movement_refactor/output_healthy_ref_split_1b",
        cache_dir: str | Path = "align_mean_movement_refactor/healthy_reference_cache",
        static_model_dir: str | Path = "align_mean_movement_refactor/static_neutral_ae_v3",
        dynamic_model_dir: str | Path = "align_mean_movement_refactor/dynamic_motion_projector_v3",
        device: str = "auto",
        neutral_identity_blend: float = 0.35,
        motion_anchor_strength: float = 0.30,
        abnormal_threshold: float = 0.25,
        extra_mask_fraction: float = 0.15,
        build_config_path: Optional[str | Path] = None,
        seed: int = 7,
    ) -> None:
        self.ae_dataset_dir = Path(ae_dataset_dir)
        self.mean_ref_root = Path(mean_ref_root)
        self.cache_dir = Path(cache_dir)
        self.static_model_dir = Path(static_model_dir)
        self.dynamic_model_dir = Path(dynamic_model_dir)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if build_config_path is None:
            build_config_path = self.ae_dataset_dir / "build_config.json"
        self.preprocess_args = self._load_preprocess_args(Path(build_config_path))
        self.preprocess_args.ref_root = str(self.mean_ref_root)
        # Keep the canonical build_ae_dataset_v2 preprocessing flow, but avoid a
        # hard dependency on pandas' CSV parser in environments where it is broken.
        ae_prep.load_facemocap_csv = _load_facemocap_csv_numpy

        self.aev3_args = argparse.Namespace(
            neutral_identity_blend=float(neutral_identity_blend),
            motion_anchor_strength=float(motion_anchor_strength),
            abnormal_threshold=float(abnormal_threshold),
            extra_mask_fraction=float(extra_mask_fraction),
        )
        self.device = self._resolve_device(device)
        self._static_model = None
        self._static_ckpt = None
        self._dynamic_models: Dict[str, object] = {}
        self._dynamic_ckpts: Dict[str, Dict[str, object]] = {}

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            return twin_v3.torch.device("cuda" if twin_v3.torch.cuda.is_available() else "cpu")
        return twin_v3.torch.device(device)

    @staticmethod
    def _load_preprocess_args(config_path: Path) -> argparse.Namespace:
        if not config_path.exists():
            raise FileNotFoundError(f"Missing AE dataset build config: {config_path}")
        cfg = json.loads(config_path.read_text())
        return argparse.Namespace(**cfg)

    @staticmethod
    def normalize_movement(movement: object) -> str:
        return ae_prep.normalize_movement(movement).upper()

    def build_reference_cache(self, movements: Optional[list[str]] = None, force: bool = False) -> None:
        """Compute and save per-movement mean trajectory and mean velocity references."""
        if movements is None:
            movements = [m.upper() for m in getattr(self.preprocess_args, "movements", ["M1", "M2", "M3", "M4", "M5"])]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for movement in movements:
            self._load_references(movement, force=force)

    def preprocess_csv(self, csv_path: str | Path, movement: object) -> ProcessedTarget:
        """Run canonical raw CSV processing and return facial-only target arrays."""
        movement = self.normalize_movement(movement)
        csv_path = Path(csv_path)
        row_data = {
            "complete_filepath": str(csv_path),
            "filename": str(csv_path.name),
            "participant_id": "",
            "condition": "",
            str(self.preprocess_args.reference_split_col): "",
            "acquisition_date": "",
        }
        row = row_data
        X, M, N, qc = ae_prep.process_sample(row, self.preprocess_args, movement, self.rng)
        observed = (N[None, :, :] + X).astype(np.float32)
        return ProcessedTarget(
            movement=movement,
            csv_path=csv_path,
            observed_face=observed,
            displacement=X.astype(np.float32),
            mask=M.astype(np.uint8),
            neutral_face=N.astype(np.float32),
            qc=qc,
        )

    def reconstruct_from_csv(self, csv_path: str | Path, movement: object) -> HealthyReconstructions:
        """Preprocess a raw target CSV and generate all healthy comparator trajectories."""
        target = self.preprocess_csv(csv_path, movement)
        return self.reconstruct_processed(target)

    def reconstruct_processed(self, target: ProcessedTarget) -> HealthyReconstructions:
        """Generate healthy comparator trajectories for a preprocessed target."""
        refs = self._load_references(target.movement)
        aev3 = self._predict_aev3(target)
        sample_id = str(target.qc.get("sample_id", target.csv_path.stem))
        return HealthyReconstructions(
            movement=target.movement,
            sample_id=sample_id,
            observed_face=target.observed_face,
            healthy_mean_trajectory=refs["mean_trajectory"].copy(),
            healthy_mean_velocity_face=(target.neutral_face[None, :, :] + refs["mean_velocity_displacement"]).astype(np.float32),
            aev3_healthy_reconstruction=aev3["twin_face"],
            target_displacement=target.displacement,
            target_neutral_face=target.neutral_face,
            target_mask=target.mask,
            mean_velocity=refs["mean_velocity"].copy(),
            mean_velocity_displacement=refs["mean_velocity_displacement"].copy(),
            aev3_projected_neutral=aev3["projected_neutral"],
            aev3_projected_displacement=aev3["projected_displacement"],
            aev3_abnormal_mask=aev3["abnormal_mask"],
            aev3_abnormal_severity=aev3["abnormal_severity"],
            qc=target.qc,
        )

    def _reference_cache_path(self, movement: str) -> Path:
        return self.cache_dir / movement / "healthy_references.npz"

    def _load_references(self, movement: str, force: bool = False) -> Dict[str, np.ndarray]:
        movement = self.normalize_movement(movement)
        path = self._reference_cache_path(movement)
        if path.exists() and not force:
            data = np.load(path)
            return {
                "mean_trajectory": data["mean_trajectory"].astype(np.float32),
                "mean_velocity": data["mean_velocity"].astype(np.float32),
                "mean_velocity_displacement": data["mean_velocity_displacement"].astype(np.float32),
            }
        refs = self._compute_references(movement)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **refs)
        meta = {
            "movement": movement,
            "mean_trajectory_source": str(self.mean_ref_root / movement / "mean_healthy.npy"),
            "mean_velocity_source": str(self.ae_dataset_dir / movement / "X_train_healthy.npy"),
            "cohort": "condition == healthy and reference_split == reference",
            "n_frames": int(refs["mean_trajectory"].shape[0]),
            "n_markers": int(refs["mean_trajectory"].shape[1]),
        }
        (path.parent / "healthy_references.json").write_text(json.dumps(meta, indent=2))
        return refs

    def _compute_references(self, movement: str) -> Dict[str, np.ndarray]:
        mean_path = self.mean_ref_root / movement / "mean_healthy.npy"
        if not mean_path.exists():
            raise FileNotFoundError(f"Missing healthy mean trajectory: {mean_path}")
        mean_full = np.load(mean_path).astype(np.float32)
        dental_n = int(self.preprocess_args.dental_n)
        n_markers = int(self.preprocess_args.n_markers)
        n_frames = int(self.preprocess_args.n_frames)
        mean_full = mean_full[:, :n_markers, :]
        if mean_full.shape[0] != n_frames:
            mean_full = ae_prep.resample_sequence_nan_robust(mean_full, n_frames).astype(np.float32)
        mean_trajectory = mean_full[:, dental_n:n_markers, :].astype(np.float32)

        x_path = self.ae_dataset_dir / movement / "X_train_healthy.npy"
        m_path = self.ae_dataset_dir / movement / "M_train_healthy.npy"
        if not x_path.exists() or not m_path.exists():
            raise FileNotFoundError(f"Missing reference healthy arrays for {movement}: {x_path} / {m_path}")
        X = np.load(x_path).astype(np.float32)
        M = np.load(m_path).astype(bool)
        if X.shape[0] == 0:
            raise RuntimeError(f"No reference healthy samples for {movement}")

        velocity = X[:, 1:, :, :] - X[:, :-1, :, :]
        valid_velocity = M[:, 1:, :] & M[:, :-1, :]
        velocity = np.where(valid_velocity[..., None], velocity, np.nan)
        with np.errstate(all="ignore"):
            mean_velocity = np.nanmean(velocity, axis=0)
        mean_velocity = np.nan_to_num(mean_velocity, nan=0.0).astype(np.float32)

        mean_velocity_displacement = np.zeros_like(mean_trajectory, dtype=np.float32)
        mean_velocity_displacement[1:] = np.cumsum(mean_velocity, axis=0)
        return {
            "mean_trajectory": mean_trajectory,
            "mean_velocity": mean_velocity,
            "mean_velocity_displacement": mean_velocity_displacement,
        }

    def _load_aev3_models(self, movement: str) -> None:
        if self._static_model is None or self._static_ckpt is None:
            self._static_model, self._static_ckpt = twin_v3.load_static(self.static_model_dir / "model.pt", self.device)
        if movement not in self._dynamic_models:
            model, ckpt = twin_v3.load_dynamic(self.dynamic_model_dir / movement / "model.pt", self.device)
            self._dynamic_models[movement] = model
            self._dynamic_ckpts[movement] = ckpt

    def _predict_aev3(self, target: ProcessedTarget) -> Dict[str, np.ndarray]:
        movement = target.movement
        self._load_aev3_models(movement)
        static_model = self._static_model
        static_ck = self._static_ckpt
        dyn_model = self._dynamic_models[movement]
        dyn_ck = self._dynamic_ckpts[movement]

        X = target.displacement[None, :, :, :].astype(np.float32)
        M = target.mask[None, :, :].astype(np.uint8)
        N = target.neutral_face[None, :, :].astype(np.float32)
        Xref = np.load(self.ae_dataset_dir / movement / "reference_displacement.npy").astype(np.float32)[:, : X.shape[2], :]
        env_records = _read_envelope_records(self.ae_dataset_dir / movement / "reference_envelope_per_marker.csv")
        ref = _marker_ref_arrays_from_records(env_records, X.shape[2])

        Mneutral = np.isfinite(N).all(axis=-1).astype(np.uint8)
        Nzero = np.where(np.isfinite(N), N, 0.0).astype(np.float32)

        Nmean = np.asarray(static_ck["mean"], dtype=np.float32)
        Nstd = np.asarray(static_ck["std"], dtype=np.float32)
        Dmean = np.asarray(dyn_ck["mean"], dtype=np.float32)
        Dstd = np.asarray(dyn_ck["std"], dtype=np.float32)

        ZN = twin_v3.normalize_static(Nzero, Mneutral, Nmean, Nstd)
        with twin_v3.torch.no_grad():
            pred_neutral_norm = static_model(
                twin_v3.torch.from_numpy(ZN).to(self.device),
                twin_v3.torch.from_numpy(Mneutral.astype(np.float32)).to(self.device),
            ).cpu().numpy()
        neutral_pred = twin_v3.denorm_static(pred_neutral_norm, Nmean, Nstd)
        neutral_projected = (
            (1.0 - self.aev3_args.neutral_identity_blend) * neutral_pred
            + self.aev3_args.neutral_identity_blend * Nzero
        ).astype(np.float32)

        abnormal, severity = twin_v3.abnormal_mask(X, M, Xref, ref, self.aev3_args)
        Min = M.copy()
        Min[abnormal[:, None, :].repeat(X.shape[1], axis=1)] = 0
        Xin = X.copy()
        Xin[Min == 0] = 0.0
        Z = twin_v3.normalize_dyn(Xin, Min, Dmean, Dstd)
        with twin_v3.torch.no_grad():
            pred_dyn_norm = dyn_model(
                twin_v3.torch.from_numpy(Z).to(self.device),
                twin_v3.torch.from_numpy(Min.astype(np.float32)).to(self.device),
            ).cpu().numpy()
        displacement_pred = twin_v3.denorm_dyn(pred_dyn_norm, Dmean, Dstd)
        displacement_projected = (
            (1.0 - self.aev3_args.motion_anchor_strength) * displacement_pred
            + self.aev3_args.motion_anchor_strength * Xref[None, :, :, :]
        ).astype(np.float32)
        twin_face = (neutral_projected[:, None, :, :] + displacement_projected).astype(np.float32)

        return {
            "twin_face": twin_face[0],
            "projected_neutral": neutral_projected[0],
            "projected_displacement": displacement_projected[0],
            "abnormal_mask": abnormal[0].astype(np.uint8),
            "abnormal_severity": severity[0].astype(np.float32),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate healthy comparator trajectories for one FaceMoCap CSV.")
    ap.add_argument("--csv", required=True, help="Raw FaceMoCap CSV path.")
    ap.add_argument("--movement", required=True, help="Movement label, e.g. M5 or 5.")
    ap.add_argument("--out_npz", default=None, help="Optional output NPZ with the four trajectories.")
    ap.add_argument("--force_cache", action="store_true", help="Recompute cached healthy references first.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    reconstructor = HealthyReferenceReconstructor()
    if args.force_cache:
        reconstructor.build_reference_cache(force=True)
    result = reconstructor.reconstruct_from_csv(args.csv, args.movement)
    if args.out_npz:
        out = Path(args.out_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            observed_face=result.observed_face,
            healthy_mean_trajectory=result.healthy_mean_trajectory,
            healthy_mean_velocity_face=result.healthy_mean_velocity_face,
            aev3_healthy_reconstruction=result.aev3_healthy_reconstruction,
            target_mask=result.target_mask,
            target_neutral_face=result.target_neutral_face,
            target_displacement=result.target_displacement,
            mean_velocity=result.mean_velocity,
            mean_velocity_displacement=result.mean_velocity_displacement,
            aev3_projected_neutral=result.aev3_projected_neutral,
            aev3_projected_displacement=result.aev3_projected_displacement,
            aev3_abnormal_mask=result.aev3_abnormal_mask,
            aev3_abnormal_severity=result.aev3_abnormal_severity,
        )
        print(f"[OK] Wrote {out}")
    print(
        "[OK] "
        f"{result.movement} {result.sample_id}: "
        f"observed={result.observed_face.shape}, "
        f"mean={result.healthy_mean_trajectory.shape}, "
        f"mean_velocity={result.healthy_mean_velocity_face.shape}, "
        f"aev3={result.aev3_healthy_reconstruction.shape}"
    )


if __name__ == "__main__":
    main()
