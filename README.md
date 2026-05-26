# FaceMoCap Healthy Reference Reconstruction

This repository builds healthy facial-movement baselines from raw FaceMoCap point-cloud CSVs and uses them to reconstruct healthy comparator trajectories for evaluation samples.

The pipeline produces four aligned facial point-cloud trajectories per evaluation sample:

1. observed target
2. healthy mean trajectory
3. target morphology driven by healthy mean velocity
4. AE-v3 healthy reconstruction

All reconstruction outputs use the 105 facial markers only. The first 3 markers in the raw 108-marker files are dental-frame support markers and are used for stabilization, then excluded from facial outputs.

## Required Inputs

You need:

- raw FaceMoCap CSV files
- a metadata CSV
- `semantic_facial_labels.csv`

The metadata CSV must contain at least:

```text
complete_filepath,filename,participant_id,acquisition_date,facial_movement,single_movement,condition,valid_for_processing
```

After running the split script, it will also contain:

```text
reference_split
```

Valid `condition` values used by the pipeline:

```text
healthy
pathological
```

Valid `reference_split` values:

```text
reference
evaluation
excluded
```

The `reference` split is used to compute/train healthy baselines. The `evaluation` split is held out for evaluation and visualization. `excluded` rows are ignored.

## Environment

Create an environment with Python, NumPy, pandas, Plotly, and PyTorch. In the original working setup this environment was named:

```bash
conda activate facemocap_ai
```

Recommended packages:

```text
numpy
pandas
plotly
torch
scipy
scikit-learn
matplotlib
```

## Repository Files

Core pipeline files:

```text
make_reference_split.py
align_and_mean_movement_reference_split_1b.py
build_ae_dataset_v2.py
build_healthy_twin_dataset_v3.py
train_static_neutral_ae_v3.py
train_dynamic_motion_projector_v3.py
predict_healthy_twin_v3.py
healthy_reference_reconstructor.py
export_four_cloud_eval_overlays.py
semantic_facial_labels.csv
```

Do not commit raw datasets, trained models, caches, or generated HTML outputs unless intentionally publishing a release artifact.

## Pipeline

Set a project root variable for readability:

```bash
ROOT=/path/to/Data_FaceMoCap
cd "$ROOT"
conda activate facemocap_ai
```

### 1. Create Reference/Evaluation Split

```bash
python make_reference_split.py \
  --metadata facemocap_metadata.csv \
  --output facemocap_metadata_reference_split.csv \
  --eval_frac 0.20 \
  --seed 42
```

This assigns usable healthy participants to `reference` or `evaluation`, assigns usable pathological rows to `evaluation`, and marks unusable rows as `excluded`.

### 2. Build Healthy Mean Trajectory

Choose a template CSV. This should be a valid FaceMoCap CSV used as the neutral alignment template.

```bash
python align_and_mean_movement_reference_split_1b.py \
  --metadata facemocap_metadata_reference_split.csv \
  --template_csv /path/to/template.csv \
  --out_dir align_mean_movement_refactor/output_healthy_ref_split_1b \
  --root_override "$ROOT" \
  --movements M1 M2 M3 M4 M5 \
  --try_yaw_flip
```

Important outputs:

```text
align_mean_movement_refactor/output_healthy_ref_split_1b/M*/mean_healthy.npy
align_mean_movement_refactor/output_healthy_ref_split_1b/M*/reference_envelope_per_marker.csv
align_mean_movement_refactor/output_healthy_ref_split_1b/M*/reference_stack_and_envelope.npz
```

`mean_healthy.npy` is the single healthy mean trajectory per movement.

### 3. Build Canonical Aligned AE Dataset

```bash
python build_ae_dataset_v2.py \
  --metadata facemocap_metadata_reference_split.csv \
  --ref_root align_mean_movement_refactor/output_healthy_ref_split_1b \
  --out_dir align_mean_movement_refactor/ae_dataset_v2 \
  --root_override "$ROOT" \
  --movements M1 M2 M3 M4 M5 \
  --try_yaw_flip
```

This is the canonical raw CSV to aligned/resampled facial displacement step.

Important outputs per movement:

```text
X_train_healthy.npy
M_train_healthy.npy
N_train_healthy.npy
X_eval_healthy.npy
M_eval_healthy.npy
N_eval_healthy.npy
X_pathological.npy
M_pathological.npy
N_pathological.npy
reference_displacement.npy
reference_envelope_per_marker.csv
```

Here, `train_healthy` corresponds to `condition == healthy` and `reference_split == reference`.

### 4. Build AE-v3 Dataset

```bash
python build_healthy_twin_dataset_v3.py \
  --v2_dataset_dir align_mean_movement_refactor/ae_dataset_v2 \
  --semantic_labels semantic_facial_labels.csv \
  --out_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --movements M1 M2 M3 M4 M5
```

This reorganizes the v2 dataset into static neutral morphology and dynamic displacement components for AE-v3.

### 5. Train AE-v3 Static Neutral Model

```bash
python train_static_neutral_ae_v3.py \
  --dataset_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --out_dir align_mean_movement_refactor/static_neutral_ae_v3 \
  --epochs 300 \
  --batch_size 16 \
  --patience 40
```

### 6. Train AE-v3 Dynamic Motion Models

```bash
python train_dynamic_motion_projector_v3.py \
  --dataset_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --out_dir align_mean_movement_refactor/dynamic_motion_projector_v3 \
  --movements M1 M2 M3 M4 M5 \
  --epochs 300 \
  --batch_size 8 \
  --patience 40
```

This trains one dynamic healthy-motion model per movement.

### 7. Optional: Run AE-v3 Evaluation Tables

```bash
python predict_healthy_twin_v3.py \
  --dataset_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --static_model_dir align_mean_movement_refactor/static_neutral_ae_v3 \
  --dynamic_model_dir align_mean_movement_refactor/dynamic_motion_projector_v3 \
  --out_dir align_mean_movement_refactor/healthy_twin_v3_eval \
  --semantic_labels semantic_facial_labels.csv \
  --movements M1 M2 M3 M4 M5 \
  --groups eval_healthy pathological
```

This writes AE-v3 score tables and `.npz` predictions.

### 8. Export Four-Cloud Evaluation Overlays

```bash
python export_four_cloud_eval_overlays.py \
  --metadata facemocap_metadata_reference_split.csv \
  --out_dir align_mean_movement_refactor/four_cloud_eval_overlays
  --save_npz
```

Open:

```text
align_mean_movement_refactor/four_cloud_eval_overlays/index.html
```

Each generated HTML overlays:

- observed target
- healthy mean trajectory
- healthy mean velocity on target morphology
- AE-v3 healthy reconstruction

The HTML includes controls to show/hide individual point clouds.

#### 9. Follow-up (patients with more than one visit)

```bash
python eval_aev3_followup.py \
  --overlay_root align_mean_movement_refactor/four_cloud_eval_overlays \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --metadata facemocap_metadata_reference_split.csv \
  --semantic_labels semantic_facial_labels.csv \
  --out_dir align_mean_movement_refactor/aev3_followup_eval \
  --fs_hz 100 \
  --lowpass_hz 8 \
  --min_dates 2
```

## Programmatic Use

Use `HealthyReferenceReconstructor` directly in downstream metric or visualization code:

```python
from healthy_reference_reconstructor import HealthyReferenceReconstructor

reconstructor = HealthyReferenceReconstructor()
result = reconstructor.reconstruct_from_csv(
    "/path/to/sample_M5.csv",
    "M5",
)

observed = result.observed_face
mean_traj = result.healthy_mean_trajectory
mean_vel = result.healthy_mean_velocity_face
aev3 = result.aev3_healthy_reconstruction
mask = result.target_mask
```

All four trajectory arrays have shape:

```text
(100, 105, 3)
```

## Generated Outputs To Ignore

Add these to `.gitignore`:

```text
Sujets_Sains/
Sujets_Patho/
align_mean_movement_refactor/
models/
cache_v3/
__pycache__/
*.pyc
*.npy
*.npz
*.pt
*.pth
*.keras
*.html
```

If you want to version small examples, place them under a separate `examples/` directory and keep them minimal.

## Notes

- The healthy mean velocity reference is computed from `X_train_healthy.npy`, which corresponds to healthy rows in the `reference` split.
- `evaluation` rows are not used to compute healthy baselines or train AE-v3 models.
- The first 3 raw markers are dental support markers. They are used for coordinate stabilization but are not part of the final 105-marker facial trajectories.
- Some raw samples may fail preprocessing QC, especially if dental-frame markers are missing or invalid. These failures are expected and are logged by the exporter.
