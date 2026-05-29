# FaceMoCap Healthy Reference Reconstruction

This repository builds healthy facial-movement references from raw FaceMoCap point-cloud CSV files and uses them to generate healthy comparator trajectories for evaluation samples.

The pipeline produces four aligned facial point-cloud trajectories per evaluation sample:

1. observed target;
2. healthy mean trajectory;
3. target morphology driven by healthy mean displacement;
4. AE healthy reconstruction.

All reconstruction outputs use the 105 facial markers only. The first 3 markers in the raw 108-marker files are dental-frame support markers. They are used for stabilization and then excluded from the final facial outputs.

Some script and directory names still contain `v3` for backward compatibility with previous experiments. In this README, the learned healthy reconstruction model is referred to as the **AE model**.

## Required Inputs

You need:

- raw FaceMoCap CSV files;
- a metadata CSV;
- `semantic_facial_labels.csv`;
- a neutral schematic face image for clinical overlays, for example `neutral_face.png`.

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

Create an environment with Python, NumPy, pandas, Plotly, PyTorch, SciPy, scikit-learn, matplotlib, and Pillow. In the original working setup this environment was named:

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
pillow
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

Clinician-oriented anatomical analysis and visualization files:

```text
export_ae_region_side_metrics_csv.py
compute_healthy_eval_region_side_percentiles.py
export_face_region_side_heatmap_overlays.py
```

Do not commit raw datasets, trained models, caches, generated NPZ files, generated HTML files, generated PNG overlays, or full generated output folders unless intentionally publishing a release artifact.

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

### 4. Build AE Dataset

```bash
python build_healthy_twin_dataset_v3.py \
  --v2_dataset_dir align_mean_movement_refactor/ae_dataset_v2 \
  --semantic_labels semantic_facial_labels.csv \
  --out_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --movements M1 M2 M3 M4 M5
```

This reorganizes the v2 dataset into static neutral morphology and dynamic displacement components for the AE model.

### 5. Train Static Neutral AE Model

```bash
python train_static_neutral_ae_v3.py \
  --dataset_dir align_mean_movement_refactor/healthy_twin_dataset_v3 \
  --out_dir align_mean_movement_refactor/static_neutral_ae_v3 \
  --epochs 300 \
  --batch_size 16 \
  --patience 40
```

### 6. Train Dynamic Motion Models

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

### 7. Optional: Run AE Evaluation Tables

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

This writes AE score tables and `.npz` predictions.

### 8. Export Four-Cloud Evaluation Overlays

```bash
python export_four_cloud_eval_overlays.py \
  --metadata facemocap_metadata_reference_split.csv \
  --out_dir align_mean_movement_refactor/four_cloud_eval_overlays \
  --save_npz
```

Open:

```text
align_mean_movement_refactor/four_cloud_eval_overlays/index.html
```

Each generated HTML overlays:

- observed target;
- healthy mean trajectory;
- target morphology with healthy mean displacement;
- AE healthy reconstruction.

The HTML includes controls to show or hide individual point clouds. When `--save_npz` is used, the script also writes one reconstruction NPZ file per sample. These NPZ files are required by the anatomical metric and clinician-oriented visualization scripts.

### 9. Follow-up Evaluation

For patients with more than one visit:

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

and

```bash
python eval_aev3_followup_matched.py \
  --overlay_root align_mean_movement_refactor/four_cloud_eval_overlays \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --metadata facemocap_metadata_reference_split.csv \
  --semantic_labels semantic_facial_labels.csv \
  --out_dir align_mean_movement_refactor/aev3_followup_eval_matched \
  --fs_hz 100 \
  --lowpass_hz 8 \
  --min_dates 2
```

### 10. Export AE Region-Side Anatomical Metrics

This step computes AE-based anatomical metrics for selected samples. Metrics are computed after applying an 8 Hz low-pass filter to both the observed target trajectory and the AE healthy reconstruction.

For selected pathological follow-up participants:

```bash
python export_ae_region_side_metrics_csv.py \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --semantic_labels semantic_facial_labels.csv \
  --metadata facemocap_metadata_reference_split.csv \
  --project_root . \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv \
  --patients <PATHOLOGICAL_ID_1> <PATHOLOGICAL_ID_2> \
  --movements M1 M2 M3 M4 M5 \
  --condition pathological
```

Important outputs:

```text
ae_region_side_metrics_matrix_raw.csv
ae_region_side_metrics_long_selected.csv
ae_region_side_metrics_matrix_softlog_0_1.csv
ae_region_side_metrics_matrix_softlog_0_1_reliable.csv
ae_region_side_sample_manifest.csv
ae_region_side_failures.csv
```

The key reusable file for percentile-based anomaly visualization is:

```text
ae_region_side_metrics_matrix_raw.csv
```

### 11. Compute Healthy-Evaluation Percentile References

This step estimates healthy-evaluation percentile thresholds for each movement, anatomical metric, and region-side group.

```bash
python compute_healthy_eval_region_side_percentiles.py \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --semantic_labels semantic_facial_labels.csv \
  --project_root . \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv \
  --movements M1 M2 M3 M4 M5
```

Important outputs:

```text
healthy_eval_region_side_percentiles.csv
healthy_eval_region_side_raw_values_long.csv
healthy_eval_region_side_matrix_raw.csv
healthy_eval_region_side_failures.csv
healthy_eval_region_side_percentile_manifest.csv
```

The percentile file contains healthy-evaluation statistics for each:

```text
movement × metric × region-side
```

including:

```text
healthy_n, reliable, p50, p75, p90, p95, p99, max
```

For pathological samples, use the full healthy-evaluation percentile file. For a healthy sanity check, use leave-one-subject-out thresholds. For example, to test a healthy participant fairly:

```bash
python compute_healthy_eval_region_side_percentiles.py \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --semantic_labels semantic_facial_labels.csv \
  --project_root . \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv \
  --movements M1 M2 M3 M4 M5 \
  --loo_participant <HEALTHY_ID> \
  --min_n_reliable 5
```

This creates:

```text
healthy_eval_excluding_<HEALTHY_ID>_region_side_percentiles.csv
```

### 12. Export Face-Overlay Anomaly Maps and Target-vs-AE HTML

This step creates one schematic face overlay per sample. Each anatomical region-side contains a local 1 × 5 mini-heatmap with the following metric order:

```text
trajectory | amplitude | hypokinesia | hyperkinesia | counter-direction
```

Each cell is classified relative to the healthy-evaluation distribution for the same movement, metric, and region-side:

```text
0 = value <= healthy p75
1 = p75 < value <= p90
2 = p90 < value <= p95
3 = p95 < value <= healthy max
4 = value > healthy max
NaN = unavailable or unreliable
```

For selected pathological samples:

```bash
python export_face_region_side_heatmap_overlays.py \
  --matrix_csv align_mean_movement_refactor/ae_region_side_clinical_csv/ae_region_side_metrics_matrix_raw.csv \
  --healthy_percentiles align_mean_movement_refactor/ae_region_side_clinical_csv/healthy_eval_region_side_percentiles.csv \
  --template_image neutral_face.png \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_face_overlays_percentile_pathological \
  --project_root . \
  --patients <PATHOLOGICAL_ID_1> <PATHOLOGICAL_ID_2> \
  --movements M1 M2 M3 M4 M5 \
  --conditions pathological
```

For a healthy leave-one-subject-out sanity check, first generate that participant’s raw anatomical metric matrix:

```bash
python export_ae_region_side_metrics_csv.py \
  --overlay_index align_mean_movement_refactor/four_cloud_eval_overlays/overlay_index.csv \
  --semantic_labels semantic_facial_labels.csv \
  --metadata facemocap_metadata_reference_split.csv \
  --project_root . \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_csv_<HEALTHY_ID> \
  --patients <HEALTHY_ID> \
  --movements M1 M2 M3 M4 M5 \
  --condition healthy
```

Then generate the overlays using the participant-excluded percentile file:

```bash
python export_face_region_side_heatmap_overlays.py \
  --matrix_csv align_mean_movement_refactor/ae_region_side_clinical_csv_<HEALTHY_ID>/ae_region_side_metrics_matrix_raw.csv \
  --healthy_percentiles align_mean_movement_refactor/ae_region_side_clinical_csv/healthy_eval_excluding_<HEALTHY_ID>_region_side_percentiles.csv \
  --template_image neutral_face.png \
  --out_dir align_mean_movement_refactor/ae_region_side_clinical_face_overlays_percentile_LOO_<HEALTHY_ID> \
  --project_root . \
  --patients <HEALTHY_ID> \
  --movements M1 M2 M3 M4 M5 \
  --conditions healthy
```

Important outputs:

```text
*_face_region_side_anomaly_overlay.png
*_target_vs_ae_overlay.html
ae_region_side_anomaly_class_long.csv
ae_region_side_anomaly_class_matrix.csv
face_region_side_anomaly_overlay_index.csv
face_region_side_anomaly_overlay_failures.csv
```

The companion HTML displays only the observed target and AE healthy reconstruction, with controls to show or hide each cloud. The displayed trajectories are low-pass filtered at 8 Hz to match the trajectories used for anatomical metric computation.

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
mean_disp = result.healthy_mean_displacement_face
ae = result.ae_healthy_reconstruction
mask = result.target_mask
```

Depending on legacy code versions, the displacement and AE fields may still be named:

```python
mean_disp = result.healthy_mean_velocity_face
ae = result.aev3_healthy_reconstruction
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
*_face_region_side_anomaly_overlay.png
*_face_region_side_heatmap_overlay.png
```

Do not ignore all CSV files globally, because metadata files and semantic label files are required inputs. Generated CSV outputs are already covered if they are written inside `align_mean_movement_refactor/`.

If you want to version small examples, place them under a separate `examples/` directory and keep them minimal.

## Notes

- The target-morphology healthy displacement reference is computed from healthy rows in the reference split.
- Evaluation rows are not used to compute healthy baselines or train the AE model.
- Healthy-evaluation rows are used only to estimate empirical anatomical percentile thresholds for anomaly visualization.
- For pathological samples, anomaly classes are computed relative to all healthy-evaluation participants.
- For healthy sanity checks, use leave-one-subject-out percentile thresholds so that the participant being visualized does not contribute to their own healthy reference distribution.
- The first 3 raw markers are dental support markers. They are used for coordinate stabilization but are not part of the final 105-marker facial trajectories.
- Some raw samples may fail preprocessing QC, especially if dental-frame markers are missing or invalid. These failures are expected and are logged by the exporter.
- The face-overlay PNG is an anatomical schematic, not a registered 2D projection of the 3D FaceMoCap markers.
