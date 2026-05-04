# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

DeepSno is a deep learning framework for the SNO+ particle physics detector. It provides neural network models, dataloaders, and training infrastructure to reconstruct particle interaction properties (position, time) from photomultiplier tube (PMT) detector data stored in ROOT files.

## Commands

```bash
# Install (requires Python 3.13)
pip install -e .

# Lint / format
ruff check deepsno/
ruff format deepsno/

# Train
deepsno --config config.yaml --model ModelClass train \
  --entity <wandb-entity> --project <wandb-project> \
  --checkpoint_dir /path/to/checkpoints --device cuda

# Predict
deepsno --config config.yaml --model ModelClass predict \
  --ckpt /path/to/checkpoints --device cuda
```

There is no automated test suite — validation is done via wandb during training.

## Architecture

### Entry & Training Flow

1. **`deepsno/main.py`** — CLI entry point using `jsonargparse`. Parses YAML configs (with Jinja2 templating via `deepsno/utils/jinja.py`), validates git commit hash for reproducibility, and delegates to `loops.train()` or `loops.predict()`.

2. **`deepsno/loops.py`** — Core training/validation/inference loops. `train()` handles gradient accumulation, AMP (`autocast_dtype`), gradient clipping, LR scheduling, checkpoint saving, and periodic validation. Checkpoints are saved when validation loss improves.

3. **`deepsno/utils/config_parse.py`** — Dynamic class instantiation from config dicts. All models, datasets, optimizers, and schedulers are specified by `class_path` + `init_args` in YAML and instantiated at runtime.

### Models (`deepsno/models/`)

- **`position_reco.py`** — `PositionReco`: Transformer-based position/time reconstruction using `SetEncoderVarlenPadded` backbone. Main production model.
- **`multihit.py`** — `MultiHit`: Advanced transformer with multi-head scaled dot-product attention for variable-length padded inputs.
- **`hit_time_autoencoder.py`** — `HitTimeAutoencoder`: Per-PMT time-walk correction using exponential + linear calibration model.
- **`transformers.py`** — Shared transformer building blocks (set encoder, attention layers).

### Data (`deepsno/data/`)

- **`datasets.py`** — `ChunkedUprootDataset`: Streaming ROOT file reader with configurable buffer sizes. Handles variable-length PMT hit data.
- **`multihit.py`** — Core multi-hit datasets, collation, and utilities. Key classes:
  - `UprootMultiFileDataset` — base iterable dataset; DDP- and worker-aware file sharding, reservoir-sampling shuffle.
  - `MultiHitDatasetBase` — adds nested per-PMT hit loading + voxelized track truth. Abstract `_make_pmt_inputs()` hook.
  - `MultiHitDatasetExpanded` / `MultiHitDatasetUnique` — expand-per-hit and unique-PMT subclasses of the above.
  - `MultiHitVertexDataset` — flat hit arrays (`hit_times`, `hit_ids`) + raw vertex truth from ROOT `vertices` branch.
  - `TrackStepDataset` — flat per-PMT hit loading (same as `MultiHitDatasetBase`) + RDP-compressed track step midpoints as targets.
  - `MultiHitVarlenCollate` — collate function converting padded per-event hits to varlen format (`cu_seqlens`, `max_seqlen`) for flash-attention kernels.
  - `rdp_compress_track(points, epsilon_xyz, epsilon_t)` — RDP simplification for a single (N,4) [x,y,z,t] track polyline.
- **`cuts.py`**, **`filter_pmts.py`**, **`pmt_info.py`** — Event selection, PMT quality filtering, and detector geometry.

### Track step structure (ROOT MC data)

In the ROOT trees produced for this project, `tracks["steps"]["position"]` stores the **END position** of each Geant4 step. The **first step is always zero-length** (step 0 = particle start, zero energy deposited). Step i represents the segment from `position[i-1]` to `position[i]` with energy `deposited_energy[i]`.

Midpoints of real steps: `0.5 * (pos[:-1] + pos[1:])`, energies: `e[1:]`.

### Data flow: dataset → model

```
dataset yields  (inputs, truth)
  inputs: {"pmt_ids": (L,), "hit_times": (L,)}   — padded to max_context_len
  truth:  {"position": (V,3), "time": (V,), "energy": (V,), "exists": (V,), ...}
                                                  — padded to max_n_vertices

MultiHitVarlenCollate converts a batch into:
  inputs: {"pmt_ids": (total_hits,), "hit_times": (total_hits,),
           "cu_seqlens": int32 (B+1,), "max_seqlen": int}
  truth:  stacked normally with batch dim

model(**inputs) →
  {"exists_logit": (B,V), "position": (B,V,3), "time": (B,V), "energy": (B,V), "log_sigma2": {...}}
```

### Metrics (`deepsno/metrics/`)

- **`metrics.py`** — Abstract `Metric` base class and `BatchedMetric` for running averages.
- **`metric_monitor.py`** — `MonitorCollection`, `TaskWeightMonitor`, `MultiLossMonitor` — wired into the training loop to log to wandb.
- Per-model metric files mirror the models directory structure.

### Utilities (`deepsno/utils/`)

- **`scheduler.py`** — `LinearWarmupCosineAnnealingLR`: custom LR scheduler (linear warmup + cosine decay).
- **`jinja.py`** — Jinja2-templated YAML loader. Supports `{{ load_yaml('...') }}` includes and custom filters like `model_save_directory()` and `path_join()`.
- **`train.py`** — `get_best_ckpt()` finds lowest-validation-loss checkpoint; unit conversion helpers.

## Configuration System

Configs are YAML files with optional Jinja2 templating. All classes are referenced by dotted `class_path` with `init_args`:

```yaml
model:
  class_path: deepsno.models.position_reco.PositionReco
  init_args:
    n_pmts: 9728
    d_model: 64

train:
  optimizer:
    class_path: torch.optim.Adam
    init_args:
      lr: 1e-5
  scheduler:
    class_path: deepsno.utils.scheduler.LinearWarmupCosineAnnealingLR
    init_args:
      warmup_epochs: 1000
      max_epochs: 100000
```

See `example_configs/` for working examples.

## Cluster Jobs

`condor/` contains HTC Condor job submission scripts for running training/prediction on compute clusters. Use `--force` flag to skip git hash validation when iterating quickly (but prefer clean commits for reproducibility).

## Code Style

- Line length: 120 (configured in `pyproject.toml` for ruff)
- Run `ruff check` and `ruff format` before committing
