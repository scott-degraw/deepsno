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
- **`multihit.py`** — `UprootMultiFileDataset` and multi-hit event handling with Numba JIT compilation.
- **`cuts.py`**, **`filter_pmts.py`**, **`pmt_info.py`** — Event selection, PMT quality filtering, and detector geometry.

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
