# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

DeepSNO is a deep learning framework for the SNO+ particle physics detector. It provides neural network models, dataloaders, and training infrastructure to reconstruct particle interaction properties (position, time) from photomultiplier tube (PMT) detector data stored in ROOT files.

## Dependencies

uv is used for Python package management and all dependencies. It is preferred to always run with `uv run`.

On cluster nodes, the virtualenv is placed on `$TMPDIR` (local scratch) rather than the project directory, as done in `slurm/env.sh`:

```bash
export UV_PROJECT_ENVIRONMENT="$TMPDIR/venv_deepsno"
uv sync --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

This avoids hammering the network/scratch filesystem with the many small files a Python venv creates — those filesystems are tuned for large sequential I/O, not small-file access.

## Commands

```bash
# Install (requires Python 3.13)
uv sync

# Lint / format
ruff check deepsno/
ruff format deepsno/

# Train
uv run python deepsno/train.py --config-dir=configs --config-name=bi214 \
  train.entity=<wandb-entity> train.project=<wandb-project> \
  train.checkpoint_dir=/path/to/checkpoints train.device=cuda

# Predict
uv run python deepsno/predict.py --config-dir=configs --config-name=bi214 \
  predict.ckpt=/path/to/checkpoints predict.device=cuda

# Bench (iterate the train dataloader and report throughput)
uv run python deepsno/bench.py --config-dir=configs --config-name=bi214
```

`--config-dir=<dir>` adds `<dir>` to Hydra's config search path (native Hydra
flag — real experiment configs live in `configs/` or `example_configs/`, not
inside the package); `--config-name=<stem>` selects `<dir>/<stem>.yaml` as the
primary config. Each entry point's own `deepsno/conf/*_defaults.yaml` is the
*other* search root (set as `config_path` on its `@hydra.main`), so every
experiment config pulls in the relevant baseline(s) via its own `defaults:`
list, e.g.:

```yaml
defaults:
  - /train_defaults
  - /predict_defaults
  - /bench_defaults
  - _self_
```

— this is what makes omitted optional keys resolve instead of KeyError'ing
(and, since `--config-dir` only accepts one directory, this list also lets
the same config serve all three entry points regardless of which one loads
it). Everything else is a Hydra override (dotted-path `key=value`, e.g.
`model.d_model=128`); Hydra's own run-dir/output-subdir management is
disabled since each script manages its own checkpoint/output paths. Path
joining and timestamped checkpoint directories don't need a custom resolver —
plain string interpolation (`${vars.base_dir}/checkpoints`) and Hydra's
built-in `${now:%Y-%m-%d}`/`${now:%H-%M-%S}` cover both (resolved once and
cached per node, so multiple references to the same value stay consistent
within a run). The one custom resolver that remains, `${load_yaml:path}`
(`deepsno/utils/resolvers.py`), loads an external YAML file's contents inline
— there's no Hydra-native equivalent for that. YAML
anchors/aliases/merge-keys (`&x`, `*x`, `<<:`) still work as plain YAML, but
prefer `${...}` interpolation (e.g. a `vars:` block, or referencing another
key like `${train.seed}` directly) for any value you might want to override
on the CLI — a YAML alias is copied at parse time, before Hydra ever sees it,
so overriding the anchor's own key does not propagate to its aliases.

There is no automated test suite — validation is done via wandb during training.

## Architecture

### Entry & Training Flow

1. **`deepsno/train.py`** / **`deepsno/predict.py`** / **`deepsno/bench.py`** — Hydra (`@hydra.main`) CLI entry points, one per workflow, with `config_path` set to `deepsno/conf/` (that script's `*_defaults.yaml` base layer). Real experiment configs are selected via native `--config-dir=<dir> --config-name=<stem>` flags, accept Hydra dotted-path overrides, validate the git commit hash for reproducibility (`deepsno/utils/cli.py`), and delegate to `loops.train()` / `loops.predict()` / `loops.bench_dataloader()`.

2. **`deepsno/loops.py`** — Core training/validation/inference loops. `train()` handles gradient accumulation, AMP (`autocast_dtype`), gradient clipping, LR scheduling, checkpoint saving, and periodic validation. Checkpoints are saved when validation loss improves.

3. Models, datasets, optimizers, schedulers, loss functions, and monitors are all instantiated at runtime via `hydra.utils.instantiate`, driven by `_target_` + flat kwargs in YAML.

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
- **`resolvers.py`** — Registers the `load_yaml` OmegaConf custom resolver.
- **`hydra_cli.py`** — Appends the standing Hydra overrides (disabling run-dir/output-subdir management) to `sys.argv` before `@hydra.main` parses it.
- **`cli.py`** — Shared `train.py`/`predict.py`/`bench.py` helpers: git-hash validation/snapshotting, log-file tee.
- **`train.py`** — `get_best_ckpt()` finds lowest-validation-loss checkpoint; unit conversion helpers.

## Configuration System

Configs are plain YAML files, composed through Hydra (anchors/aliases/merge-keys and `${...}` interpolation/resolvers all work). All classes are referenced via Hydra's native `_target_` convention, with constructor kwargs as flat siblings of `_target_` (no `init_args` indirection):

```yaml
model:
  _target_: deepsno.models.position_reco.PositionReco
  n_pmts: 9728
  d_model: 64

train:
  optimizer:
    _target_: torch.optim.Adam
    lr: 1e-5
  scheduler:
    _target_: deepsno.utils.scheduler.LinearWarmupCosineAnnealingLR
    warmup_epochs: 1000
    max_epochs: 100000
```

`hydra.utils.instantiate(cfg, *extra_args, _convert_="all")` is called at each entry point's call sites (`train.py`, `predict.py`, `bench.py`) — `_convert_="all"` ensures nested non-`_target_` values come back as plain `dict`/`list` rather than `DictConfig`/`ListConfig`. Extra runtime-only positional args (e.g. `param_groups` for an optimizer, `optimizer` for a scheduler, `run` for a metric monitor) are passed as additional positional args to `instantiate()`, not baked into the YAML. To reference a class/callable itself rather than calling it (e.g. an `activation` argument expecting a class object), use `_target_: <path>` with `_partial_: true` and no other keys — calling the result with no args calls the class/callable directly.

See `example_configs/` for working examples.

## Cluster Jobs

`condor/` contains HTC Condor job submission scripts for running training/prediction on compute clusters. Pass `force=true` to skip git hash validation when iterating quickly (but prefer clean commits for reproducibility).

## Code Style

- Line length: 120 (configured in `pyproject.toml` for ruff)
- Run `ruff check` and `ruff format` before committing
