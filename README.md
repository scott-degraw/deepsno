# DeepSNO

Deep learning models, dataloaders, and training infrastructure for reconstructing
particle interaction properties (position, time) from PMT detector data in the
SNO+ experiment.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

On cluster nodes, source `slurm/env.sh` instead — it places the venv on
`$TMPDIR`/local scratch rather than the project directory, then runs `uv sync`:

```bash
source slurm/env.sh
```

## Running train / predict / bench

Each entry point is a Hydra CLI app. Config files live outside the package, in
`configs/` (your own experiments) or `example_configs/` (templates) — select
one with `--config-dir=<dir> --config-name=<stem>`, then add any Hydra
dotted-path overrides:

```bash
uv run python deepsno/train.py --config-dir=configs --config-name=bi214 \
  train.entity=<wandb-entity> train.project=<wandb-project> \
  train.checkpoint_dir=/path/to/checkpoints train.device=cuda

uv run python deepsno/predict.py --config-dir=configs --config-name=bi214 \
  predict.ckpt=/path/to/checkpoints predict.device=cuda

uv run python deepsno/bench.py --config-dir=configs --config-name=bi214
```

Pass `force=true` to skip the git-clean-tree check when iterating quickly.

## Writing a config

Classes are instantiated via Hydra's `_target_` convention — constructor kwargs
are flat siblings of `_target_` (no nesting):

```yaml
model:
  _target_: deepsno.models.position_reco.PositionReco
  n_pmts: 9728
  d_model: 64
```

Every config needs a `defaults:` list pulling in that entry point's baseline
(so omitted optional keys resolve instead of erroring), e.g.:

```yaml
defaults:
  - /train_defaults
  - /predict_defaults
  - /bench_defaults
  - _self_
```

For DRY composition, use:
- a `vars:` block + `${vars.x}` interpolation for any value you might want to
  override on the CLI (e.g. `${vars.batch_size}`), including computed ones —
  string interpolation joins paths directly (`${vars.base_dir}/checkpoints`),
  and Hydra's `${now:%Y-%m-%d}`/`${now:%H-%M-%S}` build timestamped dirs.
- plain YAML anchors/merge-keys (`&x`, `*x`, `<<:`) for large structural reuse
  (e.g. deriving `val_dataloader` from `train_dataloader`). Note these are
  copied at parse time, before Hydra sees them, so overriding the anchor's own
  key on the CLI does **not** propagate to its aliases — don't anchor anything
  you'd want to override later.

See `example_configs/` for working examples.
