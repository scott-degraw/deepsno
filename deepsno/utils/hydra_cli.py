"""Shared argv plumbing for the Hydra-based CLI entry points.

Each entry point's ``@hydra.main`` decorator points ``config_path`` at
``deepsno/conf/`` (the base-defaults layer for that script). Real experiment
configs live outside the package — in ``configs/`` (private, gitignored
exempt) or ``example_configs/`` (tracked templates) — and are selected with
Hydra's native ``--config-dir=<dir> --config-name=<stem>`` flags; no copying
or synthesized wrapper files are involved. Each experiment config pulls in
its base-defaults layer(s) itself via a ``defaults:`` list (e.g.
``- /train_defaults``), resolved against ``deepsno/conf/`` since that's the
primary config search root.

Importing this module also registers the OmegaConf resolvers
(``deepsno.utils.resolvers``) that configs rely on, e.g. ``${load_yaml:...}``.
"""

from deepsno.utils import resolvers  # noqa: F401  (registers OmegaConf resolvers)

# Hydra's own run-directory/logging management is irrelevant here — every
# entry point manages its own checkpoint/output paths — so it's disabled.
_HYDRA_OVERRIDES = [
    "hydra.job.chdir=False",
    "hydra.output_subdir=null",
    "hydra.run.dir=.",
]


def prepare_argv(argv: list[str]) -> list[str]:
    """Append the standing Hydra overrides that disable run-dir/output-subdir management."""
    return [*argv, *_HYDRA_OVERRIDES]
