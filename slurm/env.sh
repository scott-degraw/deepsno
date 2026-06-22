#!/usr/bin/bash
# Must be sourced from the project root (where pyproject.toml lives).

module load python/3.13

# $SLURM_TMPDIR is per-job local NVMe on Compute Canada; fall back to /tmp
# for interactive sessions. Either way, avoid reading torch off the network FS.
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
VENV_DIR="$LOCAL_SCRATCH/venv_deepsno"

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --frozen

source "$UV_PROJECT_ENVIRONMENT/bin/activate"
