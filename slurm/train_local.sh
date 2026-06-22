#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <config> [extra args]" >&2
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501 # Use a free port
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

source "$SCRIPT_DIR/../slurm/env.sh"
srun --cpu-bind=none torchrun \
	--nnodes=$SLURM_NNODES \
	--nproc-per-node=gpu \
	--rdzv_id=$SLURM_JOB_ID \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
	deepsno -c "$1" --force train \
	"${@:2}"

