#!/bin/bash
#SBATCH --job-name=requeue_cpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-mchen
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x.out

echo "Working directory: $(pwd)"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <config.yaml> [hydra overrides]" >&2
    exit 1
fi

config="$1"
shift
config_dir=$(dirname "$config")
config_name=$(basename "$config" .yaml)
extra_args=("$@")

echo
echo "Loading environment..."
source /home/degraw/deepsno/slurm/env.sh
echo "Environment loaded. Starting training..."

mkdir -p "slurm/logs"

export PYTHONFAULTHANDLER=1

echo
echo "Launching CPU training (single process, no torchrun)..."
train_log="slurm/logs/train_${SLURM_JOB_ID:-$$}.log"
srun --overlap python deepsno/train.py --config-dir="$config_dir" --config-name="$config_name" \
    force=true train.resume=true train.device=cpu "${extra_args[@]}" 2>&1 | tee "$train_log"
train_status="${PIPESTATUS[0]}"

checkpoint_dir=$(grep -oP 'Saving model config and checkpoints to \K.*' "$train_log" | tail -n1)

if [[ "$train_status" -eq 75 ]]; then
    echo "Time limit approaching; resubmitting via sbatch"
    resubmit_args=()
    for arg in "${extra_args[@]}"; do
        if [[ "$arg" != train.checkpoint_dir=* ]]; then
            resubmit_args+=("$arg")
        fi
    done
    if [[ -n "$checkpoint_dir" ]]; then
        resubmit_args+=("train.checkpoint_dir=$checkpoint_dir")
    fi
    sbatch "$0" "$config" "${resubmit_args[@]}"
    rm -f "$train_log"
    exit 0
fi

if [[ "$train_status" -ne 0 ]]; then
    exit "$train_status"
fi

echo "Training completed without preemption."
rm -f "$train_log"
