#!/bin/bash
#SBATCH --job-name=multihit_2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=10
#SBATCH --account=def-mchen
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.out
#SBATCH --exclude=fc10512

echo "Working directory: $(pwd)"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <config.yaml> [--predict] [hydra overrides, e.g. train.device=cuda]" >&2
    exit 1
fi

config="$1"
shift

config_dir=$(dirname "$config")
config_name=$(basename "$config" .yaml)

# pull --predict out of the extra args so it isn't forwarded to train.py
run_predict=false
extra_args=()
for arg in "$@"; do
    if [[ "$arg" == "--predict" ]]; then
        run_predict=true
    else
        extra_args+=("$arg")
    fi
done

nvidia-smi

echo
echo "Loading environment..."
source slurm/env.sh
echo "Environment loaded. Starting training..."

mkdir -p "slurm/logs"

# --- derived variables ---

# strip GPU type prefix (e.g. h100:2 -> 2), fall back to cuda device count
if [[ "$SLURM_GPUS_PER_NODE" == *:* ]]; then
    GPUS_PER_NODE="${SLURM_GPUS_PER_NODE##*:}"
elif [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
    GPUS_PER_NODE="$SLURM_GPUS_PER_NODE"
else
    GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
fi

NNODES="${SLURM_NNODES:-1}"

# first node in the allocation is the rendezvous host
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -i 20000-65000 -n1)

# --- environment ---

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=none
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# on a shared node, be explicit about which GPUs we have
# CUDA_VISIBLE_DEVICES is already set correctly by Slurm when using --gpus-per-node
# but print it for debugging
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"

mkdir -p logs

# --- launch ---

# srun with --overlap allows this to work whether the job was allocated
# via salloc or sbatch, and on shared nodes
echo
echo "Launching training with torchrun..."
train_log="slurm/logs/train_${SLURM_JOB_ID:-$$}.log"
srun --overlap torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="$SLURM_JOB_ID" \
	deepsno/train.py --config-dir="$config_dir" --config-name="$config_name" force=true train.resume=true train.requeue_buffer_seconds=120 "${extra_args[@]}" 2>&1 | tee "$train_log"
train_status="${PIPESTATUS[0]}"

# the train run prints this path right before training starts
checkpoint_dir=$(grep -oP 'Saving model config and checkpoints to \K.*' "$train_log" | tail -n1)

if [[ "$train_status" -eq 75 ]]; then
    echo "Time limit approaching; resubmitting via sbatch"
    # scontrol requeue keeps the same JobID but imposes a ~30 min
    # EligibleTime cooldown on this cluster before the requeued job can
    # start again. A fresh sbatch of this same script avoids that cooldown
    # entirely (new JobID each cycle), so checkpoint_dir is passed straight
    # through as an explicit override rather than pinned via a
    # JobID-keyed marker file.
    resubmit_args=()
    for arg in "${extra_args[@]}"; do
        if [[ "$arg" != train.checkpoint_dir=* ]]; then
            resubmit_args+=("$arg")
        fi
    done
    if [[ -n "$checkpoint_dir" ]]; then
        resubmit_args+=("train.checkpoint_dir=$checkpoint_dir")
    fi
    if [[ "$run_predict" == true ]]; then
        resubmit_args+=(--predict)
    fi
    sbatch "$0" "$config" "${resubmit_args[@]}"
    rm -f "$train_log"
    exit 0
fi

if [[ "$train_status" -ne 0 ]]; then
    exit "$train_status"
fi

if [[ "$run_predict" == true ]]; then
    if [[ -z "$checkpoint_dir" ]]; then
        echo "Could not determine checkpoint directory from training output; skipping prediction." >&2
        exit 1
    fi

    echo
    echo "Running prediction using checkpoint: $checkpoint_dir"
    srun --overlap deepsno/predict.py --config-dir="$config_dir" --config-name="$config_name" force=true predict.ckpt="$checkpoint_dir"
fi

rm -f "$train_log"
