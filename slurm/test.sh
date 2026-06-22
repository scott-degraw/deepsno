#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --account=def-mchen
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:20:00
#SBATCH --output=slurm/logs/%x.out

set -eou pipefail

config="${1:-configs/imaging.yaml.j3}"
ckpt="$2"

config_dir=$(dirname "$config")
config_name=$(basename "$config" .yaml)

source slurm/env.sh
srun --cpu-bind=none deepsno/predict.py --config-dir="$config_dir" --config-name="$config_name" force=true predict.ckpt="$ckpt"
