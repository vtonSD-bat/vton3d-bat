#!/bin/bash
#SBATCH -p performance
#SBATCH -w server0109
#SBATCH --job-name=vton_pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate

mkdir -p logs

CONFIG_PATH=${1:-configs/vton_pipeline.yaml}

python -m vton3d.pipeline.run_pipeline --config "$CONFIG_PATH"
