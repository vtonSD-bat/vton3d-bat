#!/bin/bash
#SBATCH -p p4500
#SBATCH --job-name=vton_pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load anaconda

source /opt/conda/etc/profile.d/conda.sh

conda activate vton

mkdir -p logs

CONFIG_PATH=${1:-configs/vton_pipeline.yaml}

export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"

python -m vton3d.pipeline.run_pipeline --config "$CONFIG_PATH"

deactivate

module load anaconda

source /opt/conda/etc/profile.d/conda.sh

conda activate gsplat310vton

python -m vton3d.pipeline.run_gsplat --config "$CONFIG_PATH"

echo "=== Job finished successfully ==="
