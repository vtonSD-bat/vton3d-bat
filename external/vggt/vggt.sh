#!/bin/bash
#SBATCH -p performance
#SBATCH -w server0103
#SBATCH --job-name=vggt_compute
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PYTHON_BIN=$(which python3)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

BASE_DIR="$SLURM_SUBMIT_DIR/frames_florian_50"
IMG_DIR="$BASE_DIR"
mkdir -p "$IMG_DIR"


# Caches auf lokalen Speicher (optional)
export HF_HOME="${SLURM_TMPDIR:-$PWD}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1


CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" torch torchvision
pip install pillow huggingface_hub einops  safetensors

python vton3d_viser_slurm.py
