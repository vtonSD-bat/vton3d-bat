#!/usr/bin/env bash
#SBATCH -p performance
#SBATCH --job-name=env_setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G

set -e

echo ">>> Initializing conda"
eval "$(conda shell.bash hook)"



# vton Env


ENV_NAME_VTON=vton

echo ">>> Creating conda env: $ENV_NAME_VTON"
conda create -n $ENV_NAME_VTON python=3.11 -y

echo ">>> Activating $ENV_NAME_VTON"
conda activate $ENV_NAME_VTON

echo ">>> Python:"
which python
python -V

echo ">>> Upgrading pip"
pip install -U pip

echo ">>> Installing PyTorch CUDA for vton"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing vton project with pip -e"
pip install -e .

echo ">>> VTON environment done"
conda deactivate



# Gaussian Splatting Env

ENV_NAME_GSPLAT=gsplat310vton

echo ">>> Creating conda env: $ENV_NAME_GSPLAT"
conda create -n $ENV_NAME_GSPLAT python=3.10 -y

echo ">>> Activating $ENV_NAME_GSPLAT"
conda activate $ENV_NAME_GSPLAT

echo ">>> Python:"
which python
python -V
echo ">>> Pip:"
which pip

echo ">>> Upgrading pip"
python -m pip install -U pip

echo ">>> Installing numpy"
python -m pip install numpy

echo ">>> Installing PyTorch CUDA 12.1 for gsplat"
python -m pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('Torch version:', torch.__version__)"

echo ">>> Installing base deps"
python -m pip install ninja jaxtyping rich

echo ">>> Installing gsplat (no build isolation)"
python -m pip install --no-build-isolation --no-cache-dir gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121

echo ">>> Installing gsplat example requirements"
python -m pip install -r gsplat/examples/requirements.txt

echo ">>> GSPLAT environment done"
conda deactivate
