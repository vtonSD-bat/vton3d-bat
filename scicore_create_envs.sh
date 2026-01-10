#!/bin/bash

module purge || true
module load Miniconda3 

set -e

echo ">>> Initializing conda"
eval "$(conda shell.bash hook)"

export PIP_DEFAULT_TIMEOUT=300
export PIP_RETRIES=10




ENV_NAME_VTON=vton

# >>> einzige Änderung: Env nur erstellen, wenn sie nicht existiert
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME_VTON"; then
    echo ">>> Conda env '$ENV_NAME_VTON' already exists – skipping creation"
else
    echo ">>> Creating conda env: $ENV_NAME_VTON"
    conda create -n $ENV_NAME_VTON python=3.11 -y
fi
# <<< Ende Änderung

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


export INSIGHTFACE_HOME="$HOME/.insightface"
MODEL_DIR="$INSIGHTFACE_HOME/models/buffalo_l"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/det_10g.onnx" ]; then
  echo ">>> insightface buffalo_l already present – skipping download"
else
  echo ">>> Downloading insightface buffalo_l"
  mkdir -p "$INSIGHTFACE_HOME/models"
python - <<'PY'
import os, zipfile, pathlib, urllib.request
url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
root = pathlib.Path(os.path.expanduser("~/.insightface/models"))
zip_path = root / "buffalo_l.zip"
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(root)
PY

  # falls ONNX im falschen Ort gelandet ist, reparieren:
  mkdir -p "$MODEL_DIR"
  shopt -s nullglob
  for f in "$INSIGHTFACE_HOME/models/"*.onnx; do mv "$f" "$MODEL_DIR/"; done
  if [ -f "$INSIGHTFACE_HOME/models/buffalo_l.zip" ]; then mv "$INSIGHTFACE_HOME/models/buffalo_l.zip" "$MODEL_DIR/"; fi
fi


echo ">>> VTON environment done"
conda deactivate



# Gaussian Splatting Env

ENV_NAME_GSPLAT=gsplat310vton

# >>> einzige Änderung: Env nur erstellen, wenn sie nicht existiert
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME_GSPLAT"; then
    echo ">>> Conda env '$ENV_NAME_GSPLAT' already exists – skipping creation"
else
    echo ">>> Creating conda env: $ENV_NAME_GSPLAT"
    conda create -n $ENV_NAME_GSPLAT python=3.10 -y
fi
# <<< Ende Änderung

module load Miniconda3/24.7.1-0
eval "$(conda shell.bash hook)"


echo ">>> Activating $ENV_NAME_GSPLAT"
conda activate $ENV_NAME_GSPLAT

echo ">>> Python:"
which python
python -V
echo ">>> Pip:"
which pip

echo ">>> Python:"
which python
python -V
echo ">>> Pip:"
which pip

echo ">>> Upgrading pip"
python -m pip install -U pip

echo ">>> Installing numpy"
python -m pip install numpy wandb

echo ">>> Installing PyTorch CUDA 12.1 for gsplat"
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('Torch version:', torch.__version__)"

echo ">>> Installing base deps"
python -m pip install ninja jaxtyping rich

echo ">>> Installing gsplat (no build isolation)"
python -m pip install --no-build-isolation --no-cache-dir gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121

echo ">>> Installing gsplat example requirements"
python -m pip install -r gsplat/examples/scicore_requirements.txt

python - <<'PY'
from torchvision.models import alexnet, AlexNet_Weights
m = alexnet(weights=AlexNet_Weights.DEFAULT)  # lädt und cached
print("AlexNet weights cached OK")
PY


echo ">>> GSPLAT environment done"
conda deactivate
