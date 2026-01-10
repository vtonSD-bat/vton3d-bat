#!/bin/bash


#SBATCH --job-name=vton_pipeline     #Name of your job
#SBATCH --cpus-per-task=8    #Number of cores to reserve
#SBATCH --mem-per-cpu=10G     #Amount of RAM/core to reserve
#SBATCH --time=01:00:00      #Maximum allocated time
#SBATCH --qos=gpu6hours      #Selected queue to allocate your job
#SBATCH --output=logs/vton_pipeline.o%j   #Path and name to the file for the STDOUT, %j will be substituted >
#SBATCH --error=logs/vton_pipeline.e%j    #Path and name to the file for the STDERR
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:1

module purge || true
module load Miniconda3

eval "$(conda shell.bash hook)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

conda activate vton

mkdir -p logs

CONFIG_PATH=${1:-configs/vton_pipeline.yaml}

export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"

echo "=== SLURM / GPU DEBUG ==="
echo "JobID=$SLURM_JOB_ID Node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
nvidia-smi || true
which nvidia-smi || true

python -m vton3d.pipeline.run_pipeline --config "$CONFIG_PATH"

conda deactivate

module load Miniconda3

conda activate gsplat310vton

echo "=== SLURM / GPU DEBUG ==="
echo "JobID=$SLURM_JOB_ID Node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
nvidia-smi || true
which nvidia-smi || true


python -m vton3d.pipeline.run_gsplat --config "$CONFIG_PATH"

echo "=== Job finished successfully ==="
