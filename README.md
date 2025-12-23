# vton3d-bat
Main Repo for Virtual Try-On Pipeline

```
Installation (Only Linux):
1. Go in create_envs.sh and set your desired Cuda Pytorch version 
(from https://pytorch.org/get-started/locally/) in this line: 

echo ">>> Installing PyTorch CUDA for vton"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

2. Create Conda Envs by running sbatch create_envs.sh 
This will create two Conda Envs. 
One for the VTON and VGGT model (vton) and one for the Gaussian Splatting Model (gsplat310vton) 
as they both need same libraries in different versions.

3. Create a folder `data/videos` and put your input video/s there.
4. run python scripts/extract_frames.py -n `numberofframes` (for further params see extract_frames.py)
5. go into configs/vton_pipeline.yaml and set your desired parameters 
(at least the input folder where the images of the person are under: path: scene_dir, 
and the clothing image under: qwen: clothing_image)
6. run sbatch run_pipeline.sh to run the pipeline on your images
```
