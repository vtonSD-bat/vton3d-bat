import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import glob
import json
import argparse
from typing import List


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
#model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

image_folder = r"./frames_florian_50"
max_images = 16

# Load and preprocess example images (replace with your own image paths)
image_paths: List[str] = sorted(glob.glob(os.path.join(image_folder, "*")))
if max_images and max_images > 0:
    image_paths = image_paths[:max_images]

if len(image_paths) == 0:
    raise RuntimeError(f"Keine Bilder gefunden in: {image_folder}")

print(f"[Info] Lade & preprocess {len(image_paths)} Bilder...")
    
images = load_and_preprocess_images(image_paths).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        predictions_cpu = {
    k: v.detach().cpu() if torch.is_tensor(v) else v
    for k, v in predictions.items()
}

torch.save(predictions_cpu, "vggt_predictions_scene03.pt")
        
