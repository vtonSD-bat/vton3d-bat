import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference.segmentation import classes


def get_clothing_class_idx(flag: str) -> int:
    """
    Returns the Sapiens class index for "upper" or "lower" clothing.
    """
    flag = flag.lower()
    if flag == "upper":
        target = "Upper Clothing"
    elif flag == "lower":
        target = "Lower Clothing"
    else:
        raise ValueError(f"Invalid flag '{flag}'. Expected 'upper' or 'lower'.")

    if target not in classes:
        raise ValueError(f"Class '{target}' not found in Sapiens class list.")

    return classes.index(target)


def _idx(name: str):
    try:
        return classes.index(name)
    except ValueError:
        return None

def compute_psnr_from_mse(mse_value: float, max_val: float = 1.0) -> float:
    """
    Computes PSNR from a given MSE value and maximum pixel value.
    """
    if mse_value <= 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse_value)


def qwen_eval_masked(img1_path, img2_path, flag, length_flag, estimator):
    """
    Loads two images, segments the first image, excludes a clothing class
    (upper or lower), computes MSE only on unmasked pixels, and returns
    both the MSE value and a masked difference heatmap.
    """
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")
    if img2_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    if img1_bgr.shape != img2_bgr.shape:
        raise ValueError("Images must have the same shape.")

    seg_map = estimator(img1_bgr).astype(np.int32)

    flag = flag.lower()
    length_flag = length_flag.lower()

    if flag not in ("upper", "lower"):
        raise ValueError(f"Invalid flag '{flag}', expected 'upper' or 'lower'.")
    if length_flag not in ("short", "long"):
        raise ValueError(f"Invalid length_flag '{length_flag}', expected 'short' or 'long'.")

    clothing_name = "Upper Clothing" if flag == "upper" else "Lower Clothing"
    clothing_idx = _idx(clothing_name)
    if clothing_idx is None:
        raise ValueError(f"Class '{clothing_name}' not found in classes list.")

    mask_exclude = (seg_map == clothing_idx)

    if length_flag == "long":
        if flag == "upper":
            arm_names = ["Left Upper Arm", "Right Upper Arm", "Left Lower Arm", "Right Lower Arm"]
            arm_indices = [i for i in (_idx(n) for n in arm_names) if i is not None]

            has_arm = False
            for ai in arm_indices:
                if np.any(seg_map == ai):
                    has_arm = True
                    break

            if has_arm:
                for ai in arm_indices:
                    mask_exclude |= (seg_map == ai)

        else:  # lower
            leg_names = ["Left Upper Leg", "Right Upper Leg", "Left Lower Leg", "Right Lower Leg"]
            leg_indices = [i for i in (_idx(n) for n in leg_names) if i is not None]

            has_leg = False
            for li in leg_indices:
                if np.any(seg_map == li):
                    has_leg = True
                    break

            if has_leg:
                for li in leg_indices:
                    mask_exclude |= (seg_map == li)

    mask_include = ~mask_exclude

    img1 = img1_bgr.astype(np.float32) / 255.0
    img2 = img2_bgr.astype(np.float32) / 255.0

    diff = (img1 - img2) ** 2
    diff_masked = diff[mask_include]

    mse_value = diff_masked.mean().item()
    psnr_value = compute_psnr_from_mse(mse_value, max_val=1.0)

    abs_diff = np.abs(img1 - img2)
    heatmap_gray = abs_diff.mean(axis=2)
    heatmap_gray[mask_exclude] = 0.0

    norm = heatmap_gray / (heatmap_gray.max() + 1e-8)
    colormap = cm.get_cmap("Reds")
    heatmap_red = (colormap(norm)[..., :3] * 255).astype(np.uint8)
    heatmap_red = cv2.cvtColor(heatmap_red, cv2.COLOR_RGB2BGR)

    return mse_value, psnr_value, heatmap_red
