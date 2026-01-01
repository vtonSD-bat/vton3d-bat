import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np

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




def qwen_eval_masked(img1_path, img2_path, flag, estimator):
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
    class_name = "Upper Clothing" if flag == "upper" else "Lower Clothing"
    class_idx = classes.index(class_name)

    mask_exclude = (seg_map == class_idx)
    mask_include = ~mask_exclude

    img1 = img1_bgr.astype(np.float32) / 255.0
    img2 = img2_bgr.astype(np.float32) / 255.0

    diff = (img1 - img2) ** 2
    diff_masked = diff[mask_include]

    mse_value = diff_masked.mean().item()

    abs_diff = np.abs(img1 - img2)
    heatmap_gray = abs_diff.mean(axis=2)
    heatmap_gray[mask_exclude] = 0.0

    return mse_value, heatmap_gray
