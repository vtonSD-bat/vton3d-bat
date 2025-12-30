import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np
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


def qwen_eval_masked(
    img1_path: str,
    img2_path: str,
    flag: str,
    estimator
) -> float:
    """
    Loads both images from paths, runs segmentation on the first image,
    excludes either upper or lower clothing pixels based on the flag,
    applies the same mask to both images, and computes MSE only on unmasked pixels.
    """
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")

    if img2_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    if img1_bgr.shape != img2_bgr.shape:
        raise ValueError(f"Images have different shapes: {img1_bgr.shape} vs {img2_bgr.shape}")

    seg_map = estimator(img1_bgr).astype(np.int32)
    class_idx = get_clothing_class_idx(flag)

    clothing_mask = (seg_map == class_idx)
    unmasked_mask = ~clothing_mask

    if not np.any(unmasked_mask):
        raise ValueError("No unmasked pixels available for MSE computation.")

    img1 = img1_bgr.astype(np.float32) / 255.0
    img2 = img2_bgr.astype(np.float32) / 255.0

    mask3 = unmasked_mask[..., None]

    diff2 = (img1 - img2) ** 2
    diff2_masked = diff2[mask3]

    return diff2_masked.mean().item()
