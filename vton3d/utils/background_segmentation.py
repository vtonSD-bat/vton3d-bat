from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"

import sys
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType


def background_segmentation(
    images_dir: str | Path,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
    white_value: int = 255,
):
    """
    Segments the person using Sapiens and sets background to white.
    Overwrites images in-place (same filenames).

    """
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    exts = {e.lower() for e in extensions}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = SapiensSegmentation(
        SapiensSegmentationType.SEGMENTATION_1B,
        device=device,
        dtype=torch.float16,
    )

    BACKGROUND_CLASS = 0

    image_paths = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]

    if not image_paths:
        print(f"[background_segmentation] No images found in {images_dir}")
        return

    processed = 0
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[background_segmentation][WARN] Could not read: {img_path}")
            continue

        H, W = img_bgr.shape[:2]

        seg_map = estimator(img_bgr).astype(np.int32)

        if seg_map.shape[0] != H or seg_map.shape[1] != W:
            print(
                f"[background_segmentation][WARN] seg_map shape mismatch for {img_path.name}: "
                f"seg={seg_map.shape} img={(H, W)} -> skipping"
            )
            continue

        person_mask = (seg_map != BACKGROUND_CLASS)

        #white background
        out_bgr = img_bgr.copy()
        out_bgr[~person_mask] = (white_value, white_value, white_value)

        assert out_bgr.shape[:2] == (H, W)

        #Overwrite in-place (keeps same filename)
        ok = cv2.imwrite(str(img_path), out_bgr)
        if not ok:
            print(f"[background_segmentation][WARN] Failed to write: {img_path}")
            continue

        processed += 1

    print(f"[background_segmentation] Processed {processed}/{len(image_paths)} images in {images_dir}")
