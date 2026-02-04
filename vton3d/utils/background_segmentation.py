from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

from transformers import Sam3Processor, Sam3Model


@dataclass
class BackgroundSegmentationConfig:
    model_id: str = "facebook/sam3"
    text_prompt: str = "human"
    threshold: float = 0.5
    mask_threshold: float = 0.5

    # Optional filtering
    min_score: float = 0.0
    top_k: int = 0  # 0 => keep all

    # If no mask found: keep original (recommended) or make white image
    keep_original_if_no_mask: bool = True

    # Output behavior
    overwrite: bool = True
    out_dir: Optional[Path] = None  # None => overwrite in-place


def list_images(images_dir: Path) -> list[Path]:
    images_dir = images_dir.expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def load_sam3(model_id: str, device: Optional[str] = None) -> tuple[Sam3Model, Sam3Processor, str]:
    """
    Loads SAM3 model + processor and returns (model, processor, device_str).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam3Model.from_pretrained(model_id).to(device)
    processor = Sam3Processor.from_pretrained(model_id)
    model.eval()
    return model, processor, device


@torch.no_grad()
def segment_union_human_mask(
    image: Image.Image,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    cfg: BackgroundSegmentationConfig,
) -> Optional[np.ndarray]:
    """
    Returns boolean union mask (H,W) for all detected instances matching cfg.text_prompt.
    Returns None if no mask found after filtering.
    """
    image = image.convert("RGB")
    inputs = processor(images=image, text=cfg.text_prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = [(image.height, image.width)]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=cfg.threshold,
        mask_threshold=cfg.mask_threshold,
        target_sizes=target_sizes,
    )[0]

    masks = results.get("masks", None)
    scores = results.get("scores", None)

    if masks is None or len(masks) == 0:
        return None

    masks = masks.detach().to("cpu")
    if scores is not None:
        scores = scores.detach().to("cpu")

        keep = torch.ones(len(masks), dtype=torch.bool)
        if cfg.min_score > 0:
            keep &= (scores >= cfg.min_score)

        idxs = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            return None

        if cfg.top_k and cfg.top_k > 0 and idxs.numel() > cfg.top_k:
            kept_scores = scores[idxs]
            topk = torch.topk(kept_scores, k=cfg.top_k).indices
            idxs = idxs[topk]

        masks = masks[idxs]

    if masks.dtype != torch.bool:
        masks = masks > 0.5

    union = torch.any(masks, dim=0).cpu().numpy().astype(bool)
    return union


def apply_white_background(image: Image.Image, human_mask: np.ndarray) -> Image.Image:
    """
    Keeps pixels where human_mask==True, sets background to white.
    """
    img = np.array(image.convert("RGB"), dtype=np.uint8)
    out = img.copy()
    out[~human_mask] = 255
    return Image.fromarray(out, mode="RGB")


def process_image_path_inplace_or_to_dir(
    img_path: Path,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    cfg: BackgroundSegmentationConfig,
) -> tuple[Path, bool]:
    """
    Processes one image. Returns (saved_path, had_mask).
    Saves either in-place or into cfg.out_dir (same filename).
    """
    img_path = img_path.expanduser().resolve()
    image = Image.open(img_path).convert("RGB")

    mask = segment_union_human_mask(image, model, processor, device, cfg)

    if mask is None:
        if cfg.keep_original_if_no_mask:
            out_img = image
        else:
            white = np.full((image.height, image.width, 3), 255, dtype=np.uint8)
            out_img = Image.fromarray(white, mode="RGB")
        had_mask = False
    else:
        out_img = apply_white_background(image, mask)
        had_mask = True

    out_dir = cfg.out_dir.expanduser().resolve() if cfg.out_dir is not None else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name

    if (not cfg.overwrite) and out_path.exists():
        return out_path, had_mask

    out_img.save(out_path)
    return out_path, had_mask


def run_background_segmentation_on_images_dir(
    images_dir: Path,
    cfg: Optional[BackgroundSegmentationConfig] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Segments humans in all images inside images_dir and writes results back
    with same filenames (default: overwrite in-place).
    Returns stats dict.
    """
    cfg = cfg or BackgroundSegmentationConfig()
    images_dir = images_dir.expanduser().resolve()

    paths = list_images(images_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model, processor, device_str = load_sam3(cfg.model_id, device=device)

    processed = 0
    saved = 0
    no_mask = 0

    for p in paths:
        processed += 1
        _, had_mask = process_image_path_inplace_or_to_dir(p, model, processor, device_str, cfg)
        saved += 1
        if not had_mask:
            no_mask += 1

    return {"processed": processed, "saved": saved, "no_mask": no_mask, "images_dir": str(images_dir)}


def run_background_segmentation_for_scene(
    scene_dir: Path,
    cfg: Optional[BackgroundSegmentationConfig] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Convenience wrapper: runs on <scene_dir>/real/images
    """
    scene_dir = scene_dir.expanduser().resolve()
    return run_background_segmentation_on_images_dir(scene_dir / "real" / "images", cfg=cfg, device=device)
