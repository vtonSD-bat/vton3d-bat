"""
vton3d.pipeline.run_pipeline

Purpose:
--------
This script is the main entry point for the complete VTON pipeline.

Current functionality:
- loads a YAML configuration file
- automatically appends 'real' to the scene_dir from the config
- runs vggt_colmap.py (VGGT + COLMAP reconstruction) as the first pipeline step

"""

from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import torch
import os
import subprocess
import shutil
import wandb


from scripts.vggt_colmap import demo_fn
from vton3d.qwen.run_qwen import run_qwen_from_config_dict
from argparse import Namespace
from PIL import Image


#helper
def normalize_images_to_png(images_dir: Path, remove_jpg: bool = False):
    """
    Convert all .jpg/.jpeg images in images_dir to .png with same stem.
    Optionally remove the original jpg files.
    """
    images_dir = images_dir.resolve()
    converted = 0

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg"]:
            png_path = img_path.with_suffix(".png")
            if png_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            img.save(png_path)
            converted += 1

            if remove_jpg:
                img_path.unlink()

    print(f"  -> Normalized images to PNG in {images_dir} (converted {converted})")


def load_config(config_path: str | Path) -> dict:
    """
    Load the YAML configuration file and return it as a dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def copy_colmap_sparse(real_scene_dir: Path, qwen_scene_dir: Path):
    """
    Copy COLMAP sparse reconstruction from real -> qwen.
    Expects:
      real_scene_dir/sparse exists (usually sparse/0/*)
    Creates:
      qwen_scene_dir/sparse (same structure)
    """
    src = real_scene_dir / "sparse"
    dst = qwen_scene_dir / "sparse"

    if not src.exists():
        raise FileNotFoundError(f"Missing COLMAP sparse folder: {src}")

    # Copy whole sparse tree; overwrite if exists
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"  -> Copied COLMAP sparse: {src}  ->  {dst}")


def build_vggt_args_from_config(cfg: dict) -> Namespace:
    """
    Build an argparse.Namespace for vggt_colmap.demo_fn() using the YAML config.

    It reads 'scene_dir' from cfg['paths']['scene_dir'],
    appends 'real', and maps additional parameters from cfg['vggt'].
    """
    base_scene_dir = Path(cfg["paths"]["scene_dir"])
    real_scene_dir = base_scene_dir / "real"

    vggt_cfg = cfg.get("vggt", {})

    args = Namespace(
        scene_dir=str(real_scene_dir),
        seed=int(vggt_cfg.get("seed", 42)),
        use_ba=bool(vggt_cfg.get("use_ba", False)),
        max_reproj_error=float(vggt_cfg.get("max_reproj_error", 8.0)),
        shared_camera=bool(vggt_cfg.get("shared_camera", False)),
        camera_type=str(vggt_cfg.get("camera_type", "SIMPLE_PINHOLE")),
        vis_thresh=float(vggt_cfg.get("vis_thresh", 0.2)),
        query_frame_num=int(vggt_cfg.get("query_frame_num", 8)),
        max_query_pts=int(vggt_cfg.get("max_query_pts", 4096)),
        fine_tracking=bool(vggt_cfg.get("fine_tracking", True)),
        conf_thres_value=float(vggt_cfg.get("conf_thres_value", 5.0)),
        keep_top_percent=float(vggt_cfg.get("keep_top_percent", 0.2)),
    )

    return args


# pipeline steps

def run_step_vggt_colmap(cfg: dict):
    """
    First pipeline step:
    - prepare VGGT arguments
    - call vggt_colmap.demo_fn()
    """
    print("=== [Step 1] VGGT + COLMAP Reconstruction ===")

    vggt_args = build_vggt_args_from_config(cfg)

    print(f"  -> Base scene path: {cfg['paths']['scene_dir']}")
    print(f"  -> Using scene subdirectory: {vggt_args.scene_dir}")

    with torch.no_grad():
        demo_fn(vggt_args)

    print("=== [Step VGGT] Done ===\n")


def run_step_qwen_clothing(cfg: dict):
    """
    Second pipeline step:
    run the Qwen clothing edit batch and store outputs in <scene_dir>/qwen/images.
    Qwen uses input images from <scene_dir>/real/images.
    """
    print("=== [Step 2] Qwen VTON edit ===")

    base_scene_dir = Path(cfg["paths"]["scene_dir"])

    input_dir = base_scene_dir / "real" / "images"
    output_dir = base_scene_dir / "qwen" / "images"

    qwen_cfg = cfg.get("qwen", {}).copy()
    if not qwen_cfg:
        raise ValueError("Missing 'qwen' section in config for Qwen clothing step.")

    qwen_cfg["source_dir"] = str(input_dir)
    qwen_cfg["output_dir"] = str(output_dir)

    print(f"  -> Qwen input images: {input_dir}")
    print(f"  -> Qwen output directory: {output_dir}")

    run_qwen_from_config_dict(qwen_cfg)

    real_scene_dir = base_scene_dir / "real"
    qwen_scene_dir = base_scene_dir / "qwen"
    copy_colmap_sparse(real_scene_dir, qwen_scene_dir)


    print("=== [Step Qwen] Done ===\n")


def run_pipeline(config_path: str | Path):
    """
    Main pipeline function.

    - loads YAML config
    - runs the VGGT reconstruction step
    - space for additional steps in the future
    """
    print(f"[Pipeline] Loading config: {config_path}")
    cfg = load_config(config_path)

    wandb.login()

    os.makedirs("logs", exist_ok=True)

    wandb.init(
        project="vton_pipeline",
        name=cfg.get("wandb", {}).get("run_name", None),
        config=cfg
    )

    run_id_path = Path("logs") / "wandb_current_pipe_id.txt"
    with run_id_path.open("w") as f:
        f.write(wandb.run.id)

    print(f"[Pipeline] Saved WandB run ID to: {run_id_path}")

    base_scene_dir = Path(cfg["paths"]["scene_dir"]).expanduser().resolve()
    real_images_dir = base_scene_dir / "real" / "images"

    normalize_images_to_png(real_images_dir, remove_jpg=True)

    run_step_vggt_colmap(cfg)

    run_step_qwen_clothing(cfg)

    print("[Pipeline] All defined steps completed.")


#cli
def parse_cli_args():
    """
    CLI parser for this pipeline script.
    """
    parser = argparse.ArgumentParser(description="VTON3D Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_cli_args()
    run_pipeline(cli_args.config)
