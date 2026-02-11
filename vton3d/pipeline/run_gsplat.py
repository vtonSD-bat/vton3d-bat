#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from vton3d.utils.depth_maps import SapiensDepthMapBuilder


def load_config(path: Path) -> Dict[str, Any]:
    """
    Loads config from YAML or JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml") from e
        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        return cfg

    if suffix == ".json":
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("JSON root must be an object/dict.")
        return cfg

    raise ValueError(f"Unsupported config format: {suffix} (use .yaml/.yml or .json)")


def run_step_gsplat(cfg: dict) -> None:
    """
    Runs gsplat training. If pipeline.steps.depth_loss is enabled:
      - optionally builds depth maps via SapiensDepthMapBuilder
      - passes depth-loss flags to gsplat trainer
    """
    print("=== Step GSplat training ===")

    gs_cfg = cfg.get("gsplat", {}) or {}
    test_every = gs_cfg.get("test_every", 8)
    data_factor = gs_cfg.get("data_factor", 1)
    eval_steps = gs_cfg.get("eval_steps", [7000, 30000])
    tile_size = gs_cfg.get("tile_size", 16)
    max_steps = gs_cfg.get("max_steps", 30000)
    disable_video = gs_cfg.get("disable_video", False)
    disable_viewer = gs_cfg.get("disable_viewer", True)

    wandb_cfg = cfg.get("wandb", {}) or {}
    wandb_project = wandb_cfg.get("project", "vton_pipeline")

    project_root = Path(__file__).resolve().parents[2]
    gsplat_repo = project_root / "gsplat"

    base_scene_dir = Path(cfg["paths"]["scene_dir"])
    ef_cfg = cfg.get("extract_frames", {}) or {}
    num_frames = int(ef_cfg.get("num_frames", 0))
    scene_dir = base_scene_dir / f"{base_scene_dir.name}_{num_frames}"

    data_dir = (Path("..") / scene_dir / "qwen").resolve()
    result_dir = (Path("..") / scene_dir / "results" / "qwen_gsplat").resolve()

    steps = (cfg.get("pipeline", {}) or {}).get("steps", {}) or {}
    depth_enabled = bool(steps.get("depth_loss", False))

    depth_cfg = (gs_cfg.get("depth", {}) or {}) if depth_enabled else {}

    if depth_enabled:
        build_maps = bool(depth_cfg.get("build_maps", True))
        overwrite_maps = bool(depth_cfg.get("overwrite_maps", False))

        images_subdir = depth_cfg.get("images_subdir", None)
        if images_subdir in (None, "null"):
            images_subdir = f"images_{data_factor}" if int(data_factor) > 1 else "images"

        depth_maps_subdir = depth_cfg.get("depth_maps_subdir", "depth_maps")
        human_mask_subdir = depth_cfg.get("human_mask_subdir", "human_masks")
        human_mask_suffix = depth_cfg.get("human_mask_suffix", "human_masks")

        invert_depth = bool(depth_cfg.get("invert_depth", False))
        save_mode = str(depth_cfg.get("save_mode", "u16"))

        sapiens_cfg = depth_cfg.get("sapiens", {}) or {}
        sapiens_repo_dir = sapiens_cfg.get("repo_dir", None)
        if sapiens_repo_dir is None:
            raise ValueError("gsplat.depth.sapiens.repo_dir is required when depth_loss is enabled.")

        sapiens_device = sapiens_cfg.get("device", None)
        sapiens_dir = Path(sapiens_repo_dir).expanduser().resolve()

        if build_maps:
            print("=== [Depth] Building Sapiens depth maps ===")
            builder = SapiensDepthMapBuilder(
                sapiens_dir=sapiens_dir,
                device=sapiens_device,
            )
            builder.build_for_dataset(
                data_dir=data_dir,
                images_subdir=str(images_subdir),
                human_mask_subdir=str(human_mask_subdir),
                human_mask_suffix=str(human_mask_suffix),
                out_subdir=str(depth_maps_subdir),
                invert_depth=invert_depth,
                save_mode=save_mode,
                skip_existing=not overwrite_maps,
                fail_on_missing_mask=False,
            )
            print("=== [Depth] Done ===\n")

    cmd = [
        sys.executable,
        "examples/gsplat_trainer_pipe.py",
        "default",
        "--data_dir", str(data_dir),
        "--data_factor", str(data_factor),
        "--result_dir", str(result_dir),
        "--test_every", str(test_every),
        "--eval_steps", *[str(s) for s in eval_steps],
        "--tile_size", str(tile_size),
        "--max_steps", str(max_steps),
        "--wandb_project", str(wandb_project),
    ]

    if depth_enabled:
        cmd.append("--depth_loss")

        if "depth_lambda" in depth_cfg:
            cmd += ["--depth_lambda", str(depth_cfg["depth_lambda"])]
        if "depth_mode" in depth_cfg:
            cmd += ["--depth_mode", str(depth_cfg["depth_mode"])]
        if "depth_warmup" in depth_cfg:
            cmd += ["--depth_warmup", str(depth_cfg["depth_warmup"])]
        if "depth_ramp" in depth_cfg:
            cmd += ["--depth_ramp", str(depth_cfg["depth_ramp"])]
        if "depth_num_pairs" in depth_cfg:
            cmd += ["--depth_num_pairs", str(depth_cfg["depth_num_pairs"])]
        if "depth_margin" in depth_cfg:
            cmd += ["--depth_margin", str(depth_cfg["depth_margin"])]

    if disable_video:
        cmd.append("--disable_video")
    if disable_viewer:
        cmd.append("--disable_viewer")

    print(f"  -> gsplat repo: {gsplat_repo}")
    print(f"  -> running: {' '.join(cmd)}")

    subprocess.run(cmd, cwd=str(gsplat_repo), check=True)

    print("=== [Step GSplat] Done ===\n")



def main() -> None:
    """
    CLI entrypoint.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config file (.yaml/.yml or .json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)

    steps = (cfg.get("pipeline", {}) or {}).get("steps", {}) or {}
    run_gsplat = bool(steps.get("gsplat", False))

    if run_gsplat:
        run_step_gsplat(cfg)
    else:
        print("=== Step GSplat training skipped (pipeline.steps.gsplat = false) ===")


if __name__ == "__main__":
    main()
