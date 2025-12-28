"""Thin wrapper around the VGGT package's demo_colmap entry point.

This module exposes a small helper that takes a scene directory and calls the
`demo_colmap.py` script from the VGGT package to produce a COLMAP sparse
reconstruction.

Expected directory layout::

    scene_dir/
        images/
            *.jpg / *.png ...

Output layout (produced by VGGT)::

    scene_dir/
        sparse/
            cameras.bin
            images.bin
            points3D.bin
            points.ply

The heavy lifting is performed by the `vggt` package. Here we just provide a
simple call-site that fits into the vton3d utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# We assume the VGGT package is installed and exposes the original demo script
# as `vggt.scripts.demo_colmap`. If your local VGGT install uses a different
# module path, you can adjust the import below accordingly.
try:  # pragma: no cover - import path is environment dependent
    from vggt.scripts import demo_colmap
except ImportError as exc:  # Fallback: try a flat module name
    raise ImportError(
        "Could not import 'vggt.scripts.demo_colmap'. Make sure the VGGT "
        "repository/package is installed and on PYTHONPATH."
    ) from exc


@dataclass
class VggtReconstructionConfig:
    """Configuration for running VGGT's COLMAP demo.

    Parameters
    ----------
    scene_dir: str
        Directory containing an `images/` subfolder.
    use_ba: bool, optional
        Whether to enable BA mode in VGGT (matches `--use_ba`).
    seed: int, optional
        Random seed (matches `--seed`). If None, VGGT defaults are used.
    max_reproj_error, shared_camera, camera_type, vis_thresh,
    query_frame_num, max_query_pts, fine_tracking, conf_thres_value:
        Optional overrides for the corresponding flags in `demo_colmap.py`.
    """

    scene_dir: str
    use_ba: bool = False

    seed: Optional[int] = None
    max_reproj_error: Optional[float] = None
    shared_camera: Optional[bool] = None
    camera_type: Optional[str] = None
    vis_thresh: Optional[float] = None
    query_frame_num: Optional[int] = None
    max_query_pts: Optional[int] = None
    fine_tracking: Optional[bool] = None
    conf_thres_value: Optional[float] = None


def run_vggt_reconstruction(config: VggtReconstructionConfig) -> str:
    """Run VGGT's `demo_colmap` on a scene directory.

    This is a thin wrapper that builds an `argparse.Namespace` compatible with
    `demo_colmap.demo_fn` / `demo_colmap.main` and calls it. It returns the
    path to the `sparse/` directory containing the COLMAP reconstruction.
    """

    scene_dir = os.fspath(config.scene_dir)
    image_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Expected images under '{image_dir}'")

    # Build an args-like object expected by demo_colmap.demo_fn
    # We start from the default parser (if exposed) and then override fields.
    if hasattr(demo_colmap, "parse_args"):
        # Reuse VGGT's own ArgumentParser defaults
        parser = demo_colmap.parse_args  # type: ignore[attr-defined]
        # parse_args with no CLI arguments to get defaults
        # Some implementations allow parse_args(args=None) to use sys.argv,
        # so we explicitly pass an empty list.
        base_args = parser([]) if callable(parser) else demo_colmap.parse_args([])
    else:
        # Fallback: build a minimal dummy namespace with expected attributes
        from argparse import Namespace

        base_args = Namespace(
            scene_dir=scene_dir,
            seed=42,
            use_ba=False,
            max_reproj_error=8.0,
            shared_camera=False,
            camera_type="SIMPLE_PINHOLE",
            vis_thresh=0.2,
            query_frame_num=8,
            max_query_pts=4096,
            fine_tracking=True,
            conf_thres_value=5.0,
        )

    # Override with our config values
    base_args.scene_dir = scene_dir
    if config.seed is not None:
        base_args.seed = config.seed
    if config.use_ba is not None:
        base_args.use_ba = config.use_ba
    if config.max_reproj_error is not None:
        base_args.max_reproj_error = config.max_reproj_error
    if config.shared_camera is not None:
        base_args.shared_camera = config.shared_camera
    if config.camera_type is not None:
        base_args.camera_type = config.camera_type
    if config.vis_thresh is not None:
        base_args.vis_thresh = config.vis_thresh
    if config.query_frame_num is not None:
        base_args.query_frame_num = config.query_frame_num
    if config.max_query_pts is not None:
        base_args.max_query_pts = config.max_query_pts
    if config.fine_tracking is not None:
        base_args.fine_tracking = config.fine_tracking
    if config.conf_thres_value is not None:
        base_args.conf_thres_value = config.conf_thres_value

    # Call into VGGT's demo function
    if hasattr(demo_colmap, "demo_fn"):
        demo_colmap.demo_fn(base_args)
    elif hasattr(demo_colmap, "main"):
        # Some repos expose `main(args)` instead
        demo_colmap.main(base_args)
    else:
        raise RuntimeError(
            "VGGT demo_colmap module does not expose a 'demo_fn' or 'main' "
            "function. Please check the VGGT version you installed."
        )

    sparse_dir = os.path.join(scene_dir, "sparse")
    return sparse_dir


def main(scene_dir: str, use_ba: bool = False) -> str:
    """Convenience entry point.

    Parameters
    ----------
    scene_dir: str
        Path to a scene directory with an `images/` subfolder.
    use_ba: bool
        Whether to enable BA mode (equivalent to `--use_ba` in VGGT).
    """

    config = VggtReconstructionConfig(scene_dir=scene_dir, use_ba=use_ba)
    return run_vggt_reconstruction(config)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser.add_argument("scene_dir", type=str, help="Path to scene directory containing an 'images/' subfolder.")
    parser.add_argument("--use-ba", action="store_true", help="Enable VGGT+BA reconstruction mode.")
    args = parser.parse_args()

    main(args.scene_dir, use_ba=args.use_ba)

