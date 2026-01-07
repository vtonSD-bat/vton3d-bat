import argparse
from pathlib import Path
import numpy as np
import cv2
import pycolmap


def _normalize_camera_model(model):
    """
    Normalizes pycolmap camera model to an uppercase string like 'PINHOLE'.
    Handles:
      - 'PINHOLE'
      - CameraModelId.PINHOLE
      - 'CameraModelId.PINHOLE'
    """
    if model is None:
        return None

    # Most robust: Enum-like with .name
    if hasattr(model, "name"):
        return str(model.name).upper()

    s = str(model)
    # e.g. 'CameraModelId.PINHOLE' -> 'PINHOLE'
    if "." in s:
        s = s.split(".")[-1]
    return s.upper()


def get_pinhole_params(camera):
    """
    Returns fx, fy, cx, cy, model_name for common COLMAP camera models.

    Note:
      - ignores distortion parameters for the sparse "render" validation
      - But must still parse fx/fy/cx/cy correctly.
    """
    # Different pycolmap versions expose different fields
    model = getattr(camera, "model_name", None)
    if model is None:
        model = getattr(camera, "model", None)
    model = _normalize_camera_model(model)

    params = np.array(camera.params, dtype=np.float64)

    if model == "PINHOLE":
        # fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        # f, cx, cy
        f, cx, cy = params[:3]
        fx, fy = f, f
    elif model == "SIMPLE_RADIAL":
        # f, cx, cy, k
        f, cx, cy = params[:3]
        fx, fy = f, f
    elif model == "RADIAL":
        # f, cx, cy, k1, k2
        f, cx, cy = params[:3]
        fx, fy = f, f
    elif model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[:4]
    elif model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        fx, fy, cx, cy = params[:4]
    elif model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4
        fx, fy, cx, cy = params[:4]
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")

    return float(fx), float(fy), float(cx), float(cy), model


def project_world_to_image_Rt(Xw, R, t, camera):
    """
    World -> Cam via Xc = R*Xw + t, then pinhole projection.
    R: (3,3), t: (3,)
    """
    Xc = (R @ Xw.T).T + t[None, :]
    z = Xc[:, 2]

    fx, fy, cx, cy, cam_model = get_pinhole_params(camera)
    u = fx * (Xc[:, 0] / z) + cx
    v = fy * (Xc[:, 1] / z) + cy
    return u, v, z, cam_model


def render_zbuffer(H, W, u, v, z, rgb):
    """
    Vectorized z-buffer point splat: nearest point per pixel wins.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (ui >= 0)
        & (ui < W)
        & (vi >= 0)
        & (vi < H)
    )
    if not np.any(valid):
        return img, mask

    ui = ui[valid]
    vi = vi[valid]
    zz = z[valid].astype(np.float32)
    cc = rgb[valid].astype(np.uint8)

    lin = vi * W + ui
    order = np.argsort(zz)  # nearest first
    lin_s = lin[order]

    # keep the first occurrence per pixel (nearest depth)
    _, first_idx = np.unique(lin_s, return_index=True)
    sel = order[first_idx]

    lin_sel = lin[sel]
    img.reshape(-1, 3)[lin_sel] = cc[sel]
    mask.reshape(-1)[lin_sel] = True
    return img, mask


def masked_metrics(orig_rgb, rend_rgb, mask):
    """
    Metrics only on pixels where we rendered something.
    """
    if mask.sum() == 0:
        return dict(coverage=0.0, l1=np.nan, rmse=np.nan, psnr=np.nan)

    o = orig_rgb.astype(np.float32) / 255.0
    r = rend_rgb.astype(np.float32) / 255.0
    diff = o[mask] - r[mask]

    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    psnr = float(20.0 * np.log10(1.0 / (rmse + 1e-12)))
    coverage = float(mask.mean())
    return dict(coverage=coverage, l1=l1, rmse=rmse, psnr=psnr, mse=mse)


def find_model_dir(sparse_root: Path) -> Path:
    """
    Accept:
      sparse/cameras.bin ...
    or:
      sparse/0/cameras.bin ...
    else: first subdir containing cameras.bin
    """
    if (sparse_root / "cameras.bin").exists():
        return sparse_root
    if (sparse_root / "0" / "cameras.bin").exists():
        return sparse_root / "0"
    for p in sorted(sparse_root.rglob("cameras.bin")):
        return p.parent
    raise FileNotFoundError(f"No cameras.bin found under: {sparse_root}")


def get_cam_from_world_matrix(im):
    """
    Compatibility helper for different pycolmap versions:
    - im.cam_from_world can be a property (Rigid3d) or a method returning Rigid3d
    - Rigid3d.matrix() returns (3,4) [R|t] world->cam
    """
    if hasattr(im, "has_pose"):
        hp = im.has_pose() if callable(im.has_pose) else im.has_pose
        if not hp:
            return None

    if not hasattr(im, "cam_from_world"):
        return None

    pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    if pose is None:
        return None

    if hasattr(pose, "matrix"):
        T = pose.matrix() if callable(pose.matrix) else pose.matrix
    else:
        return None

    return np.array(T, dtype=np.float64)  # expected (3,4)


def make_diff_heatmap(orig_rgb, rend_rgb, mask):
    """
    Returns a BGR heatmap image visualizing per-pixel RGB error magnitude (no overlay).
    Background (mask==False) is black.

    Steps:
      - abs RGB diff
      - scalar error = mean over RGB
      - robust normalize (percentiles on masked pixels)
      - apply colormap
    """
    # abs channel-wise difference
    d = cv2.absdiff(orig_rgb, rend_rgb).astype(np.float32)  # (H,W,3)

    # scalar error (0..255)
    err = d.mean(axis=2)  # (H,W)
    err[~mask] = 0.0

    if mask.any():
        lo, hi = np.percentile(err[mask], [1, 99])
        hi = max(float(hi), float(lo) + 1e-6)
        err_n = np.clip((err - lo) / (hi - lo), 0.0, 1.0)
    else:
        err_n = np.zeros_like(err, dtype=np.float32)

    err_u8 = (err_n * 255.0).astype(np.uint8)

    heat = cv2.applyColorMap(err_u8, cv2.COLORMAP_BONE)
    heat[~mask] = (0, 0, 0)
    return heat






