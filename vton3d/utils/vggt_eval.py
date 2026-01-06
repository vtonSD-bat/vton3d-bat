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
    return dict(coverage=coverage, l1=l1, rmse=rmse, psnr=psnr)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data/<scene>/real")
    ap.add_argument("--out", default=None, help="Output dir (default: <root>/vggt_validation)")
    ap.add_argument("--auto_resize", action="store_true", help="Resize images to camera width/height if mismatch")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all, else limit")
    ap.add_argument("--every", type=int, default=1, help="Process every n-th image")
    args = ap.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    sparse_root = root / "sparse"
    out_dir = Path(args.out) if args.out else (root / "vggt_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = find_model_dir(sparse_root)
    print(f"[INFO] Using COLMAP model dir: {model_dir}")

    rec = pycolmap.Reconstruction(str(model_dir))

    # Gather 3D points
    pids = list(rec.points3D.keys())
    if len(pids) == 0:
        raise SystemExit("[ERROR] No points3D found in reconstruction.")

    Xw = np.stack([rec.points3D[pid].xyz for pid in pids], axis=0)
    rgb = np.stack([rec.points3D[pid].color for pid in pids], axis=0).astype(np.uint8)
    print(f"[INFO] Loaded {Xw.shape[0]} 3D points")

    # Iterate images
    image_items = list(rec.images.items())
    if args.max_images and args.max_images > 0:
        image_items = image_items[: args.max_images]
    image_items = image_items[:: max(1, args.every)]
    print(f"[INFO] Processing {len(image_items)} images")

    rows = []
    for image_id, im in image_items:
        name = im.name
        img_path = images_dir / name
        if not img_path.exists():
            print(f"[WARN] missing file: {img_path}")
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] unreadable: {img_path}")
            continue
        orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cam = rec.cameras[im.camera_id]
        W, H = int(cam.width), int(cam.height)

        if orig.shape[1] != W or orig.shape[0] != H:
            msg = f"[SIZE] {name}: image={orig.shape[1]}x{orig.shape[0]} vs cam={W}x{H}"
            if args.auto_resize:
                print(msg + " -> resizing for comparison (interpret metrics carefully)")
                orig = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
            else:
                print(msg + " -> skipping (use --auto_resize or fix preprocessing mismatch)")
                continue

        T = get_cam_from_world_matrix(im)
        if T is None:
            print(f"[WARN] {name}: no pose/cam_from_world -> skip")
            continue
        if T.shape != (3, 4):
            print(f"[WARN] {name}: unexpected cam_from_world matrix shape {T.shape} -> skip")
            continue

        R = T[:, :3]
        t = T[:, 3]

        # project and render
        u, v, z, cam_model = project_world_to_image_Rt(Xw, R, t, cam)
        behind_ratio = float(np.mean(np.isfinite(z) & (z <= 0)))

        rendered, mask = render_zbuffer(H, W, u, v, z, rgb)
        m = masked_metrics(orig, rendered, mask)

        overlay = orig.copy()
        overlay[mask] = (0.7 * orig[mask] + 0.3 * rendered[mask]).astype(np.uint8)

        # Heatmap diff (no overlay)
        diff_heat_bgr = make_diff_heatmap(orig, rendered, mask)

        stem = Path(name).stem
        cv2.imwrite(str(out_dir / f"{stem}_render.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{stem}_diff_heat.png"), diff_heat_bgr)

        rows.append(
            {
                "image": name,
                "camera_model": cam_model,
                "coverage": m["coverage"],
                "l1": m["l1"],
                "rmse": m["rmse"],
                "psnr": m["psnr"],
                "behind_ratio": behind_ratio,
                "W": W,
                "H": H,
                "num_points": int(Xw.shape[0]),
            }
        )

        print(
            f"[{name}] model={cam_model} cov={m['coverage']:.4f} "
            f"L1={m['l1']:.4f} RMSE={m['rmse']:.4f} PSNR={m['psnr']:.2f} behind={behind_ratio:.3f}"
        )

    # Save CSV
    if rows:
        import csv

        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved metrics: {csv_path}")
    else:
        print("[WARN] No rows written (all images skipped?)")

    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
