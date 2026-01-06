import argparse
from pathlib import Path
import numpy as np
import cv2
import pycolmap

from vton3d.utils.vggt_eval import (_normalize_camera_model, get_pinhole_params,
                                     project_world_to_image_Rt, render_zbuffer,masked_metrics, find_model_dir,
                                     get_cam_from_world_matrix, make_diff_heatmap)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data/<scene>/real")
    ap.add_argument("--out", default=None, help="Output dir (default: <root>/vggt_eval)")
    ap.add_argument("--auto_resize", action="store_true", help="Resize images to camera width/height if mismatch")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all, else limit")
    ap.add_argument("--every", type=int, default=1, help="Process every n-th image")
    args = ap.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    sparse_root = root / "sparse"
    out_dir = Path(args.out) if args.out else (root / "vggt_eval")
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