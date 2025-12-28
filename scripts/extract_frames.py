#!/usr/bin/env python3
"""
Extract a fixed number of frames from all videos in ../data/videos and save them to:
  ../data/<videoname_numframes>/real/images

Requires: opencv-python

Usage (from project root, e.g.):
  python scripts/extract_frames.py -n 50
  python scripts/extract_frames.py -n 100 -E png
  python scripts/extract_frames.py -n 50 -s 100 -t 1000
  python scripts/extract_frames.py -n 50 --rotate 90
"""

import cv2
import argparse
from pathlib import Path


def extract_frames(
    video_path: Path,
    data_dir: Path,
    num_frames: int,
    prefix: str | None = None,
    start: int = 0,
    end: int | None = None,
    ext: str = "jpg",
    overwrite: bool = False,
    rotate: int = 90,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be one of [0, 90, 180, 270], got {rotate}")

    # Mapping für OpenCV-Rotationscodes
    rotate_code = None
    if rotate == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate == 180:
        rotate_code = cv2.ROTATE_180
    elif rotate == 270:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video seems to have 0 frames: {video_path}")

    # Standard-Ende: gesamte Videolänge
    if end is None or end > total_frames:
        end = total_frames

    if start < 0 or start >= end:
        cap.release()
        raise ValueError(
            f"Invalid range for {video_path.name}: "
            f"start={start}, end={end}, total={total_frames}"
        )

    # Ordnername: <videoname_numframes> (z.B. myvideo_50)
    scene_name = f"{video_path.stem}_{num_frames}"

    # Output-Pfad: ../data/<videoname_numframes>/real/images
    out_dir = data_dir / scene_name / "real" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default prefix = video filename (without extension)
    if not prefix:
        prefix = video_path.stem

    # Zero padding based on end-1
    digits = max(4, len(str(max(0, end - 1))))

    # Ziel-Frames: num_frames Stück gleichmäßig verteilt im Bereich [start, end)
    frame_range = end - start
    if num_frames >= frame_range:
        # Wenn mehr Frames verlangt als vorhanden, dann jeden Frame nehmen
        target_indices = list(range(start, end))
    else:
        # Gleichmäßige Verteilung
        step = frame_range / num_frames
        # Start bei 0 .. num_frames-1, dann auf Frameindex mappen
        indices = []
        for i in range(num_frames):
            idx = int(start + i * step)
            if idx >= end:
                idx = end - 1
            indices.append(idx)
        # Doppelte entfernen & sortieren
        target_indices = sorted(set(indices))

    # Nochmals check, falls durch unique weniger geworden sind
    actual_target_count = len(target_indices)

    saved = 0
    target_pos = 0

    # Zur ersten Zielposition springen
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    current = start

    while current < end and target_pos < actual_target_count:
        ret, frame = cap.read()
        if not ret:
            break

        target_idx = target_indices[target_pos]

        if current == target_idx:
            # Falls Rotation gewünscht: anwenden
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            filename = f"{prefix}_{current:0{digits}d}.{ext.lower()}"
            out_path = out_dir / filename
            if out_path.exists() and not overwrite:
                # Skip existing file unless overwrite is requested
                pass
            else:
                ok = cv2.imwrite(str(out_path), frame)
                if not ok:
                    cap.release()
                    raise RuntimeError(f"Failed to write frame to {out_path}")
                saved += 1

            target_pos += 1

        current += 1

    cap.release()
    print(
        f"[{video_path.name}] Done. Requested frames: {num_frames}, "
        f"actually saved: {saved}, range: [{start}, {end}), "
        f"rotate: {rotate}°, output: {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a fixed number of frames from all videos in ../data/videos "
            "and save them to ../data/<videoname_numframes>/real/images."
        )
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        required=True,
        help="Number of frames to extract per video (e.g. 50)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix (default: video filename without extension)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Start frame index (inclusive, default: 0)",
    )
    parser.add_argument(
        "-t",
        "--end",
        type=int,
        default=None,
        help="End frame index (exclusive, default: video length)",
    )
    parser.add_argument(
        "-E",
        "--ext",
        type=str,
        choices=["jpg", "jpeg", "png", "bmp", "tiff"],
        default="jpg",
        help="Image format/extension (default: jpg)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images (default: skip existing files)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=90,
        help="Rotate frames clockwise in degrees (0, 90, 180, 270). Default: 0",
    )

    args = parser.parse_args()

    # Basis: dieses Script liegt in scripts/, data liegt eine Ebene darüber
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    videos_dir = data_dir / "videos"

    if not videos_dir.exists():
        raise FileNotFoundError(
            f"Videos directory not found: {videos_dir}\n"
            f"Expected structure: project_root/data/videos"
        )

    # Gängige Video-Extensions
    exts = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
    videos = sorted(
        p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )

    if not videos:
        raise FileNotFoundError(f"No video files found in {videos_dir}")

    print(f"Found {len(videos)} video(s) in {videos_dir}\n")

    for video_path in videos:
        print(f"Processing: {video_path.name}")
        extract_frames(
            video_path=video_path,
            data_dir=data_dir,
            num_frames=args.num_frames,
            prefix=args.prefix,
            start=args.start,
            end=args.end,
            ext=args.ext,
            overwrite=args.overwrite,
            rotate=args.rotate,
        )


if __name__ == "__main__":
    main()
