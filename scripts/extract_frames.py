#!/usr/bin/env python3
"""
Extract frames from a video and save them with name + index.
Requires: opencv-python

Examples:
  python extract_frames.py input.mp4 -o frames
  python extract_frames.py input.mp4 -o frames -p myshot -E png
  python extract_frames.py input.mp4 -o frames -s 100 -t 1000 -e 5
"""

import cv2
import os
import argparse
from pathlib import Path

def extract_frames(
    video_path: Path,
    out_dir: Path,
    prefix: str | None = None,
    start: int = 0,
    end: int | None = None,
    every: int = 1,
    ext: str = "jpg",
    overwrite: bool = False,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if end is None or end > total_frames:
        end = total_frames

    if start < 0 or start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, total={total_frames}")

    # Default prefix = video filename (without extension)
    if not prefix:
        prefix = video_path.stem

    # Zero padding based on end-1
    digits = max(4, len(str(max(0, end - 1))))

    saved = 0
    current = 0

    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    current = start

    while current < end:
        ret, frame = cap.read()
        if not ret:
            break

        if (current - start) % every == 0:
            filename = f"{prefix}_{current:0{digits}d}.{ext.lower()}"
            out_path = out_dir / filename
            if out_path.exists() and not overwrite:
                # Skip existing file unless overwrite is requested
                pass
            else:
                ok = cv2.imwrite(str(out_path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to write frame to {out_path}")
                saved += 1

        current += 1

    cap.release()
    print(
        f"Done. Frames considered: {end - start}, saved: {saved}, "
        f"range: [{start}, {end}), step: {every}, output: {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and save with name + index."
    )
    parser.add_argument("video", type=Path, help="Path to input video (e.g., input.mp4)")
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path("frames"),
        help="Output directory (default: ./frames)"
    )
    parser.add_argument(
        "-p", "--prefix", type=str, default=None,
        help="Filename prefix (default: video filename without extension)"
    )
    parser.add_argument(
        "-s", "--start", type=int, default=0,
        help="Start frame index (inclusive, default: 0)"
    )
    parser.add_argument(
        "-t", "--end", type=int, default=None,
        help="End frame index (exclusive, default: video length)"
    )
    parser.add_argument(
        "-e", "--every", type=int, default=1,
        help="Save every Nth frame (default: 1 = every frame)"
    )
    parser.add_argument(
        "-E", "--ext", type=str, choices=["jpg", "jpeg", "png", "bmp", "tiff"], default="jpg",
        help="Image format/extension (default: jpg)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing images (default: skip existing files)"
    )

    args = parser.parse_args()
    extract_frames(
        video_path=args.video,
        out_dir=args.out_dir,
        prefix=args.prefix,
        start=args.start,
        end=args.end,
        every=max(1, args.every),
        ext=args.ext,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
