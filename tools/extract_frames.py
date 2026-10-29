"""Step 1 of fine-tuning: pull training frames out of your videos.

    python tools/extract_frames.py --dataset rickshaw --video videos/*.mp4 --every-sec 1.5

Frames go to ``datasets/<dataset>/images`` and are recorded in ``sources.csv`` so that training can
later split train / validation BY VIDEO. Near-duplicate frames (a queue that has not moved) and blurry
frames are skipped. Run it again with more videos to grow the dataset; existing images are kept.
"""
from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.dataset import append_sources  # noqa: E402
from src.logger import setup_logging  # noqa: E402
from src.stream import parse_source  # noqa: E402
from src.utils import sharpness  # noqa: E402

log = logging.getLogger("extract_frames")


def slug(path: str) -> str:
    """Short file-system-safe name for a video file (used as the image prefix)."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem).strip("-").lower()
    return (s[:40] or "video")


def signature(frame: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail used to detect near-duplicate frames."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract(video: str, out_dir: Path, every_sec: float, max_frames: int, min_diff: float,
            min_sharpness: float, max_side: int, start_sec: float = 0.0, end_sec: Optional[float] = None) -> list[tuple[str, str, int]]:
    """Save sampled frames of one video; returns ``(image name, video, frame index)`` rows."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * every_sec))
    first = int(start_sec * fps)
    last = min(total, int(end_sec * fps)) if end_sec else total
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix, rows, last_sig = slug(video), [], None
    skipped = {"duplicate": 0, "blurry": 0}
    idx = first
    while idx < last and len(rows) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx_now, idx = idx, idx + step
        sig = signature(frame)
        if last_sig is not None and float(np.abs(sig - last_sig).mean()) < min_diff:
            skipped["duplicate"] += 1
            continue
        if min_sharpness > 0 and sharpness(frame) < min_sharpness:
            skipped["blurry"] += 1
            continue
        h, w = frame.shape[:2]
        if max_side and max(h, w) > max_side:
            k = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        name = f"{prefix}_{idx_now:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((name, video, idx_now))
        last_sig = sig
    cap.release()
    log.info("%s: saved %d frames (skipped %d near-duplicates, %d blurry)", Path(video).name, len(rows), skipped["duplicate"], skipped["blurry"])
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract training frames from videos")
    p.add_argument("--dataset", required=True, help="dataset name; frames go to datasets/<name>/images")
    p.add_argument("--video", nargs="+", required=True, help="video files (wildcards allowed)")
    p.add_argument("--root", default="datasets", help="folder holding all datasets")
    p.add_argument("--every-sec", type=float, default=1.0, help="sample one frame every N seconds")
    p.add_argument("--max-per-video", type=int, default=400)
    p.add_argument("--min-diff", type=float, default=6.0, help="skip a frame that differs from the last saved one by less than this (0 = keep all)")
    p.add_argument("--min-sharpness", type=float, default=0.0, help="skip frames sharper-than test (variance of Laplacian), e.g. 40; 0 = off")
    p.add_argument("--max-side", type=int, default=1600, help="downscale so the longest side is at most this (0 = keep size)")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--end-sec", type=float, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging({"logging": {"level": "INFO"}})
    files = sorted({f for pattern in args.video for f in (glob.glob(pattern) or [pattern])})
    out_dir = Path(args.root) / args.dataset / "images"
    sources = Path(args.root) / args.dataset / "sources.csv"
    total = 0
    for f in files:
        if not isinstance(parse_source(f), str) or not Path(f).is_file():
            log.error("Not a video file: %s", f)
            return 2
        try:
            rows = extract(f, out_dir, args.every_sec, args.max_per_video, args.min_diff, args.min_sharpness,
                           args.max_side, args.start_sec, args.end_sec)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        append_sources(sources, rows)
        total += len(rows)
    log.info("Done: %d new frames in %s. Next: python tools/prelabel.py --dataset %s", total, out_dir, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
