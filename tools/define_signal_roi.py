"""Define the traffic-signal ROI by drawing a box around the light and save it to config.yaml.

Interactive (local machine with a display):
    python tools/define_signal_roi.py --video traffic.mp4 --camera junction_1 --frame 300

    Drag a box tightly around the signal head, then press Enter/Space.
    The window then shows what the HSV classifier reads inside the box.
    Keys: s = save | r = redo | q or Esc = quit without saving
    Tip: pick a frame where the light is clearly lit (use --frame).

Headless (Colab / SSH):
    python tools/define_signal_roi.py --video traffic.mp4 --export-frame frame.png
    python tools/define_signal_roi.py --video traffic.mp4 --roi 812,96,38,110 \\
        --preview roi.png --scan 40

``--scan N`` classifies the ROI on N evenly spaced frames and prints how often each
state appears, a quick check that the HSV thresholds suit this camera.

The ROI is [x, y, w, h] in the ORIGINAL video frame and is stored with ``frame_size``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.calibration import MIN_ROI_PX, validate_roi  # noqa: E402,F401  (shared with the dashboard's Setup page)
from src.logger import setup_logging  # noqa: E402
from src.signal_classifier import SignalReading, build_classifier  # noqa: E402
from src.utils import (  # noqa: E402
    SIGNAL_COLORS,
    display_scale,
    draw_label,
    load_config,
    read_frame,
    safe_crop,
    save_approach_value,
    save_config_value,
)

log = logging.getLogger("define_signal_roi")
WINDOW = "Define signal ROI"


def parse_roi(text: str) -> list[int]:
    """Parse ``"x,y,w,h"``.

    Raises:
        ValueError: If the text is not four numbers or the size is not positive.
    """
    try:
        v = [int(round(float(p))) for p in text.replace(" ", "").split(",")]
    except ValueError as exc:
        raise ValueError(f"Expected x,y,w,h numbers, got {text!r}") from exc
    if len(v) != 4:
        raise ValueError(f"Expected 4 numbers (x,y,w,h), got {len(v)}")
    return v


def classify_roi(cfg: dict, frame: np.ndarray, roi: list[int]) -> SignalReading:
    """Run the configured classifier on the ROI of one frame."""
    x, y, w, h = roi
    crop = safe_crop(frame, (x, y, x + w, y + h))
    return build_classifier(cfg).classify(crop) if crop is not None else SignalReading()


def scan_video(cfg: dict, video: str, roi: list[int], samples: int) -> Counter:
    """Classify the ROI on ``samples`` evenly spaced frames; returns state counts."""
    cap = cv2.VideoCapture(video)
    counts: Counter = Counter()
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        indices = np.linspace(0, max(0, total - 1), num=max(1, samples)).astype(int) if total else [0]
        clf = build_classifier(cfg)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            x, y, w, h = roi
            crop = safe_crop(frame, (x, y, x + w, y + h))
            counts[clf.classify(crop).state if crop is not None else "unknown"] += 1
    finally:
        cap.release()
    return counts


def render(frame: np.ndarray, roi: list[int], reading: SignalReading) -> np.ndarray:
    """Return a copy of ``frame`` with the ROI outlined in the detected state's colour."""
    img = frame.copy()
    x, y, w, h = roi
    color = SIGNAL_COLORS.get(reading.state, SIGNAL_COLORS["unknown"])
    thick = max(2, frame.shape[1] // 500)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
    draw_label(img, f"ROI reads: {reading.state.upper()} ({reading.confidence:.2f})",
               (x, max(18, y - 8)), bg=color, scale=0.7)
    return img


def select_roi_interactive(frame: np.ndarray) -> Optional[list[int]]:
    """Let the user drag a box; returns ``[x, y, w, h]`` in original pixels or ``None``."""
    scale = display_scale(frame)
    shown = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    x, y, w, h = cv2.selectROI(WINDOW, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW)
    if w == 0 or h == 0:
        return None
    return [int(round(v / scale)) for v in (x, y, w, h)]


def confirm_interactive(cfg: dict, frame: np.ndarray, roi: list[int]) -> str:
    """Show the ROI with its classification; returns ``save`` / ``redo`` / ``quit``."""
    scale = display_scale(frame)
    img = render(frame, roi, classify_roi(cfg, frame, roi))
    draw_label(img, "s = save | r = redo | q = quit", (10, 26), scale=0.7)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW, img)
    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                return "quit"
            if key in (ord("s"), 13):
                return "save"
            if key == ord("r"):
                return "redo"
    finally:
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Define the traffic-signal ROI and save it to config.yaml")
    p.add_argument("--video", required=True, help="video file or stream URL")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--camera", default=None, help="camera/site id (default: camera.active)")
    p.add_argument("--frame", type=int, default=0, help="frame index to draw on (pick one with the light lit)")
    p.add_argument("--roi", default=None, help="headless: x,y,w,h in original pixels")
    p.add_argument("--preview", default=None, help="write an annotated preview image here")
    p.add_argument("--scan", type=int, default=0, help="classify the ROI on N sampled frames and report")
    p.add_argument("--export-frame", default=None, help="save the chosen frame as an image and exit")
    p.add_argument("--no-save", action="store_true", help="do not modify the config")
    p.add_argument("--approach", help="save the ROI for this approach (the signal that controls that stop line)")
    p.add_argument("--min-pixel-ratio", type=float,
                   help="small lamp? lower the share of ROI pixels a colour needs to count (default 0.02); saved for this camera/approach")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging({"logging": {"level": "INFO"}})
    camera = args.camera or cfg.get("camera", {}).get("active", "default")
    # judge the ROI with the same signal settings the pipeline will use for this camera
    site = cfg.get("camera", {}).get("sites", {}).get(camera, {})
    over = dict(site.get("signal") or {})
    if args.approach:
        over.update(next((a.get("signal") or {} for a in site.get("approaches") or [] if a.get("name") == args.approach), {}))
    if args.min_pixel_ratio is not None:
        over["min_pixel_ratio"] = args.min_pixel_ratio
    if over:
        cfg = {**cfg, "signal": {**cfg.get("signal", {}), **over}}

    try:
        frame = read_frame(args.video, args.frame)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    h, w = frame.shape[:2]
    log.info("Frame %d: %dx%d", args.frame, w, h)

    if args.export_frame:
        cv2.imwrite(args.export_frame, frame)
        log.info("Saved frame to %s", args.export_frame)
        return 0

    try:
        if args.roi:
            roi = validate_roi(parse_roi(args.roi), (w, h))
        else:
            while True:
                try:
                    picked = select_roi_interactive(frame)
                except cv2.error as exc:
                    log.error("No display available (%s). Use --export-frame and --roi instead.", exc)
                    return 3
                if picked is None:
                    log.info("Cancelled; config unchanged")
                    return 1
                roi = validate_roi(picked, (w, h))
                action = confirm_interactive(cfg, frame, roi)
                if action == "save":
                    break
                if action == "quit":
                    log.info("Cancelled; config unchanged")
                    return 1
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    reading = classify_roi(cfg, frame, roi)
    log.info("ROI %s reads %s (conf %.2f) on frame %d", roi, reading.state, reading.confidence, args.frame)
    if reading.state == "unknown":
        log.warning("Classifier sees no lit signal here; check the ROI or try a frame where the light is on")
    if args.scan > 0:
        counts = scan_video(cfg, args.video, roi, args.scan)
        log.info("Scan of %d frames: %s", sum(counts.values()), dict(counts))
    if args.preview:
        cv2.imwrite(args.preview, render(frame, roi, reading))
        log.info("Saved preview to %s", args.preview)
    if args.no_save:
        log.info("ROI %s (not saved)", roi)
        return 0
    base = f"camera.sites.{camera}"
    if args.approach:
        save_approach_value(args.config, camera, args.approach, "signal_roi", roi)
    else:
        save_config_value(args.config, f"{base}.signal_roi", roi)
    if args.min_pixel_ratio is not None:
        if args.approach:
            save_approach_value(args.config, camera, args.approach, "signal", {"min_pixel_ratio": args.min_pixel_ratio})
        else:
            save_config_value(args.config, f"{base}.signal", {"min_pixel_ratio": args.min_pixel_ratio})
    save_config_value(args.config, f"{base}.frame_size", [w, h])
    log.info("Saved signal ROI %s for %s to %s", roi,
             f"approach '{args.approach}' of camera '{camera}'" if args.approach else f"camera '{camera}'", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
