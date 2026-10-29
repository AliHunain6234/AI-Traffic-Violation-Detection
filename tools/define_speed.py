"""Calibrate speed measurement for a camera and save it to config.yaml.

Two methods (pick the one that fits your road and what you can measure):

  lines        Two lines across the road a KNOWN distance apart (e.g. 20 m between two painted
               marks). Simple and accurate on a straight road; needs one measured distance.
  homography   Four points on the road surface that form a rectangle of KNOWN size (e.g. a lane
               3.5 m wide and 12 m long). Handles angled cameras; needs two measured numbers.

Interactive (needs a display):
    python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method lines --distance 20
        click: line A end 1, line A end 2, line B end 1, line B end 2
    python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method homography --rect 3.5x12
        click the rectangle corners in this order: near-left, near-right, far-right, far-left
    keys: z undo | r reset | Enter or s save | q or Esc quit

Headless (Colab / SSH):
    python tools/define_speed.py --video traffic.mp4 --export-frame frame.png
    python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method lines \\
        --lines "100,300,540,300;80,420,560,420" --distance 20 --preview speed.png
    python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method homography \\
        --points "300,600,500,600,460,300,340,300" --rect 3.5x12 --preview speed.png

Accuracy depends on how well the real distances were measured and on road flatness. Verify with a
vehicle of known speed (e.g. a car at a steady 40 km/h per its speedometer) before trusting results.
Pixel coordinates are in the ORIGINAL frame and are stored with ``frame_size``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.calibration import build_calibration, rect_world_points  # noqa: E402,F401  (shared with the dashboard's Setup page)
from src.logger import setup_logging  # noqa: E402
from src.utils import (  # noqa: E402
    draw_label,
    display_scale,
    load_config,
    read_frame,
    save_config_value,
)

log = logging.getLogger("define_speed")
WINDOW = "Define speed calibration"
COLOR_A, COLOR_B, COLOR_POLY = (255, 120, 40), (40, 200, 255), (60, 220, 60)
Points = list[tuple[int, int]]


def parse_numbers(text: str, count: int, what: str) -> list[float]:
    """Parse a comma/semicolon separated list that must contain exactly ``count`` numbers.

    Raises:
        ValueError: On non-numeric input or the wrong count.
    """
    try:
        vals = [float(p) for p in text.replace(";", ",").replace(" ", "").split(",") if p != ""]
    except ValueError as exc:
        raise ValueError(f"{what}: expected numbers, got {text!r}") from exc
    if len(vals) != count:
        raise ValueError(f"{what}: expected {count} numbers, got {len(vals)}")
    return vals


def parse_rect(text: str) -> tuple[float, float]:
    """Parse ``"3.5x12"`` (width x length in metres)."""
    try:
        w, l = (float(p) for p in text.lower().replace(" ", "").split("x"))
    except ValueError as exc:
        raise ValueError(f"--rect must look like 3.5x12 (width x length in metres), got {text!r}") from exc
    if w <= 0 or l <= 0:
        raise ValueError("--rect width and length must be positive")
    return w, l


def render(frame: np.ndarray, method: str, points: Points, distance_m: Optional[float],
           world: Optional[list[list[float]]], help_text: Optional[str] = None) -> np.ndarray:
    """Draw the calibration (clicked points, lines / quadrilateral, metre grid) on a copy of the frame."""
    img = frame.copy()
    thick = max(2, frame.shape[1] // 450)
    scale = max(0.6, frame.shape[1] / 1600)
    for i, p in enumerate(points):
        cv2.circle(img, p, thick * 3, COLOR_POLY, -1, cv2.LINE_AA)
        draw_label(img, str(i + 1), (p[0] + 8, p[1] - 8), scale=scale)
    if method == "lines":
        for name, color, seg in (("A", COLOR_A, points[0:2]), ("B", COLOR_B, points[2:4])):
            if len(seg) == 2:
                cv2.line(img, seg[0], seg[1], color, thick, cv2.LINE_AA)
                draw_label(img, f"line {name}", (seg[0][0], seg[0][1] - 14), bg=color, scale=scale)
        if len(points) == 4 and distance_m:
            ma = (points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2
            mb = (points[2][0] + points[3][0]) // 2, (points[2][1] + points[3][1]) // 2
            cv2.arrowedLine(img, ma, mb, (255, 255, 255), thick, cv2.LINE_AA, tipLength=0.05)
            draw_label(img, f"{distance_m:g} m", ((ma[0] + mb[0]) // 2 + 10, (ma[1] + mb[1]) // 2), scale=scale + 0.1)
    elif len(points) == 4:
        quad = np.array(points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [quad], True, COLOR_POLY, thick, cv2.LINE_AA)
        if world:
            _draw_metre_grid(img, points, world, thick)
            for p, w in zip(points, world):
                draw_label(img, f"({w[0]:g}, {w[1]:g}) m", (p[0] + 8, p[1] + 22), scale=scale * 0.9)
    if help_text:
        draw_label(img, help_text, (10, 28), scale=0.7)
    return img


def _draw_metre_grid(img: np.ndarray, points: Points, world: list[list[float]], thick: int) -> None:
    """Overlay a 1 m grid projected back into the image, a visual check of the calibration."""
    H, _ = cv2.findHomography(np.float32(world), np.float32(points), 0)
    if H is None:
        return
    xs, ys = [w[0] for w in world], [w[1] for w in world]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    def proj(x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
        return int(p[0]), int(p[1])

    for x in np.arange(x0, x1 + 1e-6, 1.0):
        cv2.line(img, proj(x, y0), proj(x, y1), (60, 220, 60), 1, cv2.LINE_AA)
    for y in np.arange(y0, y1 + 1e-6, 1.0):
        cv2.line(img, proj(x0, y), proj(x1, y), (60, 220, 60), 1, cv2.LINE_AA)


class ClickCollector:
    """Collects up to ``needed`` clicks in original-frame coordinates."""

    def __init__(self, frame: np.ndarray, needed: int = 4) -> None:
        self.frame, self.needed = frame, needed
        self.scale = display_scale(frame)
        self.points: Points = []

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.needed:
            h, w = self.frame.shape[:2]
            self.points.append((min(w - 1, max(0, int(round(x / self.scale)))), min(h - 1, max(0, int(round(y / self.scale))))))

    def run(self, view) -> bool:
        """Show ``view(points) -> image`` until saved (True) or cancelled (False)."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                img = view(self.points)
                if self.scale < 1:
                    img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                    return False
                if key == ord("z") and self.points:
                    self.points.pop()
                elif key == ord("r"):
                    self.points.clear()
                elif key in (13, ord("s")) and len(self.points) == self.needed:
                    return True
        finally:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate speed measurement and save it to config.yaml")
    p.add_argument("--video", required=True, help="video file or stream URL")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--camera", default=None, help="camera/site id (default: camera.active)")
    p.add_argument("--method", choices=["lines", "homography"], help="calibration method")
    p.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    p.add_argument("--distance", type=float, help="lines method: real distance between the two lines (metres)")
    p.add_argument("--lines", help='headless lines method: "x1,y1,x2,y2;x3,y3,x4,y4" (line A; line B)')
    p.add_argument("--points", help="headless homography: 8 numbers x1,y1,...,x4,y4 (near-left, near-right, far-right, far-left)")
    p.add_argument("--rect", help="homography: real size of the rectangle, WIDTHxLENGTH in metres, e.g. 3.5x12")
    p.add_argument("--world", help="homography alternative to --rect: 8 numbers X1,Y1,...,X4,Y4 in metres")
    p.add_argument("--preview", help="write an annotated preview image here")
    p.add_argument("--export-frame", help="save the chosen frame as an image and exit")
    p.add_argument("--no-save", action="store_true", help="validate and preview only; do not modify the config")
    p.add_argument("--clear", action="store_true", help="remove the speed calibration of this camera")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging({"logging": {"level": "INFO"}})
    camera = args.camera or cfg.get("camera", {}).get("active", "default")
    key = f"camera.sites.{camera}.speed"

    if args.clear:
        save_config_value(args.config, key, None)
        log.info("Removed speed calibration of camera '%s'", camera)
        return 0
    try:
        frame = read_frame(args.video, args.frame)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    h, w = frame.shape[:2]
    if args.export_frame:
        cv2.imwrite(args.export_frame, frame)
        log.info("Saved frame (%dx%d) to %s", w, h, args.export_frame)
        return 0
    if not args.method:
        log.error("Choose a method: --method lines or --method homography")
        return 2

    try:
        world: Optional[list[list[float]]] = None
        points: Points
        if args.method == "lines":
            if not args.distance or args.distance <= 0:
                raise ValueError("--distance (real metres between the two lines) is required and must be positive")
            if args.lines:
                nums = parse_numbers(args.lines, 8, "--lines")
                points = [(int(round(nums[i])), int(round(nums[i + 1]))) for i in range(0, 8, 2)]
            else:
                points = _click(frame, args.method, args.distance, None)
        else:
            if args.world:
                nums = parse_numbers(args.world, 8, "--world")
                world = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
            elif args.rect:
                world = rect_world_points(*parse_rect(args.rect))
            else:
                raise ValueError("homography needs --rect WIDTHxLENGTH (or --world with 8 numbers)")
            if args.points:
                nums = parse_numbers(args.points, 8, "--points")
                points = [(int(round(nums[i])), int(round(nums[i + 1]))) for i in range(0, 8, 2)]
            else:
                points = _click(frame, args.method, None, world)
        cal = build_calibration(args.method, points, args.distance, world)
    except _Cancelled:
        log.info("Cancelled; config unchanged")
        return 1
    except cv2.error as exc:
        log.error("No display available (%s). Use --export-frame, then --lines/--points.", exc)
        return 3
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
        log.warning("A point lies outside the %dx%d frame", w, h)
    if args.preview:
        cv2.imwrite(args.preview, render(frame, args.method, points, args.distance, world))
        log.info("Saved preview to %s", args.preview)
    if args.no_save:
        log.info("Calibration is valid (not saved): %s", cal)
        return 0
    save_config_value(args.config, key, cal)
    save_config_value(args.config, f"camera.sites.{camera}.frame_size", [w, h])
    log.info("Saved %s speed calibration for camera '%s' to %s", args.method, camera, args.config)
    return 0


class _Cancelled(Exception):
    pass


def _click(frame: np.ndarray, method: str, distance: Optional[float], world: Optional[list[list[float]]]) -> Points:
    """Interactive point collection; raises ``_Cancelled`` if the window is closed."""
    collector = ClickCollector(frame, 4)
    prompts = {
        "lines": ["Click line A end 1", "Click line A end 2", "Click line B end 1", "Click line B end 2"],
        "homography": ["Click NEAR-LEFT", "Click NEAR-RIGHT", "Click FAR-RIGHT", "Click FAR-LEFT"],
    }[method]

    def view(pts: Points) -> np.ndarray:
        msg = prompts[len(pts)] if len(pts) < 4 else "Enter/s = save | z undo | r reset | q quit"
        return render(frame, method, pts, distance, world, msg)

    if not collector.run(view):
        raise _Cancelled
    return collector.points


if __name__ == "__main__":
    raise SystemExit(main())
