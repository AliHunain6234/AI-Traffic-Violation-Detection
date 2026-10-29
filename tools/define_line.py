"""Define the stop line by clicking two points on a video frame and save it to config.yaml.

Interactive (local machine with a display):
    python tools/define_line.py --video traffic.mp4 --camera junction_1

    Left-click twice to place the line; drag an endpoint to adjust it.
    The arrow shows the direction vehicles must travel to count as crossing.
    Keys: d = flip direction | r = reset | Enter or s = save | q or Esc = quit

Headless (Colab / SSH, no window available):
    python tools/define_line.py --video traffic.mp4 --export-frame frame.png
    # open frame.png, read two pixel coordinates, then:
    python tools/define_line.py --video traffic.mp4 --points 120,540,880,560 \\
        --direction positive --preview preview.png

Coordinates are stored in the ORIGINAL video frame together with ``frame_size``.
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

from src.calibration import MIN_LINE_PX, validate_line  # noqa: E402,F401  (shared with the dashboard's Setup page)
from src.logger import setup_logging  # noqa: E402
from src.utils import (  # noqa: E402
    COLOR_LINE,
    display_scale,
    draw_label,
    draw_line,
    load_config,
    read_frame,
    remove_approach,
    save_approach_value,
    save_config_value,
)

log = logging.getLogger("define_line")
WINDOW = "Define stop line"
GRAB_RADIUS_PX = 14


def parse_points(text: str) -> list[tuple[int, int]]:
    """Parse ``"x1,y1,x2,y2"`` into two points.

    Raises:
        ValueError: If the text is not four numbers.
    """
    try:
        v = [int(round(float(p))) for p in text.replace(" ", "").split(",")]
    except ValueError as exc:
        raise ValueError(f"Expected x1,y1,x2,y2 numbers, got {text!r}") from exc
    if len(v) != 4:
        raise ValueError(f"Expected 4 numbers (x1,y1,x2,y2), got {len(v)}")
    return [(v[0], v[1]), (v[2], v[3])]


def direction_arrow(
    a: tuple[float, float], b: tuple[float, float], direction: str, length: float = 60.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Arrow from the line's midpoint pointing to the side vehicles travel toward.

    "positive" is the side where ``utils.side_of_line`` is > 0, i.e. the arrow
    direction is ``(-dy, dx)`` of the a->b vector.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = float(np.hypot(dx, dy)) or 1.0
    sign = 1.0 if direction == "positive" else -1.0
    nx, ny = sign * (-dy) / norm, sign * dx / norm
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return mid, (mid[0] + nx * length, mid[1] + ny * length)


def render(
    frame: np.ndarray, points: list[tuple[int, int]], direction: str, help_text: bool = True
) -> np.ndarray:
    """Return a copy of ``frame`` with the line, endpoints and direction arrow drawn."""
    img = frame.copy()
    thick = max(2, frame.shape[1] // 400)
    for p in points:
        cv2.circle(img, p, thick * 3, COLOR_LINE, -1, cv2.LINE_AA)
    if len(points) == 2:
        draw_line(img, points[0], points[1], COLOR_LINE, thick)
        start, end = direction_arrow(points[0], points[1], direction, length=frame.shape[0] * 0.08)
        cv2.arrowedLine(img, tuple(int(v) for v in start), tuple(int(v) for v in end),
                        (40, 220, 40), thick, cv2.LINE_AA, tipLength=0.3)
        draw_label(img, f"direction: {direction} (arrow = travel direction)", (10, 52), scale=0.7)
    if help_text:
        msg = ("Click 2 points" if len(points) < 2 else
               "Drag to adjust | d flip direction | Enter/s save | r reset | q quit")
        draw_label(img, msg, (10, 26), scale=0.7)
    return img


class LineEditor:
    """Mouse-driven two-point line editor working in original-frame coordinates."""

    def __init__(self, frame: np.ndarray, direction: str, points: Optional[list[tuple[int, int]]] = None) -> None:
        self.frame = frame
        self.scale = display_scale(frame)
        self.direction = direction
        self.points: list[tuple[int, int]] = list(points or [])
        self._drag: Optional[int] = None

    def _to_frame(self, x: int, y: int) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return (min(w - 1, max(0, int(round(x / self.scale)))), min(h - 1, max(0, int(round(y / self.scale)))))

    def _nearest(self, x: int, y: int) -> Optional[int]:
        for i, (px, py) in enumerate(self.points):
            if np.hypot(px * self.scale - x, py * self.scale - y) <= GRAB_RADIUS_PX:
                return i
        return None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._nearest(x, y)
            if idx is not None:
                self._drag = idx
            elif len(self.points) < 2:
                self.points.append(self._to_frame(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self.points[self._drag] = self._to_frame(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def view(self) -> np.ndarray:
        img = render(self.frame, self.points, self.direction)
        if self.scale < 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return img

    def run(self) -> bool:
        """Show the window until saved (True) or cancelled (False)."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.view())
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return False
                if key in (ord("q"), 27):
                    return False
                if key == ord("r"):
                    self.points.clear()
                elif key == ord("d"):
                    self.direction = "negative" if self.direction == "positive" else "positive"
                elif key in (13, ord("s")):
                    if len(self.points) == 2:
                        return True
        finally:
            cv2.destroyAllWindows()


def save_line(config: Path, camera: str, points: list[tuple[int, int]], direction: str, frame_size: tuple[int, int]) -> None:
    """Write line, direction and frame size into ``camera.sites.<camera>`` of the config."""
    base = f"camera.sites.{camera}"
    save_config_value(config, f"{base}.stop_line", [list(p) for p in points])
    save_config_value(config, f"{base}.direction", direction)
    save_config_value(config, f"{base}.frame_size", list(frame_size))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Define the stop line and save it to config.yaml")
    p.add_argument("--video", required=True, help="video file or stream URL")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--camera", default=None, help="camera/site id (default: camera.active)")
    p.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    p.add_argument("--direction", choices=["positive", "negative"], default=None)
    p.add_argument("--points", default=None, help="headless: x1,y1,x2,y2 in original pixels")
    p.add_argument("--preview", default=None, help="write an annotated preview image here")
    p.add_argument("--export-frame", default=None, help="save the chosen frame as an image and exit")
    p.add_argument("--no-save", action="store_true", help="do not modify the config")
    p.add_argument("--approach", help="name of the approach (stop line + its own signal); use one per signal/lane group")
    p.add_argument("--remove-approach", metavar="NAME", help="delete this approach from the camera and exit")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging({"logging": {"level": "INFO"}})
    camera = args.camera or cfg.get("camera", {}).get("active", "default")
    site = cfg.get("camera", {}).get("sites", {}).get(camera, {})
    if args.remove_approach:
        ok = remove_approach(args.config, camera, args.remove_approach)
        log.info("Removed approach '%s' from camera '%s'" if ok else "No approach '%s' on camera '%s'", args.remove_approach, camera)
        return 0 if ok else 2
    if args.approach:   # work on one named approach instead of the camera-level line
        site = next((a for a in site.get("approaches") or [] if a.get("name") == args.approach), {})
    direction = args.direction or site.get("direction", "positive")

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

    if args.points:
        try:
            points = parse_points(args.points)
        except ValueError as exc:
            log.error("%s", exc)
            return 2
    else:
        preset = [tuple(p) for p in site["stop_line"]] if site.get("stop_line") else []
        editor = LineEditor(frame, direction, preset)
        try:
            saved = editor.run()
        except cv2.error as exc:
            log.error("No display available (%s). Use --export-frame and --points instead.", exc)
            return 3
        if not saved:
            log.info("Cancelled; config unchanged")
            return 1
        points, direction = editor.points, editor.direction

    try:
        validate_line(points)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
        log.warning("A point lies outside the %dx%d frame", w, h)

    if args.preview:
        cv2.imwrite(args.preview, render(frame, points, direction, help_text=False))
        log.info("Saved preview to %s", args.preview)
    if args.no_save:
        log.info("Line %s, direction %s (not saved)", points, direction)
        return 0
    if args.approach:
        save_approach_value(args.config, camera, args.approach, "stop_line", [list(p) for p in points])
        save_approach_value(args.config, camera, args.approach, "direction", direction)
        save_config_value(args.config, f"camera.sites.{camera}.frame_size", [w, h])
        log.info("Saved stop line %s (direction=%s) for approach '%s' of camera '%s'", points, direction, args.approach, camera)
        return 0
    save_line(Path(args.config), camera, points, direction, (w, h))
    log.info("Saved stop line %s (direction=%s) for camera '%s' to %s", points, direction, camera, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
