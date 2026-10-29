"""Shared helpers: config I/O, geometry, time, drawing and small image utilities.

Everything here is dependency-light and free of pipeline state so that every other
module (and the tests) can import it safely.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Iterable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2

# BGR colours used across overlays (red = violation, amber = caution, green = legal).
COLOR_RED = (60, 60, 235)
COLOR_YELLOW = (0, 190, 250)
COLOR_GREEN = (80, 200, 90)
COLOR_UNKNOWN = (160, 160, 160)
COLOR_LINE = (255, 120, 40)
COLOR_TEXT = (255, 255, 255)
SIGNAL_COLORS = {
    "red": COLOR_RED,
    "yellow": COLOR_YELLOW,
    "green": COLOR_GREEN,
    "unknown": COLOR_UNKNOWN,
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _yaml():
    """Return a round-trip YAML instance (keeps comments when saving)."""
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120
    return y


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg['a']['b']['c'] = value`` from ``'a.b.c'``, creating dicts as needed."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def to_plain(obj: Any) -> Any:
    """Convert ruamel containers into plain dict/list so downstream code is simple."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def load_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> dict:
    """Load a YAML config and apply dotted-key overrides.

    Args:
        path: Path to ``config.yaml``.
        overrides: Mapping like ``{"detector.model": "yolov8n.pt"}``. ``None``
            values are ignored so unset CLI flags never clobber the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = to_plain(_yaml().load(fh) or {})
    for key, value in (overrides or {}).items():
        if value is not None:
            set_dotted(cfg, key, value)
    return cfg


def _to_flow(value: Any) -> Any:
    """Turn (nested) lists into flow-style YAML sequences, e.g. ``[[10, 20], [30, 40]]``."""
    if isinstance(value, dict):
        return {k: _to_flow(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        from ruamel.yaml.comments import CommentedSeq

        seq = CommentedSeq([_to_flow(v) for v in value])
        seq.fa.set_flow_style()
        return seq
    return value


def save_config_value(path: str | Path, dotted_key: str, value: Any) -> None:
    """Write one value into the YAML file in place, preserving comments/layout.

    Used by the line / ROI tools to persist their results to ``config.yaml``.
    """
    path = Path(path)
    y = _yaml()
    with path.open("r", encoding="utf-8") as fh:
        data = y.load(fh)
    set_dotted(data, dotted_key, _to_flow(value))
    with path.open("w", encoding="utf-8") as fh:
        y.dump(data, fh)
    log.info("Saved %s to %s", dotted_key, path)


def get_camera(cfg: dict, camera_id: Optional[str] = None) -> dict:
    """Return the active camera/site block (``camera.sites[<id>]``) with its ``id``.

    Raises:
        KeyError: If the camera id is not defined in the config.
    """
    cam_cfg = cfg.get("camera", {})
    cam_id = camera_id or cam_cfg.get("active", "default")
    sites = cam_cfg.get("sites", {})
    if cam_id not in sites:
        raise KeyError(
            f"Camera '{cam_id}' not found in config (have: {list(sites)}). "
            f"Create it first: python tools/define_line.py --video <your video> --camera {cam_id}"
        )
    site = dict(sites[cam_id])
    site["id"] = cam_id
    return site


def get_approaches(camera: dict) -> list[dict]:
    """Normalised list of approaches (stop line + its own signal) for a camera block.

    A camera with an ``approaches:`` list uses it (each entry: ``name``, ``stop_line``,
    ``direction``, ``signal_roi``, optional ``violation_zones``). Older single-line configs
    (``stop_line`` / ``signal_roi`` directly on the camera) become one approach named ``main``.

    Raises:
        ValueError: If two approaches share a name.
    """
    raw = camera.get("approaches") or []
    if not raw:
        return [{
            "name": "main", "stop_line": camera.get("stop_line"), "direction": camera.get("direction", "positive"),
            "signal_roi": camera.get("signal_roi"), "violation_zones": camera.get("violation_zones") or [],
            "signal_overrides": dict(camera.get("signal") or {}),
        }]
    out = [{
        "name": str(a.get("name") or f"approach{i + 1}"), "stop_line": a.get("stop_line"),
        "direction": a.get("direction", "positive"), "signal_roi": a.get("signal_roi"),
        "violation_zones": a.get("violation_zones") or [],
        # camera-level `signal:` overrides apply to every approach; an approach's own `signal:` wins
        "signal_overrides": {**(camera.get("signal") or {}), **(a.get("signal") or {})},
    } for i, a in enumerate(raw)]
    names = [a["name"] for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Approach names must be unique, got {names}")
    return out


def save_approach_value(path: str | Path, camera_id: str, approach: str, key: str, value: Any) -> None:
    """Write ``key`` of the named approach (created if missing) into the config, keeping comments."""
    path = Path(path)
    y = _yaml()
    with path.open("r", encoding="utf-8") as fh:
        data = y.load(fh)
    cam = data.setdefault("camera", {}).setdefault("sites", {}).setdefault(camera_id, {})
    items = cam.get("approaches")
    if items is None:
        items = cam["approaches"] = []
    entry = next((a for a in items if a.get("name") == approach), None)
    if entry is None:
        entry = {"name": approach}
        items.append(entry)
    entry[key] = _to_flow(value)
    with path.open("w", encoding="utf-8") as fh:
        y.dump(data, fh)
    log.info("Saved %s of approach '%s' (camera '%s') to %s", key, approach, camera_id, path)


def remove_approach(path: str | Path, camera_id: str, approach: str) -> bool:
    """Delete an approach from the config; returns False if it did not exist."""
    path = Path(path)
    y = _yaml()
    with path.open("r", encoding="utf-8") as fh:
        data = y.load(fh)
    items = ((data.get("camera") or {}).get("sites") or {}).get(camera_id, {}).get("approaches") or []
    keep = [a for a in items if a.get("name") != approach]
    if len(keep) == len(items):
        return False
    data["camera"]["sites"][camera_id]["approaches"] = keep
    with path.open("w", encoding="utf-8") as fh:
        y.dump(data, fh)
    return True


def resolve_device(preference: str = "auto") -> str:
    """Resolve ``auto`` to ``cuda`` / ``mps`` / ``cpu`` depending on availability."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def bottom_center(box: Sequence[float]) -> Point:
    """Bottom-centre of an ``x1, y1, x2, y2`` box (where the wheels meet the road)."""
    x1, _, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, float(y2))


def box_center(box: Sequence[float]) -> Point:
    """Centre of an ``x1, y1, x2, y2`` box."""
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def reference_point(box: Sequence[float], mode: str = "bottom_center") -> Point:
    """Return the point used for line tests, as selected in the config."""
    return box_center(box) if mode == "center" else bottom_center(box)


def side_of_line(p: Point, a: Point, b: Point) -> float:
    """Signed distance of ``p`` from the infinite line a->b (in pixels).

    The sign flips when ``p`` moves across the line; ~0 means on the line.
    With image coordinates (y down) positive is to the visual right of a->b.

    Raises:
        ValueError: If ``a`` and ``b`` are identical.
    """
    ax, ay = a
    bx, by = b
    length = float(np.hypot(bx - ax, by - ay))
    if length == 0.0:
        raise ValueError("Degenerate line: both endpoints are identical")
    return ((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)) / length


def project_on_segment(p: Point, a: Point, b: Point) -> float:
    """Normalised position of ``p`` projected on segment a->b (0 = a, 1 = b)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        raise ValueError("Degenerate line: both endpoints are identical")
    return ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom


def within_segment(p: Point, a: Point, b: Point, margin_px: float = 0.0) -> bool:
    """True if ``p`` projects onto segment a->b, allowing ``margin_px`` of slack."""
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if length == 0.0:
        return False
    slack = margin_px / length
    return -slack <= project_on_segment(p, a, b) <= 1.0 + slack


def point_in_polygon(p: Point, polygon: Sequence[Point]) -> bool:
    """True if ``p`` lies inside (or on the edge of) ``polygon``."""
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0


def distance(p: Point, q: Point) -> float:
    """Euclidean distance between two points."""
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a box to the frame and return integer ``x1, y1, x2, y2``."""
    x1, y1, x2, y2 = box[:4]
    return (
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    )


def pad_box(box: Sequence[float], frac: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow a box by ``frac`` of its size on each side, clipped to the frame."""
    x1, y1, x2, y2 = box[:4]
    pw, ph = (x2 - x1) * frac, (y2 - y1) * frac
    return clip_box((x1 - pw, y1 - ph, x2 + pw, y2 + ph), width, height)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_start_time(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 start time (``Z`` accepted). ``None``/empty returns ``None``."""
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def wall_clock(start: Optional[datetime], video_seconds: float) -> str:
    """ISO wall-clock time of a video position.

    Uses ``start + video_seconds`` when a start time is configured, otherwise the
    current processing time (recorded video has no real wall clock).
    """
    if start is None:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")
    return (start + timedelta(seconds=video_seconds)).isoformat(timespec="milliseconds")


class RollingFPS:
    """Rolling-window frames-per-second meter."""

    def __init__(self, window: int = 30) -> None:
        self._stamps: Deque[float] = deque(maxlen=max(2, window))

    def tick(self) -> float:
        """Record a processed frame and return the current FPS estimate."""
        self._stamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / elapsed if elapsed > 0 else 0.0


# --------------------------------------------------------------------------- #
# Images / drawing
# --------------------------------------------------------------------------- #
def is_valid_frame(frame: Optional[np.ndarray]) -> bool:
    """True for a non-empty 3-channel image."""
    return frame is not None and frame.size > 0 and frame.ndim == 3 and frame.shape[2] == 3


def read_frame(video: str | Path, index: int = 0) -> np.ndarray:
    """Read one frame from a video file or stream URL.

    Raises:
        FileNotFoundError: If the video cannot be opened.
        ValueError: If the requested frame cannot be read.
    """
    from .stream import is_live_source, open_source, parse_source

    source = parse_source(video)
    if is_live_source(source):           # webcam / stream: seeking is impossible, so read forward
        live = open_source(source)
        try:
            warmup = 10 if isinstance(source, int) else 0      # webcams need a few frames to settle exposure
            frame = None
            for _ in range(warmup + max(0, index) + 1):
                item = live.read()
                if item is None:
                    break
                frame = item[0]
        finally:
            live.close()
        if not is_valid_frame(frame):
            raise ValueError(f"Could not read a frame from live source {video!r}")
        return frame  # type: ignore[return-value]
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video}")
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or not is_valid_frame(frame):
            raise ValueError(f"Could not read frame {index} from {video}")
        return frame
    finally:
        cap.release()


def display_scale(frame: np.ndarray, max_w: int = 1280, max_h: int = 800) -> float:
    """Scale factor (<= 1) that fits ``frame`` inside a max_w x max_h window."""
    h, w = frame.shape[:2]
    return min(1.0, max_w / w, max_h / h)


def safe_crop(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """Crop ``box`` from ``frame``; ``None`` if the clipped region is empty."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2].copy()


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: higher means a sharper image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = COLOR_TEXT,
    bg: Optional[tuple[int, int, int]] = (30, 30, 30),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw text with an optional filled background; ``origin`` is the bottom-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(2, min(origin[0], frame.shape[1] - tw - 4)))  # keep the label inside the frame
    y = int(origin[1])
    if bg is not None:
        cv2.rectangle(frame, (x - 2, y - th - base - 2), (x + tw + 2, y + base), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_line(
    frame: np.ndarray,
    a: Point,
    b: Point,
    color: tuple[int, int, int] = COLOR_LINE,
    thickness: int = 2,
) -> None:
    """Draw the stop line."""
    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)


def draw_polygon(
    frame: np.ndarray,
    polygon: Iterable[Point],
    color: tuple[int, int, int] = COLOR_LINE,
    thickness: int = 2,
) -> None:
    """Draw a closed polygon outline."""
    pts = np.asarray(list(polygon), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
