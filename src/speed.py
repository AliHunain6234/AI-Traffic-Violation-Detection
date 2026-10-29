"""Vehicle speed estimation from tracked boxes, with two calibration methods.

* ``TwoLineSpeed``   two lines a known real distance apart. Speed = distance / time between
  the vehicle's two line crossings (sub-frame interpolation, either travel direction).
* ``HomographySpeed``  four road points with known real-world metres define a plane. The
  vehicle's bottom-centre is projected onto the ground plane and speed is displacement / time.

Both report ``SpeedMeasurement``s; ``Pipeline`` compares them to the per-class limits.
Estimates are for human review, not calibrated enforcement-grade measurements.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, Iterable, Optional, Sequence

import cv2
import numpy as np

from .utils import Point, point_in_polygon, reference_point, side_of_line, within_segment

log = logging.getLogger(__name__)

Track = tuple[int, Sequence[float]]
MS_TO_KMH = 3.6


@dataclass(frozen=True)
class SpeedMeasurement:
    """One finished speed measurement for a vehicle."""

    track_id: int
    speed_kmh: float
    frame_no: int
    seconds: float
    method: str
    travel_m: float = 0.0


class SpeedEstimator(ABC):
    """Common interface: feed every frame, read live speeds and finished measurements."""

    method = "base"

    def __init__(self, params: Optional[dict] = None) -> None:
        p = params or {}
        self.max_kmh = float(p.get("max_plausible_kmh", 200))
        self.smoothing = max(1, int(p.get("smoothing_frames", 3)))
        self._done: list[SpeedMeasurement] = []
        self._display: dict[int, float] = {}

    @abstractmethod
    def update(self, frame_no: int, seconds: float, tracks: Iterable[Track]) -> dict[int, float]:
        """Process one frame; returns the latest known speed (km/h) for tracks that have one."""

    def pop_measurements(self) -> list[SpeedMeasurement]:
        """Measurements finished since the last call."""
        out, self._done = self._done, []
        return out

    def flush(self, frame_no: int, seconds: float) -> list[SpeedMeasurement]:
        """Finish anything still pending (video ended)."""
        return self.pop_measurements()

    def reset(self) -> None:
        self._done.clear()
        self._display.clear()

    @staticmethod
    def _valid(box: Sequence[float]) -> bool:
        return (box is not None and len(box) >= 4 and all(np.isfinite(v) for v in box[:4])
                and box[2] > box[0] and box[3] > box[1])


# --------------------------------------------------------------------------- #
# Method 1: two lines
# --------------------------------------------------------------------------- #
@dataclass
class _LineState:
    last_frame: int
    points: Deque[Point]
    samples: list[Deque[tuple[float, float]]]   # per line: recent (t, signed distance)
    side: list[int] = field(default_factory=lambda: [0, 0])
    cross_t: list[Optional[float]] = field(default_factory=lambda: [None, None])
    done: bool = False


class TwoLineSpeed(SpeedEstimator):
    """Speed from the time between crossing line A and line B, a known distance apart."""

    method = "lines"

    def __init__(self, line_a: Sequence[Sequence[float]], line_b: Sequence[Sequence[float]],
                 distance_m: float, params: Optional[dict] = None) -> None:
        super().__init__(params)
        p = params or {}
        if distance_m is None or float(distance_m) <= 0:
            raise ValueError("distance_m between the two lines must be a positive number of metres")
        self.lines = [(tuple(map(float, line_a[0])), tuple(map(float, line_a[1]))),
                      (tuple(map(float, line_b[0])), tuple(map(float, line_b[1])))]
        for a, b in self.lines:
            if a == b:
                raise ValueError("A speed line needs two different end points")
        self.distance_m = float(distance_m)
        self.deadband = float(p.get("line_deadband_px", 3))
        self.margin = float(p.get("line_margin_px", 20))
        self.min_travel_s = float(p.get("min_travel_seconds", 0.15))
        self.ttl = int(p.get("state_ttl_frames", 300))
        self._states: dict[int, _LineState] = {}

    def update(self, frame_no: int, seconds: float, tracks: Iterable[Track]) -> dict[int, float]:
        for tid, box in tracks:
            if not self._valid(box):
                continue
            st = self._states.get(tid)
            if st is None:
                st = self._states[tid] = _LineState(frame_no, deque(maxlen=self.smoothing),
                                                    [deque(maxlen=8), deque(maxlen=8)])
            st.last_frame = frame_no
            st.points.append(reference_point(box))
            if st.done:
                continue
            n = len(st.points)
            p = (sum(q[0] for q in st.points) / n, sum(q[1] for q in st.points) / n)
            for i, (a, b) in enumerate(self.lines):
                d = side_of_line(p, a, b)
                st.samples[i].append((seconds, d))
                if abs(d) < self.deadband:
                    continue
                side = 1 if d > 0 else -1
                prev, st.side[i] = st.side[i], side
                if prev and prev != side and within_segment(p, a, b, self.margin):
                    st.cross_t[i] = self._interpolate(st.samples[i])
            if st.cross_t[0] is not None and st.cross_t[1] is not None:
                self._finish(tid, st, frame_no, seconds)
        self._prune(frame_no)
        return dict(self._display)

    @staticmethod
    def _interpolate(samples: Deque[tuple[float, float]]) -> float:
        """Time where the signed distance crosses zero (latest sign change in the history)."""
        pts = list(samples)
        for (t0, d0), (t1, d1) in zip(reversed(pts[:-1]), reversed(pts[1:])):
            if d0 * d1 < 0:
                return t0 + (t1 - t0) * (-d0) / (d1 - d0)
            if d0 == 0:
                return t0
        return pts[-1][0]

    def _finish(self, tid: int, st: _LineState, frame_no: int, seconds: float) -> None:
        st.done = True
        t_a, t_b = st.cross_t  # type: ignore[misc]
        dt = abs(t_b - t_a)
        if dt < self.min_travel_s:
            log.debug("Track %d: line-to-line time %.3fs below minimum; discarded", tid, dt)
            return
        kmh = self.distance_m / dt * MS_TO_KMH
        if kmh > self.max_kmh:
            log.debug("Track %d: implausible speed %.0f km/h; discarded", tid, kmh)
            return
        self._display[tid] = kmh
        self._done.append(SpeedMeasurement(tid, kmh, frame_no, seconds, self.method, self.distance_m))

    def _prune(self, frame_no: int) -> None:
        for tid in [t for t, s in self._states.items() if frame_no - s.last_frame > self.ttl]:
            del self._states[tid]
            self._display.pop(tid, None)

    def reset(self) -> None:
        super().reset()
        self._states.clear()


# --------------------------------------------------------------------------- #
# Method 2: homography
# --------------------------------------------------------------------------- #
@dataclass
class _GroundState:
    last_frame: int
    points: Deque[Point]
    samples: Deque[tuple[float, float, float]] = field(default_factory=deque)   # (t, X, Y)
    speeds: list[float] = field(default_factory=list)
    first_xy: Optional[tuple[float, float]] = None
    last_xy: Optional[tuple[float, float]] = None
    last_t: float = 0.0
    done: bool = False
    inside: bool = False


class HomographySpeed(SpeedEstimator):
    """Speed from ground-plane displacement after projecting the vehicle through a homography.

    ``image_points`` are 4 pixel positions on the road surface; ``world_points`` are the same
    4 points in metres (any consistent planar frame, e.g. X across, Y along the road).
    Only positions inside the quadrilateral are used (extrapolating is unreliable).
    """

    method = "homography"

    def __init__(self, image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]],
                 params: Optional[dict] = None) -> None:
        super().__init__(params)
        p = params or {}
        img = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        wld = np.asarray(world_points, dtype=np.float32).reshape(-1, 2)
        if len(img) != 4 or len(wld) != 4:
            raise ValueError("Homography calibration needs exactly 4 image points and 4 world points")
        if cv2.contourArea(cv2.convexHull(img)) < 1.0 or cv2.contourArea(cv2.convexHull(wld)) < 1e-6:
            raise ValueError("Calibration points are collinear or duplicated")
        self.H, _ = cv2.findHomography(img, wld, 0)
        if self.H is None or not np.all(np.isfinite(self.H)):
            raise ValueError("Could not compute a homography from the calibration points")
        self.region = [tuple(map(float, pt)) for pt in cv2.convexHull(img).reshape(-1, 2)]
        self.window_s = float(p.get("window_seconds", 1.0))
        self.min_samples = int(p.get("min_samples", 8))
        self.min_travel_m = float(p.get("min_travel_m", 3.0))
        self.absent_frames = int(p.get("absent_frames", 6))
        self._states: dict[int, _GroundState] = {}

    def to_ground(self, pt: Point) -> tuple[float, float]:
        """Project an image point onto the ground plane (metres)."""
        out = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self.H)[0, 0]
        return float(out[0]), float(out[1])

    def update(self, frame_no: int, seconds: float, tracks: Iterable[Track]) -> dict[int, float]:
        seen: set[int] = set()
        for tid, box in tracks:
            if not self._valid(box):
                continue
            seen.add(tid)
            st = self._states.get(tid)
            if st is None:
                st = self._states[tid] = _GroundState(frame_no, deque(maxlen=self.smoothing))
            st.last_frame = frame_no
            st.points.append(reference_point(box))
            if st.done or len(st.points) < self.smoothing:
                continue    # skip the smoothing warm-up: a partly filled average lags and biases the first speeds
            n = len(st.points)
            p = (sum(q[0] for q in st.points) / n, sum(q[1] for q in st.points) / n)
            if point_in_polygon(p, self.region):
                st.inside = True
                x, y = self.to_ground(p)
                if st.first_xy is None:
                    st.first_xy = (x, y)
                st.last_xy, st.last_t = (x, y), seconds
                st.samples.append((seconds, x, y))
                while st.samples and seconds - st.samples[0][0] > self.window_s:
                    st.samples.popleft()
                v = self._window_speed(st)
                if v is not None and v <= self.max_kmh:
                    st.speeds.append(v)
                    self._display[tid] = float(median(st.speeds[-5:]))
            elif st.inside:
                self._finalize(tid, st, frame_no, seconds)       # left the calibrated region
        for tid, st in list(self._states.items()):
            if tid not in seen and not st.done and st.inside and frame_no - st.last_frame >= self.absent_frames:
                self._finalize(tid, st, frame_no, seconds)
        self._prune(frame_no)
        return dict(self._display)

    @staticmethod
    def _window_speed(st: _GroundState) -> Optional[float]:
        if len(st.samples) < 3:
            return None
        (t0, x0, y0), (t1, x1, y1) = st.samples[0], st.samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return float(np.hypot(x1 - x0, y1 - y0) / dt * MS_TO_KMH)

    def _finalize(self, tid: int, st: _GroundState, frame_no: int, seconds: float) -> None:
        st.done = True
        if len(st.speeds) < self.min_samples or st.first_xy is None or st.last_xy is None:
            return
        travel = float(np.hypot(st.last_xy[0] - st.first_xy[0], st.last_xy[1] - st.first_xy[1]))
        kmh = float(median(st.speeds))
        if travel < self.min_travel_m or kmh < 1.0:
            return
        self._display[tid] = kmh
        self._done.append(SpeedMeasurement(tid, kmh, frame_no, seconds, self.method, travel))

    def flush(self, frame_no: int, seconds: float) -> list[SpeedMeasurement]:
        for tid, st in list(self._states.items()):
            if not st.done and st.inside:
                self._finalize(tid, st, frame_no, seconds)
        return self.pop_measurements()

    def _prune(self, frame_no: int) -> None:
        for tid in [t for t, s in self._states.items() if frame_no - s.last_frame > 300]:
            del self._states[tid]
            self._display.pop(tid, None)

    def reset(self) -> None:
        super().reset()
        self._states.clear()


# --------------------------------------------------------------------------- #
# Limits and construction
# --------------------------------------------------------------------------- #
def limit_for(cfg: dict, class_name: str) -> float:
    """Speed limit (km/h) for a vehicle class (falls back to ``speed.default_limit_kmh``)."""
    scfg = cfg.get("speed", {})
    limits = {str(k).lower(): float(v) for k, v in (scfg.get("limits_kmh") or {}).items()}
    return limits.get(str(class_name).lower(), float(scfg.get("default_limit_kmh", 60)))


def is_overspeed(cfg: dict, class_name: str, speed_kmh: float) -> bool:
    """True if ``speed_kmh`` exceeds the class limit plus the configured tolerance."""
    tol = float(cfg.get("speed", {}).get("tolerance_kmh", 0))
    return speed_kmh > limit_for(cfg, class_name) + tol


def build_speed_estimator(cfg: dict, camera: dict) -> Optional[SpeedEstimator]:
    """Create the estimator for a camera, or ``None`` if it has no speed calibration.

    Raises:
        ValueError: If the calibration is present but invalid.
    """
    scfg = cfg.get("speed", {})
    cal = camera.get("speed")
    if not scfg.get("enabled", True) or not cal or not cal.get("method"):
        return None
    method = str(cal["method"]).lower()
    if method == "lines":
        lines = cal.get("lines")
        if not lines or len(lines) != 2:
            raise ValueError("speed.lines needs exactly two lines")
        return TwoLineSpeed(lines[0], lines[1], cal.get("distance_m"), scfg)
    if method == "homography":
        if not cal.get("image_points") or not cal.get("world_points"):
            raise ValueError("speed homography needs image_points and world_points")
        return HomographySpeed(cal["image_points"], cal["world_points"], scfg)
    raise ValueError(f"Unknown speed method {method!r} (expected 'lines' or 'homography')")
