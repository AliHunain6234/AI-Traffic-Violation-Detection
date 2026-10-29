"""Stop-line / violation-zone crossing detection on top of tracked boxes.

Design notes:
  * The test point is the bottom-centre of the box (where the wheels touch the road).
  * A crossing is a *sign change* of the point's signed distance to the line, but
    only after the point has left a small dead-band around the line. The dead-band
    gives hysteresis, so box jitter or a vehicle stopped on the line never flips
    the side by accident.
  * Only crossings toward ``direction`` count, and only if the point projects onto
    the line segment (plus a margin), so the infinite extension of the line is ignored.
  * Each track ID triggers at most once (optionally re-armed after a cooldown).
  * If a crossed vehicle gets a new ID right after the line (ID fragmentation), the
    new ID inherits the "crossed" flag so it is not counted twice.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Sequence

from .utils import (
    Point,
    distance,
    point_in_polygon,
    reference_point,
    side_of_line,
    within_segment,
)

log = logging.getLogger(__name__)

Track = tuple[int, Sequence[float]]  # (track_id, [x1, y1, x2, y2])


@dataclass(frozen=True)
class CrossingEvent:
    """A vehicle crossing the stop line (or entering a violation zone)."""

    track_id: int
    frame_no: int
    point: Point
    kind: str = "line"          # "line" | "zone"
    zone_index: Optional[int] = None
    speed_px: float = 0.0       # smoothed px/frame at the moment of crossing
    signed_distance: float = 0.0


@dataclass
class _TrackState:
    first_frame: int
    last_frame: int
    points: Deque[Point]
    smooth: Optional[Point] = None
    side: int = 0                           # last confident side: -1 / +1 / 0 unknown
    in_zone: dict[int, bool] = field(default_factory=dict)
    crossed: bool = False
    last_cross_frame: Optional[int] = None
    speed: float = 0.0


class LineCrossingDetector:
    """Detects tracked vehicles crossing a stop line and/or entering violation zones.

    Args:
        line: ``((x1, y1), (x2, y2))`` stop line in pixels, or ``None``.
        direction: ``"positive"`` or ``"negative"``: the side of the line (as given
            by ``utils.side_of_line``) that vehicles travel *toward*.
        params: The ``crossing`` section of the config.
        zones: Optional polygons; entering one counts as a crossing.
    """

    def __init__(
        self,
        line: Optional[Sequence[Sequence[float]]],
        direction: str = "positive",
        params: Optional[dict] = None,
        zones: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ) -> None:
        p = params or {}
        self.line: Optional[tuple[Point, Point]] = (
            (tuple(line[0]), tuple(line[1])) if line else None  # type: ignore[assignment]
        )
        if direction not in ("positive", "negative"):
            raise ValueError(f"direction must be 'positive' or 'negative', got {direction!r}")
        self.direction_sign = 1 if direction == "positive" else -1
        self.zones = [[(float(x), float(y)) for x, y in poly] for poly in (zones or [])]
        if self.line is None and not self.zones:
            log.warning("No stop line or violation zone configured: nothing will be detected")
        if self.line is not None and self.line[0] == self.line[1]:
            raise ValueError("Stop line endpoints must differ")

        self.ref_mode = p.get("reference_point", "bottom_center")
        self.smoothing = max(1, int(p.get("point_smoothing_frames", 3)))
        self.deadband = float(p.get("min_displacement_px", 4))
        self.margin = float(p.get("segment_margin_px", 20))
        self.min_age = int(p.get("min_track_age_frames", 5))
        self.once_per_id = bool(p.get("once_per_id", True))
        self.cooldown = int(p.get("cooldown_frames", 90))
        self.ttl = int(p.get("state_ttl_frames", 300))
        self.dedup_window = int(p.get("dedup_window_frames", 45))
        self.dedup_radius = float(p.get("dedup_radius_px", 80))

        self._states: dict[int, _TrackState] = {}
        self._crossed_ids: set[int] = set()
        # crossed track id -> (last frame seen, last smoothed point); for ID-fragmentation dedup
        self._recent_crossed: dict[int, tuple[int, Point]] = {}

    @classmethod
    def from_config(cls, cfg: dict, camera: dict) -> "LineCrossingDetector":
        """Build from the full config and a camera block (see ``utils.get_camera``)."""
        return cls(
            line=camera.get("stop_line"),
            direction=camera.get("direction", "positive"),
            params=cfg.get("crossing", {}),
            zones=camera.get("violation_zones") or [],
        )

    # ------------------------------------------------------------------ public
    @property
    def crossed_count(self) -> int:
        """Number of distinct track IDs that have crossed so far."""
        return len(self._crossed_ids)

    def has_crossed(self, track_id: int) -> bool:
        """True if this track ID has already triggered a crossing."""
        return track_id in self._crossed_ids

    def is_stationary(self, track_id: int, threshold_px: float) -> bool:
        """True if the track's smoothed speed is below ``threshold_px`` px/frame."""
        st = self._states.get(track_id)
        return st is not None and st.speed < threshold_px

    def reset(self) -> None:
        """Forget all tracks (call between videos)."""
        self._states.clear()
        self._crossed_ids.clear()
        self._recent_crossed.clear()

    def update(self, frame_no: int, tracks: Iterable[Track]) -> list[CrossingEvent]:
        """Feed one frame of tracked boxes and return the crossings it produced.

        Boxes with NaN/inf values or non-positive size are skipped.
        """
        events: list[CrossingEvent] = []
        for track_id, box in tracks:
            if not self._valid_box(box):
                continue
            raw = reference_point(box, self.ref_mode)
            st = self._states.get(track_id)
            if st is None:
                st = self._new_state(track_id, frame_no, raw)
            self._advance(st, frame_no, raw)
            assert st.smooth is not None

            if self.line is not None:
                ev = self._check_line(st, track_id, frame_no)
                if ev:
                    events.append(ev)
            for idx, poly in enumerate(self.zones):
                ev = self._check_zone(st, track_id, frame_no, idx, poly)
                if ev:
                    events.append(ev)
            if st.crossed:
                self._recent_crossed[track_id] = (frame_no, st.smooth)

        self._prune(frame_no)
        return events

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _valid_box(box: Sequence[float]) -> bool:
        if box is None or len(box) < 4 or not all(math.isfinite(v) for v in box[:4]):
            return False
        return box[2] > box[0] and box[3] > box[1]

    def _new_state(self, track_id: int, frame_no: int, first_point: Point) -> _TrackState:
        st = _TrackState(
            first_frame=frame_no,
            last_frame=frame_no,
            points=deque(maxlen=self.smoothing),
        )
        self._states[track_id] = st
        if track_id in self._crossed_ids:
            st.crossed = True
        elif self.once_per_id:
            self._inherit_crossed(st, track_id, frame_no, first_point)
        return st

    def _inherit_crossed(self, st: _TrackState, track_id: int, frame_no: int, p: Point) -> None:
        """A new ID that appears where a just-crossed track vanished is the same vehicle."""
        for old_id, (last_frame, last_point) in list(self._recent_crossed.items()):
            gap = frame_no - last_frame
            if 0 < gap <= self.dedup_window and distance(p, last_point) <= self.dedup_radius:
                st.crossed = True
                st.last_cross_frame = last_frame
                self._crossed_ids.add(track_id)
                del self._recent_crossed[old_id]
                log.debug("Track %d inherits crossed flag from lost track %d", track_id, old_id)
                return

    def _advance(self, st: _TrackState, frame_no: int, raw: Point) -> None:
        """Push a new point, update the smoothed point and speed."""
        st.points.append(raw)
        n = len(st.points)
        smooth = (sum(q[0] for q in st.points) / n, sum(q[1] for q in st.points) / n)
        if st.smooth is not None:
            dt = max(1, frame_no - st.last_frame)
            st.speed = distance(smooth, st.smooth) / dt
        st.smooth = smooth
        st.last_frame = frame_no

    def _check_line(self, st: _TrackState, track_id: int, frame_no: int) -> Optional[CrossingEvent]:
        assert self.line is not None and st.smooth is not None
        a, b = self.line
        d = side_of_line(st.smooth, a, b)
        if abs(d) < self.deadband:
            return None  # inside the dead-band: keep the previous side
        new_side = 1 if d > 0 else -1
        prev, st.side = st.side, new_side
        if prev == 0 or prev == new_side:
            return None
        if new_side != self.direction_sign:
            return None  # crossed against the configured direction of travel
        if not within_segment(st.smooth, a, b, self.margin):
            return None  # passed around the end of the line
        return self._emit(st, track_id, frame_no, "line", None, d)

    def _check_zone(
        self, st: _TrackState, track_id: int, frame_no: int, idx: int, poly: list[Point]
    ) -> Optional[CrossingEvent]:
        assert st.smooth is not None
        inside = point_in_polygon(st.smooth, poly)
        was_inside = st.in_zone.get(idx)
        st.in_zone[idx] = inside
        if was_inside is False and inside:
            return self._emit(st, track_id, frame_no, "zone", idx, 0.0)
        return None

    def _emit(
        self, st: _TrackState, track_id: int, frame_no: int, kind: str,
        zone_index: Optional[int], signed_distance: float,
    ) -> Optional[CrossingEvent]:
        assert st.smooth is not None
        if frame_no - st.first_frame + 1 < self.min_age:
            return None  # too young: likely a spurious track
        if st.crossed:
            if self.once_per_id:
                return None
            if st.last_cross_frame is not None and frame_no - st.last_cross_frame < self.cooldown:
                return None
        st.crossed = True
        st.last_cross_frame = frame_no
        self._crossed_ids.add(track_id)
        return CrossingEvent(
            track_id=track_id,
            frame_no=frame_no,
            point=st.smooth,
            kind=kind,
            zone_index=zone_index,
            speed_px=st.speed,
            signed_distance=signed_distance,
        )

    def _prune(self, frame_no: int) -> None:
        stale = [tid for tid, st in self._states.items() if frame_no - st.last_frame > self.ttl]
        for tid in stale:
            del self._states[tid]
        for tid in [t for t, (f, _) in self._recent_crossed.items() if frame_no - f > self.dedup_window]:
            del self._recent_crossed[tid]
