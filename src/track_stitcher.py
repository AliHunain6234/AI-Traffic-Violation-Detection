"""Heals tracker ID fragmentation: a vehicle that loses its ID mid-scene gets its old ID back.

BoT-SORT / ByteTrack drop and re-create IDs under occlusion (a bus passes in front, a motorcycle
is hidden behind a car). Each fragment then looks like a new vehicle, which breaks "once per ID"
crossing logic, speed measurement across two lines, and every count.

``TrackStitcher`` sits between the tracker and the rest of the pipeline. When a brand-new raw ID
appears *inside the scene* shortly after another vehicle's track vanished *inside the scene*, and
the newcomer is where the lost vehicle should be by now (same kind of vehicle, similar size,
position predicted from its velocity), the newcomer is given the lost vehicle's ID.

It is conservative on purpose (tracks that end or start at the frame border are never merged) but
a wrong merge is still possible in dense queues, so it is OFF by default (``tracker.stitch.enabled``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from .detector import Detection

log = logging.getLogger(__name__)

DEFAULT_GROUPS = (("car", "truck", "bus", "rickshaw", "van", "pickup", "loader"), ("motorcycle", "bicycle"))


@dataclass
class _Canon:
    """State of one real vehicle (one canonical ID), possibly spanning several raw IDs."""

    canon_id: int
    raw_id: int
    class_name: str
    last_frame: int
    box: tuple[float, float, float, float]
    vx: float = 0.0
    vy: float = 0.0


def _center(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _size(box: Sequence[float]) -> tuple[float, float]:
    return max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])


class TrackStitcher:
    """Maps raw tracker IDs to stable canonical IDs.

    Args:
        params: The ``tracker.stitch`` config section.
        frame_size: ``(width, height)`` of the processed frames (to recognise the frame border).
    """

    def __init__(self, params: Optional[dict] = None, frame_size: tuple[int, int] = (1920, 1080)) -> None:
        p = params or {}
        self.max_gap = int(p.get("max_gap_frames", 30))
        self.dist_factor = float(p.get("max_distance_factor", 1.2))   # x the larger side of the lost box
        self.size_ratio = float(p.get("max_size_ratio", 2.0))
        self.edge = float(p.get("edge_margin_frac", 0.03))
        groups = p.get("class_groups") or DEFAULT_GROUPS
        self._group = {str(c).lower(): i for i, g in enumerate(groups) for c in g}
        self.width, self.height = frame_size
        self._canon: dict[int, _Canon] = {}        # canonical id -> state
        self._map: dict[int, int] = {}             # raw id -> canonical id
        self.merges = 0

    # ------------------------------------------------------------------ public
    def apply(self, frame_no: int, dets: Sequence[Detection]) -> list[Detection]:
        """Return the detections with canonical IDs (untracked boxes pass through unchanged)."""
        live_raw = {d.track_id for d in dets if d.track_id is not None}
        new = [d for d in dets if d.track_id is not None and d.track_id not in self._map]
        self._match_new(frame_no, new, live_raw)

        out: list[Detection] = []
        for d in dets:
            if d.track_id is None:
                out.append(d)
                continue
            cid = self._map.setdefault(d.track_id, d.track_id)
            self._update(cid, d, frame_no)
            out.append(d if cid == d.track_id else replace(d, track_id=cid))
        self._prune(frame_no)
        return out

    # ---------------------------------------------------------------- internals
    def _near_edge(self, box: Sequence[float]) -> bool:
        mx, my = self.width * self.edge, self.height * self.edge
        return box[0] < mx or box[1] < my or box[2] > self.width - mx or box[3] > self.height - my

    def _compatible(self, a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return a == b or (a in self._group and b in self._group and self._group[a] == self._group[b])

    def _match_new(self, frame_no: int, new: list[Detection], live_raw: set[int]) -> None:
        """Pair brand-new raw IDs with lost canonical vehicles (global greedy by distance)."""
        lost = [c for c in self._canon.values()
                if c.raw_id not in live_raw and 0 < frame_no - c.last_frame <= self.max_gap and not self._near_edge(c.box)]
        if not lost or not new:
            return
        pairs: list[tuple[float, int, int]] = []
        for i, d in enumerate(new):
            if self._near_edge(d.box):
                continue                                    # entering from the border: a genuinely new vehicle
            dw, dh = _size(d.box)
            for j, c in enumerate(lost):
                if not self._compatible(d.class_name, c.class_name):
                    continue
                cw, ch = _size(c.box)
                ratio = (dw * dh) / (cw * ch)
                if not (1 / self.size_ratio <= ratio <= self.size_ratio):
                    continue
                gap = frame_no - c.last_frame
                cx, cy = _center(c.box)
                px, py = cx + c.vx * gap, cy + c.vy * gap   # where the lost vehicle should be now
                nx, ny = _center(d.box)
                dist = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5
                if dist <= self.dist_factor * max(cw, ch):
                    pairs.append((dist, i, j))
        used_new: set[int] = set()
        used_lost: set[int] = set()
        for dist, i, j in sorted(pairs):
            if i in used_new or j in used_lost:
                continue
            used_new.add(i)
            used_lost.add(j)
            self._map[new[i].track_id] = lost[j].canon_id          # type: ignore[index]
            self._canon[lost[j].canon_id].raw_id = new[i].track_id  # type: ignore[assignment]
            self.merges += 1
            log.debug("Stitched raw track %s into #%d (gap %d frames, %.0f px)", new[i].track_id,
                      lost[j].canon_id, frame_no - lost[j].last_frame, dist)

    def _update(self, cid: int, d: Detection, frame_no: int) -> None:
        c = self._canon.get(cid)
        if c is None:
            self._canon[cid] = _Canon(cid, d.track_id, d.class_name, frame_no, tuple(d.box))  # type: ignore[arg-type]
            return
        gap = max(1, frame_no - c.last_frame)
        (ox, oy), (nx, ny) = _center(c.box), _center(d.box)
        blend = 0.5 if c.vx or c.vy else 1.0                       # smooth the velocity estimate
        c.vx = blend * (nx - ox) / gap + (1 - blend) * c.vx
        c.vy = blend * (ny - oy) / gap + (1 - blend) * c.vy
        c.box, c.last_frame, c.class_name, c.raw_id = tuple(d.box), frame_no, d.class_name, d.track_id  # type: ignore[assignment]

    def _prune(self, frame_no: int) -> None:
        stale = [cid for cid, c in self._canon.items() if frame_no - c.last_frame > self.max_gap]
        for cid in stale:
            del self._canon[cid]
        if len(self._map) > 5000:                                   # raw ids of long-gone tracks
            live = {c.raw_id for c in self._canon.values()}
            self._map = {r: c for r, c in self._map.items() if r in live or c in self._canon}

    def reset(self) -> None:
        self._canon.clear()
        self._map.clear()
        self.merges = 0
