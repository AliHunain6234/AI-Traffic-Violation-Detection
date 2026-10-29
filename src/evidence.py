"""Evidence packages for flagged events.

Per event, ``evidence/<event_id>/`` receives:
  before.jpg / crossing.jpg / after.jpg   annotated frames (box, ID, line, signal, time)
  vehicle.jpg                              sharpest crop of the vehicle
  plate.jpg, plate_enhanced.jpg            only if a plate was found
  metadata.json                            event + plate details
  manifest.json                            SHA-256 of every file above

Frames are buffered as JPEG so memory stays bounded at any resolution. The "after"
frame arrives ``evidence.frames_after`` frames later, so finished packages are
written on a background thread (plate OCR can be slow) and collected via
``process_frame`` / ``flush``.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence

import cv2
import numpy as np

from .plate_reader import PlateReader, PlateResult
from .utils import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    draw_label,
    draw_line,
    draw_polygon,
    ensure_dir,
    format_timestamp,
    get_approaches,
    is_valid_frame,
    pad_box,
    safe_crop,
    sharpness,
    sha256_file,
    wall_clock,
)

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
EVENT_COLORS = {
    "red_light_violation": COLOR_RED,
    "amber_crossing": COLOR_YELLOW,
    "legal_crossing": COLOR_GREEN,
    "overspeed": (200, 70, 190),
}
MIN_CROP_HEIGHT_PX = 24      # ignore tiny boxes when collecting crops
BEST_CROPS_PER_EVENT = 3     # crops handed to the plate reader


@dataclass
class EvidenceRequest:
    """Everything known about an event when it is detected."""

    event_id: str
    track_id: int
    vehicle_class: str
    event_type: str
    signal_state: str
    frame_no: int
    video_timestamp: str
    wall_clock_time: str
    confidence: float
    camera_id: str = ""
    speed_kmh: Optional[float] = None      # set for overspeed events
    limit_kmh: Optional[float] = None
    approach: str = ""                     # which stop line / signal pair produced the event


@dataclass
class EvidenceResult:
    """A finished evidence package."""

    request: EvidenceRequest
    path: str = ""
    plate_text: str = ""
    plate_conf: float = 0.0
    files: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class _Buffered:
    frame_no: int
    seconds: float
    jpg: bytes
    signal_state: str


@dataclass
class _Pending:
    request: EvidenceRequest
    box: Box
    crossing_frame: np.ndarray
    crossing_seconds: float
    before: Optional[_Buffered]
    before_box: Optional[Box]
    crops: list[np.ndarray]
    due_frame: int


def _jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


class EvidenceManager:
    """Buffers frames and builds evidence packages for events.

    Per-frame call order in the main loop::

        results += evidence.process_frame(frame_no, raw_frame, tracks, seconds, signal_state)
        for ev in crossings:  evidence.start_event(request, frame_no, raw_frame, box, seconds)
        ...
        results += evidence.flush()   # after the last frame

    Args:
        cfg: Full config (uses ``evidence``).
        camera: Camera block (stop line / zones are drawn on frames).
        plate_reader: Optional plate reader; ``None`` skips plate reading.
        start_time: Parsed ``video.start_time`` for wall-clock overlays.
    """

    def __init__(
        self,
        cfg: dict,
        camera: dict,
        plate_reader: Optional[PlateReader] = None,
        start_time: Optional[datetime] = None,
    ) -> None:
        ev = cfg.get("evidence", {})
        self.root = ensure_dir(cfg["paths"]["evidence_dir"])
        self.buffer_frames = max(1, int(ev.get("buffer_frames", 60)))
        self.frames_before = max(0, int(ev.get("frames_before", 15)))
        self.frames_after = max(0, int(ev.get("frames_after", 15)))
        self.quality = int(ev.get("jpeg_quality", 92))
        self.crop_pad = float(ev.get("vehicle_crop_padding", 0.10))
        self.crop_history = max(1, int(ev.get("best_crop_history", 30)))
        self.overlay = ev.get("overlay", {})
        self.font_scale = float(self.overlay.get("font_scale", 0.6))
        self.write_manifest = bool(ev.get("write_manifest", True))
        self.camera = camera
        self.plate_reader = plate_reader
        self.start_time = start_time

        self._buffer: Deque[_Buffered] = deque(maxlen=self.buffer_frames)
        self._boxes: dict[int, Deque[tuple[int, Box]]] = {}
        self._crops: dict[int, Deque[tuple[float, np.ndarray]]] = {}
        self._last_seen: dict[int, int] = {}
        self._pending: list[_Pending] = []
        self._futures: list[Future] = []
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence")
        self._last_frame: Optional[np.ndarray] = None
        self._last_seconds = 0.0
        self._last_signal = "unknown"

    # ------------------------------------------------------------ per frame
    def process_frame(
        self,
        frame_no: int,
        frame: Optional[np.ndarray],
        tracks: Iterable[tuple[int, Sequence[float]]],
        seconds: float,
        signal_state: str,
    ) -> list[EvidenceResult]:
        """Buffer the frame, track boxes/crops, and finish any due packages.

        ``frame`` must be the *raw* (unannotated) frame. Returns packages completed
        since the last call. Invalid frames are skipped.
        """
        tracks = list(tracks)
        if is_valid_frame(frame):
            assert frame is not None
            self._last_frame, self._last_seconds, self._last_signal = frame, seconds, signal_state
            self._buffer.append(_Buffered(frame_no, seconds, _jpeg(frame, self.quality), signal_state))
            self._observe(frame_no, frame, tracks)
            current = {tid: tuple(box[:4]) for tid, box in tracks}
            for p in [p for p in self._pending if p.due_frame <= frame_no]:
                self._pending.remove(p)
                box = current.get(p.request.track_id, p.box)
                self._submit(p, frame.copy(), seconds, signal_state, box, frame_no)
        self._prune(frame_no)
        return self._collect()

    def _observe(self, frame_no: int, frame: np.ndarray, tracks: list) -> None:
        for tid, box in tracks:
            box4 = tuple(float(v) for v in box[:4])
            self._last_seen[tid] = frame_no
            self._boxes.setdefault(tid, deque(maxlen=self.buffer_frames)).append((frame_no, box4))  # type: ignore[arg-type]
            if box4[3] - box4[1] < MIN_CROP_HEIGHT_PX:
                continue
            crop = safe_crop(frame, pad_box(box4, self.crop_pad, frame.shape[1], frame.shape[0]))
            if crop is not None:
                score = sharpness(crop) * float(np.sqrt(crop.shape[0] * crop.shape[1]))
                self._crops.setdefault(tid, deque(maxlen=self.crop_history)).append((score, crop))

    def _prune(self, frame_no: int) -> None:
        keep_for = self.buffer_frames + self.frames_after + 30
        wanted = {p.request.track_id for p in self._pending}
        for tid in [t for t, f in self._last_seen.items() if frame_no - f > keep_for and t not in wanted]:
            for store in (self._last_seen, self._boxes, self._crops):
                store.pop(tid, None)

    # ----------------------------------------------------------- start event
    def start_event(
        self,
        request: EvidenceRequest,
        frame_no: int,
        frame: np.ndarray,
        box: Sequence[float],
        seconds: float,
    ) -> None:
        """Begin an evidence package at the crossing frame (call after ``process_frame``)."""
        box4 = tuple(float(v) for v in box[:4])
        target = frame_no - self.frames_before
        before = None
        for b in self._buffer:
            if b.frame_no <= target:
                before = b
        if before is None and self._buffer:
            before = self._buffer[0]
        before_box = None
        for fn, bx in self._boxes.get(request.track_id, ()):
            if before is not None and fn <= before.frame_no:
                before_box = bx
        crops = [c for _, c in sorted(self._crops.get(request.track_id, ()), key=lambda s: s[0], reverse=True)]
        crossing_crop = safe_crop(frame, pad_box(box4, self.crop_pad, frame.shape[1], frame.shape[0]))
        if crossing_crop is not None:
            crops.append(crossing_crop)
        pending = _Pending(
            request=request,
            box=box4,  # type: ignore[arg-type]
            crossing_frame=frame.copy(),
            crossing_seconds=seconds,
            before=before,
            before_box=before_box,
            crops=[c.copy() for c in crops[: BEST_CROPS_PER_EVENT + 1]],
            due_frame=frame_no + self.frames_after,
        )
        if self.frames_after == 0:
            self._submit(pending, frame.copy(), seconds, request.signal_state, box4, frame_no)  # type: ignore[arg-type]
        else:
            self._pending.append(pending)

    # ---------------------------------------------------------------- finish
    def flush(self) -> list[EvidenceResult]:
        """Finish every pending package (video ended) and wait for the writer thread."""
        for p in list(self._pending):
            self._pending.remove(p)
            frame = self._last_frame if self._last_frame is not None else p.crossing_frame
            self._submit(p, frame.copy(), self._last_seconds, self._last_signal, p.box, p.request.frame_no)
        results = [f.result() for f in self._futures]
        self._futures.clear()
        return results

    def close(self) -> None:
        """Shut the writer thread down (after ``flush``)."""
        self._pool.shutdown(wait=True)

    def _submit(
        self, p: _Pending, after_frame: np.ndarray, after_seconds: float,
        after_signal: str, after_box: Box, after_frame_no: int,
    ) -> None:
        self._futures.append(
            self._pool.submit(self._write_package, p, after_frame, after_seconds, after_signal, after_box)
        )

    def _collect(self) -> list[EvidenceResult]:
        done = [f for f in self._futures if f.done()]
        for f in done:
            self._futures.remove(f)
        return [f.result() for f in done]

    # --------------------------------------------------------------- writing
    def _write_package(
        self, p: _Pending, after_frame: np.ndarray, after_seconds: float,
        after_signal: str, after_box: Box,
    ) -> EvidenceResult:
        """Runs on the writer thread. Never raises: errors are returned in the result."""
        req = p.request
        result = EvidenceResult(request=req)
        try:
            out = ensure_dir(self.root / req.event_id)
            result.path = str(out)

            before_frame, before_seconds, before_signal = p.crossing_frame, p.crossing_seconds, req.signal_state
            before_box: Optional[Box] = p.box
            if p.before is not None:
                decoded = cv2.imdecode(np.frombuffer(p.before.jpg, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    before_frame, before_seconds, before_signal = decoded, p.before.seconds, p.before.signal_state
                    before_box = p.before_box
            frames = {
                "before.jpg": (before_frame, before_box, before_seconds, before_signal),
                "crossing.jpg": (p.crossing_frame, p.box, p.crossing_seconds, req.signal_state),
                "after.jpg": (after_frame, after_box, after_seconds, after_signal),
            }
            for name, (img, box, secs, sig) in frames.items():
                self._save(out / name, self.annotate(img, box, req, secs, sig))
                result.files.append(name)

            vehicle = max(p.crops, key=sharpness) if p.crops else None
            if vehicle is not None:
                self._save(out / "vehicle.jpg", vehicle)
                result.files.append("vehicle.jpg")

            plate = PlateResult(note="plate reader not configured")
            if self.plate_reader is not None and p.crops:
                plate = self.plate_reader.read_best(p.crops, BEST_CROPS_PER_EVENT)
            if plate.crop is not None:
                self._save(out / "plate.jpg", plate.crop)
                result.files.append("plate.jpg")
            if plate.enhanced is not None:
                self._save(out / "plate_enhanced.jpg", plate.enhanced)
                result.files.append("plate_enhanced.jpg")
            result.plate_text, result.plate_conf = plate.text, plate.conf

            meta = {
                **asdict(req),
                "frames": {"before": p.before.frame_no if p.before else None,
                           "crossing": req.frame_no, "after": p.due_frame},
                "plate": {"text": plate.text, "conf": round(plate.conf, 4),
                          "format": plate.format_name, "note": plate.note},
                "files": list(result.files),
            }
            (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            result.files.append("metadata.json")

            if self.write_manifest:
                manifest = {name: sha256_file(out / name) for name in result.files}
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                result.files.append("manifest.json")
            log.info("Evidence saved: %s (plate=%r)", out, plate.text)
        except Exception as exc:
            log.exception("Failed to write evidence for %s", req.event_id)
            result.error = str(exc)
        return result

    def _save(self, path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
            raise OSError(f"Could not write {path}")

    # ------------------------------------------------------------ annotation
    def annotate(
        self, frame: np.ndarray, box: Optional[Sequence[float]], req: EvidenceRequest,
        seconds: float, signal_state: "str | dict[str, str]",
    ) -> np.ndarray:
        """Return a copy of ``frame`` with box, ID, line, signal state and timestamp drawn.

        ``signal_state`` may be a dict of approach -> state (cameras with several stop lines);
        the state of the event's own approach is shown.
        """
        img = frame.copy()
        if isinstance(signal_state, dict):
            signal_state = signal_state.get(req.approach) or next(iter(signal_state.values()), "unknown")
        color = EVENT_COLORS.get(req.event_type, COLOR_YELLOW)
        if self.overlay.get("draw_line", True):
            approaches = get_approaches(self.camera)
            for ap in approaches:
                if ap.get("stop_line"):
                    line = ap["stop_line"]
                    own = ap["name"] == req.approach or len(approaches) == 1
                    draw_line(img, tuple(line[0]), tuple(line[1]), thickness=3 if own else 1)
                    if len(approaches) > 1:
                        draw_label(img, ap["name"], (int(line[0][0]) + 4, int(line[0][1]) - 6), scale=0.5)
                for poly in ap.get("violation_zones") or []:
                    draw_polygon(img, [tuple(pt) for pt in poly])
        if box is not None and self.overlay.get("draw_box", True):
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            draw_label(img, f"#{req.track_id} {req.vehicle_class}", (x1, max(16, y1 - 4)),
                       bg=color, scale=self.font_scale)
        if self.overlay.get("draw_signal", True):
            draw_label(img, f"SIGNAL: {signal_state.upper()}", (10, 28),
                       bg=_signal_bg(signal_state), scale=self.font_scale + 0.1, thickness=2)
        if self.overlay.get("draw_timestamp", True):
            stamp = f"{req.camera_id or 'cam'} | video {format_timestamp(seconds)} | {wall_clock(self.start_time, seconds)}"
            draw_label(img, stamp, (10, img.shape[0] - 10), scale=self.font_scale)
        draw_label(img, req.event_type.replace("_", " ").upper(), (10, 58), bg=color, scale=self.font_scale)
        if req.speed_kmh is not None:
            limit = f" (limit {req.limit_kmh:.0f})" if req.limit_kmh is not None else ""
            draw_label(img, f"SPEED {req.speed_kmh:.0f} km/h{limit}", (10, 88), bg=color, scale=self.font_scale + 0.1, thickness=2)
        return img


def _signal_bg(state: str) -> tuple[int, int, int]:
    return {"red": COLOR_RED, "yellow": COLOR_YELLOW, "green": COLOR_GREEN}.get(state, (110, 110, 110))
