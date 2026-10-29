"""AI Traffic Violation Detection System: CLI entry point and pipeline.

    python main.py --video traffic.mp4 --config config.yaml --model yolo26n.pt \\
        --tracker botsort --output outputs/annotated.mp4

``Pipeline`` is importable (benchmark.py and the dashboard reuse it).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import cv2
import numpy as np

from src.detector import Detection, VehicleDetector
from src.evidence import EvidenceManager, EvidenceRequest, EvidenceResult
from src.line_crossing import CrossingEvent, LineCrossingDetector
from src.logger import EventIdFactory, EventLogger, EventRecord, StatsWriter, setup_logging
from src.plate_reader import PlateReader
from src.signal_classifier import SignalStateTracker
from src.stabilizer import CameraStabilizer
from src.speed import SpeedEstimator, SpeedMeasurement, build_speed_estimator, is_overspeed, limit_for
from src.stream import mask_source, open_source
from src.track_stitcher import TrackStitcher
from src.tracker import VehicleTracker
from src.utils import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    draw_label,
    draw_line,
    draw_polygon,
    ensure_dir,
    format_timestamp,
    get_approaches,
    get_camera,
    load_config,
    parse_start_time,
    resolve_device,
    wall_clock,
)

log = logging.getLogger("main")

EVENT_COLORS = {
    "red_light_violation": COLOR_RED,
    "amber_crossing": COLOR_YELLOW,
    "legal_crossing": COLOR_GREEN,
    "overspeed": (200, 70, 190),
}
DEFAULT_BOX_COLOR = (200, 140, 60)
BANNER_FRAMES = 45

FrameCallback = Callable[[np.ndarray, dict], None]


def _cpu_percent() -> Optional[float]:
    """Whole-machine CPU use since the previous call (None when psutil is missing)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


@dataclass
class _Approach:
    """One stop line with its own signal (a camera can have several)."""

    name: str
    line: list
    crossing: LineCrossingDetector
    signal: SignalStateTracker


@dataclass
class RunSummary:
    """What a pipeline run produced."""

    frames_read: int = 0
    frames_processed: int = 0
    seconds: float = 0.0
    avg_fps: float = 0.0
    vehicles_tracked: int = 0
    detections: int = 0
    crossings: int = 0
    red_light_violations: int = 0
    amber_crossings: int = 0
    legal_crossings: int = 0
    suppressed_unknown_signal: int = 0
    speed_measurements: int = 0
    overspeed_events: int = 0
    plates_read: int = 0
    events_csv: str = ""
    output_video: str = ""
    # Track-quality proxies (true ID switches need ground-truth tracks): fewer short
    # tracks and a longer mean track length mean steadier IDs on the same video.
    mean_track_length: float = 0.0
    short_tracks: int = 0
    track_merges: int = 0          # IDs healed by the stitcher (0 when stitching is off)


SHORT_TRACK_FRAMES = 15


def scale_camera(camera: dict, frame_w: int, frame_h: int, proc_w: int, proc_h: int) -> dict:
    """Return a copy of the camera block with coordinates mapped to the processed frame.

    Coordinates are stored in the frame size they were defined on (``frame_size``).
    If that is missing the video's own size is assumed.
    """
    ref_w, ref_h = camera.get("frame_size") or (frame_w, frame_h)
    sx, sy = proc_w / ref_w, proc_h / ref_h
    out = dict(camera)
    if camera.get("stop_line"):
        out["stop_line"] = [[x * sx, y * sy] for x, y in camera["stop_line"]]
    if camera.get("signal_roi"):
        x, y, w, h = camera["signal_roi"]
        out["signal_roi"] = [x * sx, y * sy, w * sx, h * sy]
    out["violation_zones"] = [[[x * sx, y * sy] for x, y in poly] for poly in camera.get("violation_zones") or []]
    if camera.get("approaches"):
        scaled = []
        for ap in camera["approaches"]:
            ap = dict(ap)
            if ap.get("stop_line"):
                ap["stop_line"] = [[x * sx, y * sy] for x, y in ap["stop_line"]]
            if ap.get("signal_roi"):
                x, y, w, h = ap["signal_roi"]
                ap["signal_roi"] = [x * sx, y * sy, w * sx, h * sy]
            ap["violation_zones"] = [[[x * sx, y * sy] for x, y in poly] for poly in ap.get("violation_zones") or []]
            scaled.append(ap)
        out["approaches"] = scaled
    cal = camera.get("speed")
    if cal and cal.get("method"):
        cal = dict(cal)   # world-space values (metres) are never scaled, only pixel coordinates
        if cal.get("lines"):
            cal["lines"] = [[[x * sx, y * sy] for x, y in line] for line in cal["lines"]]
        if cal.get("image_points"):
            cal["image_points"] = [[x * sx, y * sy] for x, y in cal["image_points"]]
        out["speed"] = cal
    return out


class Pipeline:
    """Detect, track, classify the signal, detect crossings, log events, save evidence.

    Args:
        cfg: Full configuration dict.
        camera_id: Camera/site to use (default ``camera.active``).
        detector / tracker / plate_reader: Optional injected components (tests, benchmark).
        force_signal: Testing only: treat the signal as this state on every frame.
    """

    def __init__(
        self,
        cfg: dict,
        camera_id: Optional[str] = None,
        detector: Optional[VehicleDetector] = None,
        tracker: Optional[VehicleTracker] = None,
        plate_reader: Optional[PlateReader] = None,
        force_signal: Optional[str] = None,
        write_outputs: bool = True,
    ) -> None:
        self.cfg = cfg
        self.camera = get_camera(cfg, camera_id)
        self.camera_id = self.camera["id"]
        self.force_signal = force_signal
        self.write_outputs = write_outputs
        if tracker is None:
            detector = detector or VehicleDetector(cfg)
            tracker = VehicleTracker(detector, cfg)
        self.tracker = tracker
        if plate_reader is None and cfg.get("plate", {}).get("enabled", True):
            plate_reader = PlateReader(cfg)
        self.plate_reader = plate_reader
        self.start_time = parse_start_time(cfg.get("video", {}).get("start_time"))
        ev_cfg = cfg.get("events", {})
        self.event_map: dict[str, str] = ev_cfg.get("mapping", {})
        self.flagged = set(ev_cfg.get("flagged", ["red_light_violation", "amber_crossing"]))
        self.evidence_for_legal = bool(ev_cfg.get("evidence_for_legal_crossing", False))
        self.stop = threading.Event()
        self.last_processed: Optional[np.ndarray] = None
        self.live_source = False

    # ------------------------------------------------------------------- run
    def run(
        self,
        video: str,
        output: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        on_frame: Optional[FrameCallback] = None,
        preview: Optional[Callable[[np.ndarray], None]] = None,
    ) -> RunSummary:
        """Process a video file or RTSP stream end to end.

        ``preview`` (live sources only) receives every camera frame, not just the processed ones;
        ``self.last_processed`` is the un-annotated frame that goes with the latest ``on_frame`` call.
        """
        cfg = self.cfg
        vcfg = cfg.get("video", {})
        ocfg = cfg.get("output", {})
        # A file, a webcam index ("0"), or an RTSP/HTTP stream URL. Live sources always process the newest frame.
        source = open_source(video, int(vcfg.get("rtsp_reconnect_attempts", 5)), float(vcfg.get("rtsp_reconnect_delay_s", 2.0)),
                             realtime=bool(vcfg.get("realtime", False)))
        is_stream = source.live
        smooth = is_stream or bool(getattr(source, "realtime", False))     # a camera, or a file played at its own speed
        self.live_source = smooth and preview is not None
        if preview is not None and smooth:
            source.listener = preview
            if hasattr(source, "start_preview"):
                source.start_preview()
        src_fps = source.fps
        total = source.total
        frame_skip = 0 if is_stream else max(0, int(vcfg.get("frame_skip", 0)))
        if is_stream:
            log.info("Live source %r opened (camera reports ~%.0f FPS); processing the newest frame each step", mask_source(video), src_fps)
        resize_w = vcfg.get("resize_width")

        self.tracker.reset()
        events_logger = EventLogger(cfg) if self.write_outputs else None
        stats = StatsWriter(cfg, self.camera_id, persist=self.write_outputs)
        stats.update(status="running", total_frames=total)
        ids = EventIdFactory()
        evidence: Optional[EvidenceManager] = None
        approaches: Optional[list[_Approach]] = None       # built on the first frame
        signal: Optional[SignalStateTracker] = None        # the first approach's signal (overlay / stats)
        trackers_by_roi: dict[Any, SignalStateTracker] = {}
        writer: Optional[cv2.VideoWriter] = None
        out_path = ""
        pending: dict[str, EventRecord] = {}
        status: dict[int, str] = {}                      # track_id -> last event_type
        trails: dict[int, Deque[tuple[int, int]]] = {}
        seen: set[int] = set()
        track_len: dict[int, int] = {}
        banner: Optional[tuple[str, int]] = None
        speed: Optional[SpeedEstimator] = None
        stitcher: Optional[TrackStitcher] = None
        stabilizer: Optional[CameraStabilizer] = None
        if (cfg.get("stabilization") or {}).get("enabled"):
            stabilizer = CameraStabilizer(cfg["stabilization"])
            log.info("Camera stabilization is ON")
        prev_boxes: list = []                            # last frame's vehicle boxes (kept out of the stabilizer's matching)
        speeds_live: dict[int, float] = {}               # track_id -> latest speed estimate (km/h)
        class_of: dict[int, str] = {}
        conf_of: dict[int, float] = {}
        last_box: dict[int, Any] = {}
        last_frame: Optional[np.ndarray] = None
        last_no, last_secs, last_sig = 0, 0.0, "unknown"
        summary = RunSummary(events_csv=str(events_logger.path) if events_logger else "")
        t_start = time.perf_counter()
        frame_no = -1

        def finish(results: list[EvidenceResult]) -> None:
            for res in results:
                rec = pending.pop(res.request.event_id, None)
                if rec is None:
                    continue
                rec.plate_text, rec.plate_conf = res.plate_text, res.plate_conf
                rec.evidence_path = res.path
                if events_logger:
                    events_logger.log(rec)
                stats.stats["plates_read"] += 1 if res.plate_text else 0
                summary.plates_read += 1 if res.plate_text else 0

        def process_speed(measurements: list[SpeedMeasurement], f_no: int, raw_frame: np.ndarray,
                          secs: float, sig_state: str) -> None:
            """Turn finished speed measurements into overspeed events (with evidence)."""
            nonlocal banner
            for m in measurements:
                cls = class_of.get(m.track_id)
                if cls is None:
                    continue
                rec = self._speed_event(m, cls, conf_of.get(m.track_id, 0.0), sig_state, ids, secs, summary, stats)
                if rec is None:
                    continue
                status[m.track_id] = "overspeed"
                banner = (f"OVERSPEED  #{rec.track_id} {rec.vehicle_class} {m.speed_kmh:.0f} km/h", BANNER_FRAMES)
                if evidence and "overspeed" in self.flagged and m.track_id in last_box:
                    pending[rec.event_id] = rec
                    evidence.start_event(self._request(rec), f_no, raw_frame, last_box[m.track_id], secs)
                elif events_logger:
                    events_logger.log(rec)

        perf = time.perf_counter
        timings: dict[str, float] = {}                   # smoothed per-frame costs in ms, shown on the dashboard

        def smooth(key: str, ms: float) -> None:
            timings[key] = ms if key not in timings else timings[key] * 0.8 + ms * 0.2

        try:
            while not self.stop.is_set():
                t_wait = perf()
                item = source.read(self.stop)
                if item is None:          # end of file, stream lost for good, or stopped
                    break
                t_got = perf()
                smooth("wait_ms", (t_got - t_wait) * 1000)
                frame, frame_no, seconds = item
                summary.frames_read += 1
                if frame_skip and frame_no % (frame_skip + 1):
                    continue
                if max_frames and summary.frames_processed >= max_frames:
                    break

                orig_h, orig_w = frame.shape[:2]
                if resize_w and orig_w != int(resize_w):
                    s = int(resize_w) / orig_w
                    frame = cv2.resize(frame, (int(resize_w), int(round(orig_h * s))), interpolation=cv2.INTER_AREA)
                if stabilizer is not None:
                    frame = stabilizer.stabilize(frame, prev_boxes)
                h, w = frame.shape[:2]

                if approaches is None:  # first frame: build everything that needs the frame size
                    cam = scale_camera(self.camera, orig_w, orig_h, w, h)
                    cam["id"] = self.camera_id
                    approaches = []
                    for ap in get_approaches(cam):
                        roi = ap.get("signal_roi")
                        over = ap.get("signal_overrides") or {}
                        key = (tuple(roi) if roi else None, tuple(sorted(over.items(), key=lambda kv: kv[0])) if over else ())
                        if key not in trackers_by_roi:      # approaches sharing a signal share its classifier
                            trackers_by_roi[key] = SignalStateTracker(cfg, roi, overrides=over or None)
                        approaches.append(_Approach(ap["name"], ap.get("stop_line"),
                                                    LineCrossingDetector.from_config(cfg, ap), trackers_by_roi[key]))
                        if not ap.get("stop_line") and not ap.get("violation_zones"):
                            log.warning("Approach '%s': no stop line / zone: no crossings will be detected (run tools/define_line.py)", ap["name"])
                        if not roi and not self.force_signal:
                            log.warning("Approach '%s': no signal ROI: crossings will be suppressed (run tools/define_signal_roi.py)", ap["name"])
                    signal = approaches[0].signal
                    stitch_cfg = cfg.get("tracker", {}).get("stitch") or {}
                    if stitch_cfg.get("enabled"):
                        stitcher = TrackStitcher(stitch_cfg, (w, h))
                        log.info("ID stitching is ON")
                    self._cam, self._approaches = cam, approaches
                    log.info("Approaches: %s", ", ".join(a.name for a in approaches))
                    if self.write_outputs:
                        evidence = EvidenceManager(cfg, cam, self.plate_reader, self.start_time)
                        if ocfg.get("save_video", True):
                            out_path = output or str(Path(cfg["paths"]["output_dir"]) / ocfg.get("video_name", "annotated.mp4"))
                            ensure_dir(Path(out_path).parent)
                            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*ocfg.get("codec", "mp4v")),
                                                     float(ocfg.get("live_fps", 10)) if is_stream else src_fps / (frame_skip + 1), (w, h))
                            if not writer.isOpened():
                                log.error("Cannot open video writer for %s; continuing without video", out_path)
                                writer = None
                    try:
                        speed = build_speed_estimator(cfg, cam)
                    except ValueError as exc:
                        log.error("Invalid speed calibration (%s); speed measurement is disabled", exc)
                    if speed is not None:
                        log.info("Speed measurement on (%s calibration)", speed.method)
                assert approaches is not None and signal is not None

                t_det = perf()
                dets = self.tracker.track(frame)
                smooth("detect_ms", (perf() - t_det) * 1000)
                if stitcher is not None:
                    dets = stitcher.apply(frame_no, dets)
                prev_boxes = [d.box for d in dets]
                summary.detections += len(dets)
                tracked = [d for d in dets if d.track_id is not None]
                by_id: dict[int, Detection] = {d.track_id: d for d in tracked}  # type: ignore[misc]
                seen |= set(by_id)
                for tid in by_id:
                    track_len[tid] = track_len.get(tid, 0) + 1
                tracks = [(d.track_id, d.box) for d in tracked]

                for sig in trackers_by_roi.values():
                    sig.update(frame_no, frame, ignore_boxes=[d.box for d in dets])
                    if self.force_signal:
                        sig.force(frame_no, self.force_signal)
                signals_now = {ap.name: ap.signal.current.state for ap in approaches}

                raw = frame
                if evidence:
                    # one signal: a plain state; several: the state of each approach (evidence picks the right one)
                    finish(evidence.process_frame(frame_no, raw, tracks, seconds,
                                                  signals_now if len(approaches) > 1 else signal.current.state))

                for ap in approaches:
                    for ev in ap.crossing.update(frame_no, tracks):
                        rec = self._handle_crossing(ev, by_id, ap, ids, seconds, summary, stats)
                        if rec is None:
                            continue
                        status[ev.track_id] = rec.event_type
                        banner = (f"{rec.event_type.replace('_', ' ').upper()}  #{rec.track_id} {rec.vehicle_class}"
                                  + (f" [{ap.name}]" if len(approaches) > 1 else ""), BANNER_FRAMES)
                        flagged = rec.event_type in self.flagged or (rec.event_type == "legal_crossing" and self.evidence_for_legal)
                        if flagged and evidence:
                            pending[rec.event_id] = rec
                            evidence.start_event(self._request(rec), frame_no, raw, by_id[ev.track_id].box, seconds)
                        elif events_logger:
                            events_logger.log(rec)

                for d in tracked:
                    class_of[d.track_id], conf_of[d.track_id], last_box[d.track_id] = d.class_name, d.conf, d.box  # type: ignore[index]
                last_frame, last_no, last_secs, last_sig = raw, frame_no, seconds, signal.current.state
                if speed is not None:
                    speeds_live = speed.update(frame_no, seconds, tracks)
                    process_speed(speed.pop_measurements(), frame_no, raw, seconds, signal.current.state)

                summary.frames_processed += 1
                fps = stats.tick()
                stats.update(
                    frame_no=frame_no, video_timestamp=format_timestamp(seconds), signal_state=signal.current.state,
                    signals=signals_now, vehicles_tracked=len(seen), active_tracks=len(tracked),
                )
                annotated = self._annotate(frame, tracked, status, trails, stats.stats, banner, ocfg, speeds_live)
                if banner:
                    banner = (banner[0], banner[1] - 1) if banner[1] > 1 else None
                if writer is not None:
                    writer.write(annotated)
                smooth("busy_ms", (perf() - t_got) * 1000)
                stats.update(timings_ms={k: round(v) for k, v in timings.items()},
                             camera_fps=round(getattr(source, "camera_fps", source.fps), 1), cpu_percent=_cpu_percent())
                if on_frame:
                    self.last_processed = frame
                    on_frame(annotated, dict(stats.stats))
                if show:
                    cv2.imshow("Traffic Violation Detection", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if summary.frames_processed % 100 == 0:
                    total_txt = f"/{total}" if total else ""
                    log.info("Frame %d%s | %.1f FPS | tracked %d | violations %d",
                             frame_no, total_txt, fps, len(seen), stats.stats["red_light_violations"])
        except KeyboardInterrupt:
            log.warning("Interrupted; finishing evidence packages")
        finally:
            source.close()
            if writer is not None:
                writer.release()
            if speed is not None and last_frame is not None:
                process_speed(speed.flush(last_no, last_secs), last_no, last_frame, last_secs, last_sig)
            if evidence:
                finish(evidence.flush())
                evidence.close()
            for rec in pending.values():  # evidence writing failed: still log the event
                if events_logger:
                    events_logger.log(rec)
            if show:
                cv2.destroyAllWindows()
            stats.update(status="finished")
            if self.write_outputs:
                stats.write()

        summary.seconds = time.perf_counter() - t_start
        summary.avg_fps = summary.frames_processed / summary.seconds if summary.seconds else 0.0
        summary.vehicles_tracked = len(seen)
        summary.track_merges = stitcher.merges if stitcher is not None else 0
        summary.mean_track_length = round(sum(track_len.values()) / len(track_len), 1) if track_len else 0.0
        summary.short_tracks = sum(1 for n in track_len.values() if n < SHORT_TRACK_FRAMES)
        summary.output_video = out_path
        log.info("Done: %d frames in %.1fs (%.1f FPS) | tracked %d | red-light %d | amber %d | legal %d | "
                 "suppressed %d | speed measured %d | overspeed %d",
                 summary.frames_processed, summary.seconds, summary.avg_fps, summary.vehicles_tracked,
                 summary.red_light_violations, summary.amber_crossings, summary.legal_crossings,
                 summary.suppressed_unknown_signal, summary.speed_measurements, summary.overspeed_events)
        return summary

    # --------------------------------------------------------------- helpers
    def _handle_crossing(
        self, ev: CrossingEvent, by_id: dict[int, Detection], ap: _Approach,
        ids: EventIdFactory, seconds: float, summary: RunSummary, stats: StatsWriter,
    ) -> Optional[EventRecord]:
        """Classify one crossing by the state of THIS approach's signal at that moment; ``None`` if suppressed."""
        raw_state = self.force_signal or ap.signal.state_for_crossing(ev.frame_no)
        state = ap.signal.resolve(raw_state)
        det = by_id[ev.track_id]
        if state is None:
            summary.suppressed_unknown_signal += 1
            log.info("Crossing by #%d on '%s' suppressed: signal state unknown", ev.track_id, ap.name)
            return None
        event_type = self.event_map.get(state, "legal_crossing")
        summary.crossings += 1
        attr = {"red_light_violation": "red_light_violations", "amber_crossing": "amber_crossings",
                "legal_crossing": "legal_crossings"}.get(event_type)
        if attr:
            setattr(summary, attr, getattr(summary, attr) + 1)
        stats.count_event(event_type)
        return EventRecord(
            event_id=ids.next(), track_id=ev.track_id, vehicle_class=det.class_name, event_type=event_type,
            signal_state=state, frame_no=ev.frame_no, video_timestamp=format_timestamp(seconds),
            wall_clock_time=wall_clock(self.start_time, seconds), confidence=det.conf, camera_id=self.camera_id,
            extra={"approach": ap.name},
        )

    def _speed_event(
        self, m: SpeedMeasurement, class_name: str, conf: float, signal_state: str,
        ids: EventIdFactory, seconds: float, summary: RunSummary, stats: StatsWriter,
    ) -> Optional[EventRecord]:
        """Compare a finished measurement to the class limit; returns an event only if over it."""
        limit = limit_for(self.cfg, class_name)
        over = is_overspeed(self.cfg, class_name, m.speed_kmh)
        summary.speed_measurements += 1
        stats.count_speed(m.speed_kmh, over)
        log.info("#%d %s: %.1f km/h (limit %.0f)%s", m.track_id, class_name, m.speed_kmh, limit, " OVERSPEED" if over else "")
        if not over:
            return None
        summary.overspeed_events += 1
        return EventRecord(
            event_id=ids.next(), track_id=m.track_id, vehicle_class=class_name, event_type="overspeed",
            signal_state=signal_state, frame_no=m.frame_no, video_timestamp=format_timestamp(seconds),
            wall_clock_time=wall_clock(self.start_time, seconds), confidence=conf, camera_id=self.camera_id,
            extra={"speed_kmh": round(m.speed_kmh, 1), "limit_kmh": limit},
        )

    def _request(self, rec: EventRecord) -> EvidenceRequest:
        return EvidenceRequest(
            event_id=rec.event_id, track_id=rec.track_id, vehicle_class=rec.vehicle_class, event_type=rec.event_type,
            signal_state=rec.signal_state, frame_no=rec.frame_no, video_timestamp=rec.video_timestamp,
            wall_clock_time=rec.wall_clock_time, confidence=rec.confidence, camera_id=rec.camera_id,
            speed_kmh=rec.extra.get("speed_kmh"), limit_kmh=rec.extra.get("limit_kmh"),
            approach=str(rec.extra.get("approach", "")),
        )

    def _annotate(
        self, frame: np.ndarray, tracked: list[Detection], status: dict[int, str],
        trails: dict[int, Deque[tuple[int, int]]], stats: dict,
        banner: Optional[tuple[str, int]], ocfg: dict, speeds: Optional[dict[int, float]] = None,
    ) -> np.ndarray:
        img = frame.copy()
        speeds = speeds or {}
        multi = len(self._approaches) > 1
        drawn: set[int] = set()
        for ap in self._approaches:
            if ap.line:
                draw_line(img, tuple(ap.line[0]), tuple(ap.line[1]))
                if multi:
                    draw_label(img, ap.name, (int(ap.line[0][0]) + 4, int(ap.line[0][1]) - 6), scale=0.5)
            if id(ap.signal) not in drawn:       # approaches sharing one signal draw it once
                drawn.add(id(ap.signal))
                ap.signal.draw(img, name=ap.name if multi else "")
        for ap in self._approaches:
            for poly in ap.crossing.zones:
                draw_polygon(img, [tuple(p) for p in poly])
        trail_len = int(ocfg.get("trail_length", 30))
        for d in tracked:
            tid = d.track_id
            x1, y1, x2, y2 = (int(v) for v in d.box)
            color = EVENT_COLORS.get(status.get(tid, ""), DEFAULT_BOX_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            kmh = speeds.get(tid)
            tag = f"#{tid} {d.class_name}" + (f" {kmh:.0f} km/h" if kmh is not None else "")
            draw_label(img, tag, (x1, max(14, y1 - 3)), bg=color, scale=0.45)
            if ocfg.get("draw_trails", True):
                tr = trails.setdefault(tid, deque(maxlen=trail_len))
                tr.append((int((x1 + x2) / 2), y2))
                if len(tr) > 1:
                    cv2.polylines(img, [np.array(tr, np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        lines = [f"Tracked: {stats['vehicles_tracked']}   Crossings: {stats['crossings']}",
                 f"Red-light: {stats['red_light_violations']}   Amber: {stats['amber_crossings']}"
                 + (f"   Overspeed: {stats['overspeed_events']}" if self._cam.get("speed") else ""),
                 f"FPS: {stats['fps']:.1f}   {stats['video_timestamp']}"]
        for i, text in enumerate(lines):
            draw_label(img, text, (10, 22 + i * 24), scale=0.6)
        if banner:
            draw_label(img, banner[0], (10, img.shape[0] - 14), bg=(40, 40, 200), scale=0.8, thickness=2)
        return img


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Traffic Violation Detection System")
    p.add_argument("--video", help="video file or rtsp:// URL (default: video.source in config)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", help="detector weights, e.g. yolo26n.pt or yolov8n.pt")
    p.add_argument("--tracker", choices=["botsort", "bytetrack"], help="tracker to use")
    p.add_argument("--output", help="annotated output video path")
    p.add_argument("--show", action="store_true", help="show a live window (needs a display)")
    p.add_argument("--camera", help="camera/site id from config (default: camera.active)")
    p.add_argument("--max-frames", type=int, help="stop after N processed frames")
    p.add_argument("--device", help="auto | cpu | cuda | mps")
    p.add_argument("--conf", type=float, help="detection confidence threshold")
    p.add_argument("--no-plates", action="store_true", help="disable plate reading")
    p.add_argument("--resize-width", type=int, help="downscale frames to this width before detection (faster on CPU)")
    p.add_argument("--realtime", action="store_true", help="play a video file at its own speed, skipping frames if the detector is slower")
    p.add_argument("--stabilize", action="store_true", help="compensate camera shake (handheld / wobbling cameras)")
    p.add_argument("--assume-signal", choices=["red", "yellow", "green"],
                   help="TESTING ONLY: pretend the signal always has this state")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {
        "detector.model": args.model, "tracker.name": args.tracker, "video.source": args.video,
        "output.show_window": True if args.show else None, "detector.device": args.device,
        "detector.conf": args.conf, "plate.enabled": False if args.no_plates else None,
        "video.resize_width": args.resize_width, "video.realtime": True if args.realtime else None,"stabilization.enabled": True if args.stabilize else None,
    }
    try:
        cfg = load_config(args.config, overrides)
    except FileNotFoundError as exc:
        print(exc)  # logging is not configured yet
        return 2
    setup_logging(cfg)
    video = args.video if args.video is not None else cfg["video"].get("source")   # 0 (webcam) is a valid source
    if video is None or str(video).strip() == "":
        log.error("No video given: pass --video or set video.source in the config")
        return 2
    if args.assume_signal:
        log.warning("--assume-signal %s is for testing only; results are not real violations", args.assume_signal)
    log.info("Device: %s", resolve_device(cfg["detector"].get("device", "auto")))
    try:
        pipeline = Pipeline(cfg, args.camera, force_signal=args.assume_signal)
        summary = pipeline.run(str(video), output=args.output, show=bool(cfg["output"].get("show_window")),
                               max_frames=args.max_frames)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    log.info("Summary: %s", asdict(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
