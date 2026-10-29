"""Benchmark detector models and trackers on the same video.

    python benchmark.py --video traffic.mp4 --models yolov8n.pt yolo26n.pt \\
        --trackers botsort bytetrack --max-frames 300

Runs every model x tracker combination through the full pipeline (no video, evidence
or CSV output) and prints a comparison table: FPS, inference time, detections,
unique track IDs, track-quality proxies and events found. True ID-switch counts need
ground-truth tracks, so ``short_tracks`` / ``mean_track_len`` are reported as proxies:
on the same video, fewer short tracks and longer tracks mean steadier IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from main import Pipeline
from src.detector import VehicleDetector
from src.logger import setup_logging
from src.tracker import VehicleTracker
from src.utils import ensure_dir, load_config, read_frame

log = logging.getLogger("benchmark")

TrackerFactory = Callable[[dict, dict], Any]  # (cfg, detector_cache) -> tracker


class TimedTracker:
    """Wraps a tracker and accumulates the time spent in detection + tracking."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0

    def track(self, frame):
        t0 = time.perf_counter()
        out = self.inner.track(frame)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out

    def reset(self) -> None:
        self.inner.reset()


def default_tracker_factory(cfg: dict, cache: dict) -> VehicleTracker:
    """Build a tracker, loading each model's weights only once per benchmark."""
    model = cfg["detector"]["model"]
    if model not in cache:
        cache[model] = VehicleDetector(cfg)
    return VehicleTracker(cache[model], cfg)


def run_benchmark(
    cfg: dict,
    video: str,
    models: list[str],
    trackers: list[str],
    max_frames: Optional[int] = None,
    camera: Optional[str] = None,
    assume_signal: Optional[str] = None,
    tracker_factory: TrackerFactory = default_tracker_factory,
) -> pd.DataFrame:
    """Run every model x tracker combination; one row per combination."""
    rows: list[dict] = []
    cache: dict = {}
    try:
        warm_frame = read_frame(video, 0)
    except (FileNotFoundError, ValueError):
        warm_frame = None
    for model in models:
        for trk in trackers:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["detector"]["model"] = model
            run_cfg["tracker"]["name"] = trk
            run_cfg["plate"]["enabled"] = False
            row: dict[str, Any] = {"model": model, "tracker": trk}
            try:
                timed = TimedTracker(tracker_factory(run_cfg, cache))
                if warm_frame is not None:                 # exclude one-off start-up cost
                    for _ in range(2):
                        timed.inner.track(warm_frame)
                    timed.inner.reset()
                pipe = Pipeline(run_cfg, camera, tracker=timed, plate_reader=None,
                                force_signal=assume_signal, write_outputs=False)
                s = pipe.run(video, max_frames=max_frames)
                row.update({
                    "fps": round(s.avg_fps, 1),
                    "infer_ms": round(1000 * timed.seconds / max(1, timed.calls), 1),
                    "frames": s.frames_processed,
                    "detections": s.detections,
                    "unique_ids": s.vehicles_tracked,
                    "mean_track_len": s.mean_track_length,
                    "short_tracks": s.short_tracks,
                    "crossings": s.crossings,
                    "violations": s.red_light_violations + s.amber_crossings,
                    "error": "",
                })
            except Exception as exc:  # one bad combination must not stop the benchmark
                log.exception("Benchmark failed for %s + %s", model, trk)
                row["error"] = str(exc)
            rows.append(row)
            log.info("%s + %s -> %s", model, trk, {k: v for k, v in row.items() if k not in ("model", "tracker")})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark detector models and trackers on one video")
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--models", nargs="+", help="default: benchmark.models in config")
    p.add_argument("--trackers", nargs="+", choices=["botsort", "bytetrack"], help="default: benchmark.trackers")
    p.add_argument("--max-frames", type=int, help="limit frames per run (default: benchmark.max_frames)")
    p.add_argument("--camera")
    p.add_argument("--device")
    p.add_argument("--assume-signal", choices=["red", "yellow", "green"], help="TESTING ONLY")
    p.add_argument("--output", help="JSON path (default: paths.benchmark_json)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, {"detector.device": args.device})
    setup_logging({"logging": {"level": "INFO"}})
    b = cfg.get("benchmark", {})
    models = args.models or b.get("models", ["yolov8n.pt", "yolo26n.pt"])
    trackers = args.trackers or b.get("trackers", ["botsort", "bytetrack"])
    max_frames = args.max_frames if args.max_frames is not None else b.get("max_frames")
    if not Path(args.video).is_file() and not str(args.video).startswith(("rtsp://", "http")):
        log.error("Video not found: %s", args.video)
        return 2
    log.info("Benchmarking %d model(s) x %d tracker(s) on %s", len(models), len(trackers), args.video)
    df = run_benchmark(cfg, args.video, models, trackers, max_frames, args.camera, args.assume_signal)
    print("\n" + df.to_string(index=False) + "\n")
    out = Path(args.output or cfg["paths"]["benchmark_json"])
    ensure_dir(out.parent)
    out.write_text(json.dumps({"video": str(args.video), "max_frames": max_frames,
                               "results": json.loads(df.to_json(orient="records"))}, indent=2), encoding="utf-8")
    log.info("Results written to %s", out)
    return 0 if (df["error"] == "").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
