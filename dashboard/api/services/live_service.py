"""Runs the detection pipeline in a background thread and exposes its live state."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.stream import is_live_source, mask_source, parse_source

log = logging.getLogger(__name__)


def _has_gpu(cfg: Optional[dict] = None) -> bool:
    """A GPU the detector can use: CUDA, Apple, or an Intel GPU through OpenVINO (torch is imported lazily: it is heavy)."""
    try:
        import torch
        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    if str(((cfg or {}).get("detector") or {}).get("backend", "auto")).lower() != "torch":
        from src.detector import openvino_gpu_available
        return openvino_gpu_available()
    return False


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SKIP_DIRS = {"node_modules", "dist", "evidence", "outputs", "models", "runs", "__pycache__", "venv"}   # not input videos


class RunError(ValueError):
    """The requested run is not allowed or cannot start."""


class LiveService:
    """Owns at most one running pipeline and the latest annotated frame."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        d = cfg.get("dashboard", {})
        self.min_interval = 1.0 / max(1, int(d.get("mjpeg_fps", 10)))
        self.max_width = max(320, int(d.get("mjpeg_max_width", 960)))
        self.quality = int(d.get("mjpeg_quality", 70))
        self.allowed_dirs = [Path(p).resolve() for p in d.get("allowed_video_dirs", ["."])]
        self.allow_live = bool(d.get("allow_live_sources", False))
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pipeline: Any = None
        self._overlay: Optional[tuple[np.ndarray, np.ndarray]] = None      # (mask, annotated frame) of the latest detection
        self._jpeg: Optional[bytes] = None
        self._last_encode = 0.0
        self.status: dict[str, Any] = {"state": "idle", "video": None, "error": "", "summary": None}
        self.stats: dict[str, Any] = {}

    # ----------------------------------------------------------------- queries
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self.status, "stats": dict(self.stats)}

    # ----------------------------------------------------------------- control
    def check_video(self, video: str) -> str:
        """Validate a source: a live camera (webcam index / stream URL) or a video file in an allowed folder."""
        source = parse_source(video)
        if is_live_source(source):
            if not self.allow_live:
                raise RunError("Live camera sources are disabled (set dashboard.allow_live_sources: true)")
            return str(source)
        video = str(video).strip().strip('"')
        p = Path(video).expanduser()
        try:
            resolved = p.resolve(strict=True)
        except OSError:
            resolved = self._find_by_name(video)     # e.g. just "clip.mp4" for videos/clip.mp4
        if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RunError(f"Not a supported video file: {video}")
        if not any(resolved.is_relative_to(d) for d in self.allowed_dirs):
            raise RunError("Video is outside dashboard.allowed_video_dirs")
        return str(resolved)

    def list_videos(self, max_depth: int = 3, limit: int = 300) -> list[Path]:
        """Video files inside the allowed folders (newest first); output/cache folders are skipped."""
        found: list[Path] = []
        for base in self.allowed_dirs:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                found += [Path(dirpath) / f for f in filenames if Path(f).suffix.lower() in VIDEO_EXTENSIONS]
        found = sorted(set(found), key=lambda q: q.stat().st_mtime, reverse=True)
        return found[:limit]

    def video_choices(self) -> list[str]:
        """The same list as paths relative to the project folder, for the UI picker."""
        cwd = Path.cwd().resolve()
        return [(q.relative_to(cwd) if q.is_relative_to(cwd) else q).as_posix() for q in self.list_videos()]

    def _find_by_name(self, video: str) -> Path:
        """Resolve a bare file name (or a path suffix) to a unique video in the allowed folders."""
        wanted = video.replace("\\", "/").lower()
        hits = [q for q in self.list_videos() if q.as_posix().lower().endswith("/" + wanted.lstrip("/")) or q.name.lower() == wanted]
        if len(hits) == 1:
            return hits[0].resolve()
        if len(hits) > 1:
            raise RunError(f"'{video}' matches {len(hits)} files; give the full path, e.g. {self.video_choices()[0]}")
        raise RunError(f"Video not found: {video}. Put it in the videos/ folder and pick it from the list.")

    def start(self, video: str, camera: Optional[str] = None, model: Optional[str] = None,
              tracker: Optional[str] = None, assume_signal: Optional[str] = None) -> None:
        """Start processing in the background. Raises ``RunError`` if busy or invalid."""
        path = self.check_video(video)
        if self.running:
            raise RunError("A video is already being processed; stop it first")
        cfg = {**self.cfg}
        # Files play at their own speed like a camera (frames are skipped, never slow motion) unless turned off.
        dcfg = cfg.get("dashboard", {})
        cfg["video"] = {**cfg.get("video", {}), "realtime": bool(dcfg.get("realtime_playback", True))}
        if cfg["video"].get("resize_width") is None and dcfg.get("resize_width"):
            cfg["video"]["resize_width"] = dcfg["resize_width"]
        if not dcfg.get("save_video", False):
            cfg["output"] = {**cfg.get("output", {}), "save_video": False}
        if model or tracker:
            cfg["detector"] = {**cfg["detector"], **({"model": model} if model else {})}
            cfg["tracker"] = {**cfg["tracker"], **({"name": tracker} if tracker else {})}
        if dcfg.get("cpu_imgsz") and str(cfg["detector"].get("device", "auto")) in ("auto", "cpu") and not _has_gpu(cfg):
            cfg["detector"] = {**cfg["detector"], "imgsz": min(int(cfg["detector"].get("imgsz", 960)), int(dcfg["cpu_imgsz"]))}
        with self._lock:
            self.status = {"state": "starting", "video": mask_source(path), "error": "", "summary": None}
            self.stats = {}
            self._jpeg = None
        self._thread = threading.Thread(target=self._run, args=(cfg, path, camera, assume_signal),
                                        name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop.set()

    def _run(self, cfg: dict, video: str, camera: Optional[str], assume_signal: Optional[str]) -> None:
        from main import Pipeline  # imported here: heavy (torch / ultralytics)

        try:
            threads = int(cfg.get("dashboard", {}).get("torch_threads") or 0)
            if threads > 0:                          # leave cores for the browser and the OS; this CPU has 4 physical ones
                import torch
                torch.set_num_threads(threads)
            self._pipeline = Pipeline(cfg, camera, force_signal=assume_signal)
            with self._lock:
                self.status["state"] = "running"
            self._overlay = None
            summary = self._pipeline.run(video, on_frame=self._on_frame, preview=self._on_camera_frame)
            from dataclasses import asdict
            with self._lock:
                self.status.update(state="finished", summary=asdict(summary))
        except Exception as exc:
            log.exception("Pipeline run failed")
            with self._lock:
                self.status.update(state="error", error=str(exc))
        finally:
            self._pipeline = None

    def _on_frame(self, frame: np.ndarray, stats: dict) -> None:
        """A processed (annotated) frame arrived: remember what was drawn and, for files, publish it."""
        now = time.monotonic()
        with self._lock:
            self.stats = stats
        raw = getattr(self._pipeline, "last_processed", None)
        if raw is not None and raw.shape == frame.shape:
            # Everything the detector drew (boxes, labels, lines, counters), kept to paint over fresher camera frames.
            self._overlay = (np.any(frame != raw, axis=2), frame)
        if self._overlay is not None and getattr(self._pipeline, "live_source", False):
            return                                   # a live camera is published by _on_camera_frame at the camera's pace
        if now - self._last_encode < self.min_interval:
            return
        self._last_encode = now
        self._publish(frame)

    def _on_camera_frame(self, raw: np.ndarray) -> None:
        """Every camera frame (reader thread): show it smoothly with the latest detections painted on top."""
        now = time.monotonic()
        overlay = self._overlay
        if overlay is None or now - self._last_encode < self.min_interval:
            return
        self._last_encode = now
        mask, annotated = overlay
        h, w = annotated.shape[:2]
        out = raw if raw.shape[:2] == (h, w) else cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
        out = out.copy()
        out[mask] = annotated[mask]
        self._publish(out)

    def _publish(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if max(h, w) > self.max_width:              # the browser decodes a smaller picture much faster (tall phone video too)
            s = self.max_width / max(h, w)
            frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()


def placeholder_jpeg(text: str = "No video running", size: tuple[int, int] = (960, 540)) -> bytes:
    """A dark placeholder frame for the MJPEG stream when nothing is running."""
    img = np.full((size[1], size[0], 3), 24, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, text, ((size[0] - tw) // 2, (size[1] + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (150, 150, 150), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()
