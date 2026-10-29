"""Frame sources: video files and live cameras (webcam index, RTSP / HTTP streams).

``open_source(video)`` returns a ``FrameSource`` whose ``read()`` yields ``(frame, index, seconds)``.

* Files: frames come one by one; ``seconds = index / fps`` (video time).
* Live sources: a background thread keeps decoding and only the *latest* frame is kept, so a
  slow detector skips frames instead of falling further and further behind the camera.
  ``seconds`` is real elapsed time since the first frame, which keeps speed estimates correct
  even when frames are skipped. Dropped connections are re-opened automatically.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

Source = Union[int, str]
LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def mask_source(source: Source) -> str:
    """Printable form of a source with any ``user:password@`` in a URL hidden (safe for logs and the UI)."""
    return re.sub(r"(//[^/:@\s]+):[^@/\s]+@", r"\1:****@", str(source))


def parse_source(video: Union[int, str]) -> Source:
    """``"0"`` / ``0`` -> webcam index 0; anything else stays a path or URL."""
    if isinstance(video, int):
        return video
    text = str(video).strip().strip('"')
    return int(text) if text.isdigit() else text


def is_live_source(source: Source) -> bool:
    """True for webcam indices and network streams."""
    return isinstance(source, int) or str(source).lower().startswith(LIVE_PREFIXES)


def _open_capture(source: Source) -> cv2.VideoCapture:
    """Open a capture with sensible per-source options (TCP for RTSP, DirectShow for Windows webcams)."""
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)       # more reliable than the default backend on Windows
        if cap.isOpened():
            return cap
        cap.release()
    elif isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")    # UDP drops packets on Wi-Fi
    return cv2.VideoCapture(source)


class FrameSource(ABC):
    """Common interface for files and live cameras."""

    live = False
    fps = 25.0
    total = 0
    # Called with every decoded frame, on the reader thread (live sources only). Keep it quick; it lets the
    # dashboard show the camera smoothly while detection runs at a slower rate.
    listener: Optional[Callable[[np.ndarray], None]] = None

    @abstractmethod
    def read(self, stop: Optional[threading.Event] = None) -> Optional[tuple[np.ndarray, int, float]]:
        """Next ``(frame, frame_index, seconds)``; ``None`` at the end of the source or when stopped."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture."""


def _valid_fps(value: float) -> float:
    return value if value and 1 < value < 240 else 25.0


class FileSource(FrameSource):
    """A video file read sequentially; ``realtime`` plays it at its own speed like a live camera.

    With ``realtime`` the file is paced by the wall clock: a detector that is slower than the video
    skips the frames it cannot keep up with (the clip never turns into slow motion), a fast one waits.
    ``seconds`` stays the video time (``index / fps``), so speed estimates are unaffected by skipping.
    """

    MAX_CATCH_UP_S = 15.0     # after a long stall, jump at most this far ahead instead of decoding it all

    def __init__(self, path: str, realtime: bool = False) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {path}")
        self.fps = _valid_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = -1
        self.realtime = realtime
        self._t0: Optional[float] = None
        self._anchored = False
        self.skipped = 0
        self.path = str(path)
        self._closing = False
        self._preview_thread: Optional[threading.Thread] = None

    def start_preview(self) -> None:
        """Show the file smoothly while the detector skips frames.

        A second reader plays the file by the same clock and hands every frame to ``listener``; the
        detector's frames keep their own (slower) path. Only meaningful with ``realtime``.
        """
        if not self.realtime or self.listener is None or self._preview_thread is not None:
            return
        self._preview_thread = threading.Thread(target=self._play_preview, name="file-preview", daemon=True)
        self._preview_thread.start()

    def _play_preview(self) -> None:
        cap = cv2.VideoCapture(self.path)
        shown = -1
        try:
            while not self._closing and self.listener is not None:
                if not self._anchored or self._t0 is None:          # the clock starts after the first processed frame
                    time.sleep(0.01)
                    continue
                due = int((time.monotonic() - self._t0) * self.fps)
                if due <= shown:
                    time.sleep(0.005)
                    continue
                while shown < due - 1:                                # drop frames that are already in the past
                    if not cap.grab():
                        return
                    shown += 1
                ok, frame = cap.read()
                if not ok:
                    return
                shown += 1
                try:
                    self.listener(frame)
                except Exception:
                    log.exception("Frame listener failed; disabling it")
                    self.listener = None
        finally:
            cap.release()

    def _pace(self, stop: Optional[threading.Event]) -> bool:
        """Wait for (or skip ahead to) the next frame's due time. False when stopped."""
        now = time.monotonic()
        if self._t0 is None:
            return True                              # first frame: no pacing yet
        if not self._anchored:
            # The first frame was just processed; its slow start (model warm-up) must not count as lag,
            # so the clock starts now, as if that frame had been shown on time.
            self._t0, self._anchored = now - (self._index + 1) / self.fps, True
        wait = self._t0 + (self._index + 1) / self.fps - now
        while wait > 0:
            if stop is not None and stop.is_set():
                return False
            time.sleep(min(wait, 0.05))
            wait = self._t0 + (self._index + 1) / self.fps - time.monotonic()
        late = min(int(-wait * self.fps), int(self.MAX_CATCH_UP_S * self.fps))
        for _ in range(max(0, late)):                # decode-and-drop the frames that are already in the past
            if not self.cap.grab():
                break
            self._index += 1
            self.skipped += 1
        return True

    def read(self, stop: Optional[threading.Event] = None) -> Optional[tuple[np.ndarray, int, float]]:
        if stop is not None and stop.is_set():
            return None
        if self.realtime and not self._pace(stop):
            return None
        if self.realtime and self._t0 is None:
            self._t0 = time.monotonic()              # placeholder; re-anchored on the next read (see _pace)
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        self._index += 1
        return frame, self._index, self._index / self.fps

    def close(self) -> None:
        self._closing = True
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)
        self.cap.release()


class LiveSource(FrameSource):
    """A webcam or network stream decoded on a background thread; ``read`` returns the newest frame."""

    live = True

    def __init__(self, source: Source, reconnect_attempts: int = 5, reconnect_delay_s: float = 2.0,
                 first_frame_timeout_s: float = 15.0) -> None:
        self.source = source
        self.attempts = reconnect_attempts
        self.delay = reconnect_delay_s
        self._cap = _open_capture(source)
        if not self._cap.isOpened():
            self._cap.release()
            hint = " (is another app using the webcam?)" if isinstance(source, int) else " (check the URL, network and credentials)"
            raise FileNotFoundError(f"Cannot open live source {mask_source(source)!r}{hint}")
        self.fps = _valid_fps(self._cap.get(cv2.CAP_PROP_FPS))
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._index = 0                    # frames decoded so far (the newest frame's index + 1)
        self._returned = 0
        self._t_first: Optional[float] = None
        self._t_last = 0.0
        self._ended = False
        self._closing = False
        self._first_timeout = first_frame_timeout_s
        self._thread = threading.Thread(target=self._pump, name="live-reader", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        failures = 0
        reconnects = 0
        while not self._closing:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size:
                failures, reconnects = 0, 0
                now = time.monotonic()
                with self._cond:
                    self._frame, self._index = frame, self._index + 1
                    self._t_first = self._t_first if self._t_first is not None else now
                    self._t_last = now
                    self._cond.notify_all()
                if self.listener is not None:
                    try:
                        self.listener(frame)
                    except Exception:                      # a broken preview must never stop the camera
                        log.exception("Frame listener failed; disabling it")
                        self.listener = None
                continue
            failures += 1
            if failures < 3:                       # a single bad read is normal on live streams
                time.sleep(0.05)
                continue
            if reconnects >= self.attempts:
                log.error("Live source %r lost and could not be reopened; stopping", mask_source(self.source))
                break
            reconnects += 1
            log.warning("Live source %r read failed; reconnecting (%d/%d)", mask_source(self.source), reconnects, self.attempts)
            self._cap.release()
            time.sleep(self.delay)
            self._cap = _open_capture(self.source)
            failures = 0
        with self._cond:
            self._ended = True
            self._cond.notify_all()

    @property
    def camera_fps(self) -> float:
        """Frames per second the camera is actually delivering (not what it claims)."""
        with self._cond:
            span = self._t_last - (self._t_first or self._t_last)
            return self._index / span if span > 1.0 else self.fps

    def read(self, stop: Optional[threading.Event] = None) -> Optional[tuple[np.ndarray, int, float]]:
        waited = 0.0
        with self._cond:
            while self._index <= self._returned:          # wait for a frame we have not returned yet
                if self._ended or (stop is not None and stop.is_set()):
                    return None
                if self._returned == 0 and waited > self._first_timeout:
                    log.error("No frame received from %r within %.0fs", mask_source(self.source), self._first_timeout)
                    return None
                self._cond.wait(0.2)
                waited += 0.2
            frame, index, t = self._frame, self._index, self._t_last
            skipped = index - self._returned - 1
            self._returned = index
            t0 = self._t_first if self._t_first is not None else t
        if skipped > 0:
            log.debug("Skipped %d frame(s) to stay real-time", skipped)
        assert frame is not None
        return frame, index - 1, t - t0

    def close(self) -> None:
        self._closing = True
        self._thread.join(timeout=2.0)
        self._cap.release()


def open_source(video: Union[int, str], reconnect_attempts: int = 5, reconnect_delay_s: float = 2.0,
                realtime: bool = False) -> FrameSource:
    """Open a file, webcam index, or stream URL as a ``FrameSource`` (``realtime``: pace files like a camera).

    Raises:
        FileNotFoundError: If the source cannot be opened.
    """
    source = parse_source(video)
    if is_live_source(source):
        return LiveSource(source, reconnect_attempts, reconnect_delay_s)
    return FileSource(str(source), realtime=realtime)
