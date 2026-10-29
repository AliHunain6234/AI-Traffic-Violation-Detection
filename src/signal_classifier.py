"""Traffic-signal state classification: per-frame classifier + temporal smoothing.

Two layers, so the colour classifier can be swapped for a CNN later without
touching the rest of the pipeline:

  * ``SignalClassifier`` (abstract): ``classify(roi_bgr) -> SignalReading``.
    ``HSVSignalClassifier`` is the default; ``CNNSignalClassifier`` is the plug-in point.
  * ``SignalStateTracker``: crops the ROI each frame, runs a classifier, smooths the
    result with a majority vote, and keeps a short history so the state *at the
    moment of a crossing* can be looked up with a lag correction.

An unreadable / occluded signal yields ``unknown``; by default that suppresses
violations rather than guessing.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

from .utils import SIGNAL_COLORS, draw_label, is_valid_frame, safe_crop

log = logging.getLogger(__name__)

RED, YELLOW, GREEN, UNKNOWN = "red", "yellow", "green", "unknown"
STATES = (RED, YELLOW, GREEN)


@dataclass(frozen=True)
class SignalReading:
    """Result of classifying one ROI image (or a smoothed state)."""

    state: str = UNKNOWN
    confidence: float = 0.0


class SignalClassifier(ABC):
    """Interface for anything that maps a traffic-light ROI image to a state."""

    @abstractmethod
    def classify(self, roi_bgr: np.ndarray) -> SignalReading:
        """Classify a BGR crop of the traffic light."""


class HSVSignalClassifier(SignalClassifier):
    """Colour-threshold classifier: counts bright red / yellow / green pixels.

    Args:
        cfg: The ``signal`` section of the config (HSV ranges, thresholds).
    """

    def __init__(self, cfg: dict) -> None:
        hsv = cfg.get("hsv", {})
        self.ranges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            name: [
                (np.array(r["lower"], dtype=np.uint8), np.array(r["upper"], dtype=np.uint8))
                for r in hsv.get(name, [])
            ]
            for name in STATES
        }
        missing = [n for n, r in self.ranges.items() if not r]
        if missing:
            raise ValueError(f"signal.hsv has no ranges for: {missing}")
        self.min_ratio = float(cfg.get("min_pixel_ratio", 0.02))
        self.min_conf = float(cfg.get("min_confidence", 0.30))
        k = int(cfg.get("morph_kernel", 3))
        self._kernel = np.ones((k, k), np.uint8) if k > 1 else None

    def pixel_ratios(self, roi_bgr: np.ndarray) -> dict[str, float]:
        """Fraction of ROI pixels falling in each colour range."""
        blurred = cv2.GaussianBlur(roi_bgr, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        area = float(hsv.shape[0] * hsv.shape[1])
        ratios: dict[str, float] = {}
        for name, ranges in self.ranges.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            if self._kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            ratios[name] = float(cv2.countNonZero(mask)) / area
        return ratios

    def classify(self, roi_bgr: np.ndarray) -> SignalReading:
        if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.ndim != 3:
            return SignalReading()
        ratios = self.pixel_ratios(roi_bgr)
        best = max(ratios, key=ratios.get)
        total = sum(ratios.values())
        if ratios[best] < self.min_ratio or total <= 0:
            return SignalReading()  # nothing lit: occluded / off / too dark
        conf = ratios[best] / total
        if conf < self.min_conf:
            return SignalReading(UNKNOWN, conf)
        return SignalReading(best, conf)


class CNNSignalClassifier(SignalClassifier):
    """Plug-in point for a small CNN (e.g. a 3-class MobileNet on ROI crops).

    To use: train a model on labelled ROI crops (red / yellow / green / off), then
    implement ``_load`` and ``classify`` below and set ``signal.classifier: cnn``
    plus ``signal.cnn_weights`` in config.yaml. The tracker needs nothing else.
    """

    def __init__(self, cfg: dict) -> None:
        self.weights = cfg.get("cnn_weights")
        if not self.weights:
            raise ValueError("signal.cnn_weights must be set to use the CNN classifier")
        self._load(self.weights)

    def _load(self, weights: str) -> None:
        raise NotImplementedError("CNN signal classifier is a placeholder; see class docstring")

    def classify(self, roi_bgr: np.ndarray) -> SignalReading:
        raise NotImplementedError("CNN signal classifier is a placeholder; see class docstring")


def build_classifier(cfg: dict) -> SignalClassifier:
    """Create the classifier named by ``signal.classifier`` in the config."""
    sig = cfg.get("signal", {})
    kind = str(sig.get("classifier", "hsv")).lower()
    if kind == "hsv":
        return HSVSignalClassifier(sig)
    if kind == "cnn":
        return CNNSignalClassifier(sig)
    raise ValueError(f"Unknown signal classifier: {kind!r} (expected 'hsv' or 'cnn')")


class SignalStateTracker:
    """Per-frame signal state with majority-vote smoothing and a lookup history.

    Args:
        cfg: Full config (uses the ``signal`` section).
        roi: ``[x, y, w, h]`` of the signal head in the *processed* frame, or
            ``None`` (every reading is then ``unknown``).
        classifier: Override the classifier (defaults to ``build_classifier(cfg)``).
        overrides: Per-camera / per-approach changes to the ``signal`` section (e.g. a lower
            ``min_pixel_ratio`` for a small, far-away lamp).

    Vehicles that overlap the ROI (a yellow taxi passing the lamp) are blacked out before
    classification, so they cannot vote ``yellow``; if they cover most of the ROI the vote is ``unknown``.
    """

    OCCLUDED_FRACTION = 0.85    # more than this much of the ROI hidden by vehicles -> unreadable

    def __init__(
        self,
        cfg: dict,
        roi: Optional[Sequence[float]] = None,
        classifier: Optional[SignalClassifier] = None,
        overrides: Optional[dict] = None,
    ) -> None:
        if overrides:
            cfg = {**cfg, "signal": {**cfg.get("signal", {}), **overrides}}
        sig = cfg.get("signal", {})
        self.mask_vehicles = bool(sig.get("mask_vehicles", True))
        self.roi = [int(v) for v in roi] if roi else None
        if self.roi is not None and (len(self.roi) != 4 or self.roi[2] <= 0 or self.roi[3] <= 0):
            raise ValueError(f"signal_roi must be [x, y, w, h] with positive size, got {roi}")
        self.classifier = classifier or build_classifier(cfg)
        self.window = max(1, int(sig.get("smoothing_window", 7)))
        self.lag = max(0, int(sig.get("lag_frames", 3)))
        self.unknown_policy = str(sig.get("unknown_policy", "suppress"))
        self._votes: Deque[SignalReading] = deque(maxlen=self.window)
        self._history: Deque[tuple[int, str]] = deque(maxlen=int(sig.get("history_frames", 300)))
        self.current = SignalReading()
        self.raw = SignalReading()
        if self.roi is None:
            log.warning("No signal ROI configured: signal state will always be 'unknown'")

    # ------------------------------------------------------------------ update
    def update(self, frame_no: int, frame: Optional[np.ndarray],
               ignore_boxes: Optional[Sequence[Sequence[float]]] = None) -> SignalReading:
        """Classify the ROI in ``frame`` and return the smoothed reading.

        Empty frames or a missing/out-of-frame ROI count as an ``unknown`` vote.
        ``ignore_boxes`` are vehicle boxes (x1, y1, x2, y2) whose pixels are not allowed to vote.
        """
        self.raw = self._classify_frame(frame, ignore_boxes)
        self._votes.append(self.raw)
        self.current = self._smooth()
        self._history.append((frame_no, self.current.state))
        return self.current

    def _classify_frame(self, frame: Optional[np.ndarray],
                        ignore_boxes: Optional[Sequence[Sequence[float]]] = None) -> SignalReading:
        if self.roi is None or not is_valid_frame(frame):
            return SignalReading()
        x, y, w, h = self.roi
        crop = safe_crop(frame, (x, y, x + w, y + h))
        if crop is None:
            return SignalReading()
        if self.mask_vehicles and ignore_boxes:
            hidden = np.zeros(crop.shape[:2], bool)
            ch, cw = crop.shape[:2]
            for b in ignore_boxes:
                x1, y1 = max(0, int(b[0]) - x), max(0, int(b[1]) - y)
                x2, y2 = min(cw, int(b[2]) - x), min(ch, int(b[3]) - y)
                if x2 > x1 and y2 > y1:
                    hidden[y1:y2, x1:x2] = True
            if hidden.any():
                if hidden.mean() > self.OCCLUDED_FRACTION:
                    return SignalReading()          # the lamp is hidden behind vehicles
                crop = crop.copy()
                crop[hidden] = 0
        return self.classifier.classify(crop)

    def _smooth(self) -> SignalReading:
        """Majority vote; ties keep the previous state, else the most recent vote."""
        counts = Counter(v.state for v in self._votes)
        top = max(counts.values())
        tied = [s for s, c in counts.items() if c == top]
        if len(tied) == 1:
            winner = tied[0]
        elif self.current.state in tied:
            winner = self.current.state
        else:
            winner = next(v.state for v in reversed(self._votes) if v.state in tied)
        confs = [v.confidence for v in self._votes if v.state == winner]
        return SignalReading(winner, float(np.mean(confs)) if confs else 0.0)

    def force(self, frame_no: int, state: str) -> SignalReading:
        """Override the smoothed state (testing / demos without a visible signal)."""
        self.current = SignalReading(state, 1.0)
        self._history.append((frame_no, state))
        return self.current

    # ------------------------------------------------------------------ lookup
    def state_at(self, frame_no: int) -> str:
        """Smoothed state at (or just before) ``frame_no``; ``unknown`` if no history."""
        for fn, state in reversed(self._history):
            if fn <= frame_no:
                return state
        return self._history[0][1] if self._history else UNKNOWN

    def state_for_crossing(self, frame_no: int) -> str:
        """State to use for a crossing: the state ``lag_frames`` before ``frame_no``.

        Compensates for the delay that majority-vote smoothing adds at transitions.
        """
        return self.state_at(frame_no - self.lag)

    def resolve(self, state: str) -> Optional[str]:
        """Apply ``unknown_policy``: ``None`` means 'do not flag anything'."""
        if state in STATES:
            return state
        return GREEN if self.unknown_policy == "treat_as_green" else None

    # ----------------------------------------------------------------- drawing
    def draw(self, frame: np.ndarray, with_label: bool = True, name: str = "") -> None:
        """Outline the ROI in the current state's colour (and label it, prefixed by ``name``)."""
        if self.roi is None:
            return
        x, y, w, h = self.roi
        color = SIGNAL_COLORS.get(self.current.state, SIGNAL_COLORS[UNKNOWN])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if with_label:
            label = (f"{name}: " if name else "SIGNAL: ") + self.current.state.upper()
            draw_label(frame, label, (x, max(14, y - 6)), color=(255, 255, 255), bg=color)
