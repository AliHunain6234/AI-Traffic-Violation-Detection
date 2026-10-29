"""Multi-object tracking via Ultralytics' built-in trackers (BoT-SORT / ByteTrack).

No custom tracker: this only selects the tracker config and calls
``model.track(..., persist=True)`` on the detector's model.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .detector import Detection, VehicleDetector
from .utils import is_valid_frame

log = logging.getLogger(__name__)

SUPPORTED_TRACKERS = ("botsort", "bytetrack")


class VehicleTracker:
    """Assigns persistent IDs to detections using an Ultralytics tracker.

    Args:
        detector: The ``VehicleDetector`` whose model is reused for tracking.
        cfg: Full config (uses the ``tracker`` section).

    Raises:
        ValueError: If ``tracker.name`` is not ``botsort`` or ``bytetrack`` and no
            ``custom_yaml`` is given.
    """

    def __init__(self, detector: VehicleDetector, cfg: dict) -> None:
        trk = cfg.get("tracker", {})
        self.detector = detector
        self.name = str(trk.get("name", "botsort")).lower()
        self.persist = bool(trk.get("persist", True))
        custom = trk.get("custom_yaml")
        if custom:
            self.tracker_yaml = str(custom)
        elif self.name in SUPPORTED_TRACKERS:
            self.tracker_yaml = f"{self.name}.yaml"
        else:
            raise ValueError(f"Unknown tracker {self.name!r}; expected one of {SUPPORTED_TRACKERS}")
        log.info("Tracker: %s (%s)", self.name, self.tracker_yaml)

    def track(self, frame: Optional[np.ndarray]) -> list[Detection]:
        """Detect and track vehicles in one frame.

        Returns every detection; ``track_id`` is ``None`` for boxes the tracker has
        not confirmed yet, so callers that need IDs should filter on it.
        Empty/invalid frames give ``[]``.
        """
        if not is_valid_frame(frame):
            return []
        results = self.detector.model.track(
            frame,
            persist=self.persist,
            tracker=self.tracker_yaml,
            **self.detector.predict_kwargs,
        )
        return self.detector.parse(results[0]) if results else []

    def reset(self) -> None:
        """Forget all tracks (call between videos so IDs do not leak across runs)."""
        predictor = getattr(self.detector.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for t in trackers:
                t.reset()
        else:
            self.detector.model.predictor = None
