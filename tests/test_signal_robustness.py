"""Signal classification must not be fooled by passing vehicles, and must accept per-camera overrides."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.signal_classifier import GREEN, RED, UNKNOWN, YELLOW, SignalStateTracker
from src.utils import get_approaches, load_config

ROOT = Path(__file__).resolve().parents[1]
ROI = [100, 50, 60, 120]


def _cfg(**signal) -> dict:
    cfg = load_config(ROOT / "config.yaml")
    cfg["signal"].update(smoothing_window=3, **signal)
    return cfg


def _scene(taxi: bool) -> np.ndarray:
    """A red lamp in the ROI; optionally a big yellow 'taxi' passing in front of the rest of the ROI."""
    f = np.full((300, 300, 3), 70, np.uint8)
    cv2.rectangle(f, (110, 60), (150, 160), (20, 20, 20), -1)
    cv2.circle(f, (130, 80), 10, (0, 0, 255), -1)                     # red lamp
    if taxi:
        cv2.rectangle(f, (100, 110), (160, 170), (0, 220, 255), -1)   # yellow body, larger than the lamp
    return f


def _run(tracker: SignalStateTracker, frame: np.ndarray, boxes=None, n: int = 6) -> str:
    for i in range(n):
        tracker.update(i, frame, ignore_boxes=boxes)
    return tracker.current.state


def test_yellow_vehicle_in_the_roi_fools_an_unmasked_classifier():
    unmasked = SignalStateTracker(_cfg(mask_vehicles=False), ROI)
    assert _run(unmasked, _scene(taxi=True)) == YELLOW                 # the failure we saw on the Bangkok clip


def test_masking_the_detected_vehicle_restores_the_red_reading():
    masked = SignalStateTracker(_cfg(mask_vehicles=True), ROI)
    assert _run(masked, _scene(taxi=True), boxes=[(95, 105, 165, 175)]) == RED
    assert _run(SignalStateTracker(_cfg(), ROI), _scene(taxi=False), boxes=[(0, 0, 10, 10)]) == RED   # unrelated boxes: no effect


def test_vehicle_covering_the_whole_roi_means_unreadable_not_a_guess():
    t = SignalStateTracker(_cfg(), ROI)
    assert _run(t, _scene(taxi=False), boxes=[(90, 40, 170, 180)]) == UNKNOWN
    assert t.resolve(t.current.state) is None                          # ... which suppresses violations


def test_masking_never_modifies_the_callers_frame():
    frame = _scene(taxi=True)
    before = frame.copy()
    _run(SignalStateTracker(_cfg(), ROI), frame, boxes=[(95, 105, 165, 175)])
    assert np.array_equal(frame, before)


def test_per_camera_override_changes_the_threshold():
    f = np.full((300, 300, 3), 70, np.uint8)
    cv2.circle(f, (130, 80), 2, (0, 0, 255), -1)                       # a tiny lamp: ~13 px in a 7200 px ROI (0.2 %)
    strict = SignalStateTracker(_cfg(), ROI)                           # default min_pixel_ratio 0.02
    relaxed = SignalStateTracker(_cfg(), ROI, overrides={"min_pixel_ratio": 0.001})
    assert _run(strict, f) == UNKNOWN and _run(relaxed, f) == RED


def test_approach_overrides_merge_camera_and_approach_levels():
    cam = {"signal": {"min_pixel_ratio": 0.005, "smoothing_window": 3}, "approaches": [
        {"name": "a", "stop_line": [[0, 0], [1, 1]], "signal": {"min_pixel_ratio": 0.001}},
        {"name": "b", "stop_line": [[0, 0], [1, 1]]},
    ]}
    a, b = get_approaches(cam)
    assert a["signal_overrides"] == {"min_pixel_ratio": 0.001, "smoothing_window": 3}
    assert b["signal_overrides"] == {"min_pixel_ratio": 0.005, "smoothing_window": 3}
    assert get_approaches({"stop_line": [[0, 0], [1, 1]], "signal": {"min_pixel_ratio": 0.3}})[0]["signal_overrides"] == {"min_pixel_ratio": 0.3}
    assert get_approaches({"stop_line": None})[0]["signal_overrides"] == {}


def test_green_lamp_is_unaffected_by_masking_flag():
    f = np.full((300, 300, 3), 70, np.uint8)
    cv2.circle(f, (130, 140), 12, (0, 255, 0), -1)
    assert _run(SignalStateTracker(_cfg(), ROI), f, boxes=[(0, 0, 20, 20)]) == GREEN
