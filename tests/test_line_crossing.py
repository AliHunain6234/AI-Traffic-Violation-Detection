"""Unit tests for src.line_crossing.

Geometry used throughout: a horizontal stop line at y=100 from x=0..200. Vehicles
travelling DOWN the image (increasing y) move toward the "positive" side.
"""
from __future__ import annotations

import math

import pytest

from src.line_crossing import LineCrossingDetector

LINE = ((0, 100), (200, 100))
W, H = 40, 60


def box(x: float, y_bottom: float, w: float = W, h: float = H):
    """Box whose bottom-centre is (x, y_bottom)."""
    return (x - w / 2, y_bottom - h, x + w / 2, y_bottom + 0.0)


def make(direction="positive", **overrides):
    params = {
        "point_smoothing_frames": 1,
        "min_displacement_px": 4,
        "segment_margin_px": 20,
        "min_track_age_frames": 1,
        "once_per_id": True,
        "cooldown_frames": 90,
        "dedup_window_frames": 45,
        "dedup_radius_px": 80,
    }
    params.update(overrides)
    return LineCrossingDetector(LINE, direction, params)


def run(det, track_id, ys, x=100, start_frame=0):
    """Feed one track moving through the given bottom-y values; return all events."""
    events = []
    for i, y in enumerate(ys):
        events += det.update(start_frame + i, [(track_id, box(x, y))])
    return events


def test_crossing_in_configured_direction_fires_once():
    det = make()
    events = run(det, 1, [80, 90, 96, 104, 110, 120])
    assert len(events) == 1
    assert events[0].track_id == 1 and events[0].kind == "line"
    assert det.has_crossed(1) and det.crossed_count == 1


def test_wrong_direction_is_ignored():
    det = make()
    assert run(det, 1, [120, 110, 104, 96, 90, 80]) == []
    assert not det.has_crossed(1)


def test_negative_direction_counts_upward_travel():
    det = make(direction="negative")
    assert len(run(det, 1, [120, 110, 104, 96, 90])) == 1
    assert run(make(direction="negative"), 2, [80, 90, 110, 120]) == []


def test_uses_bottom_center_not_box_center():
    det = make()
    # Box centre crosses y=100 when bottom is ~130, but the bottom is still before the line.
    assert run(det, 1, [80, 85, 90, 95]) == []
    # Tall box: centre (bottom - 100) is far above the line; bottom crossing still counts.
    det2 = make()
    events = []
    for i, y in enumerate([90, 96, 104, 112]):
        events += det2.update(i, [(1, box(100, y, h=200))])
    assert len(events) == 1


def test_each_track_id_counted_at_most_once():
    det = make()
    events = run(det, 1, [90, 96, 104, 110, 104, 96, 90, 96, 104, 112])  # cross, back, cross again
    assert len(events) == 1


def test_stopped_on_line_with_jitter_does_not_trigger():
    det = make()
    # Hover within the 4px dead-band around y=100 for many frames.
    assert run(det, 1, [90, 96, 99, 101, 100, 102, 98, 101, 99, 100] * 3) == []
    # Only once it clearly leaves the dead-band does it count, exactly once.
    events = run(det, 1, [106, 112, 118], start_frame=30)
    assert len(events) == 1


def test_jitter_back_and_forth_across_line_counts_once():
    det = make()
    events = run(det, 1, [95, 105, 95, 105, 95, 105, 110, 120])
    assert len(events) == 1


def test_crossing_beyond_segment_end_is_ignored():
    det = make()
    # x=300 is 100px past the end of a 200px line; margin is only 20px.
    assert run(det, 1, [90, 96, 104, 112], x=300) == []
    # Just inside the margin still counts.
    assert len(run(make(), 2, [90, 96, 104, 112], x=210)) == 1


def test_track_born_past_the_line_does_not_count():
    det = make()
    assert run(det, 1, [120, 125, 130, 140]) == []


def test_min_track_age_filters_spurious_tracks():
    det = make(min_track_age_frames=5)
    assert run(det, 1, [96, 104, 110]) == []  # crosses on its 2nd frame
    det2 = make(min_track_age_frames=5)
    assert len(run(det2, 2, [80, 85, 90, 96, 104, 110])) == 1


def test_smoothing_damps_single_frame_spike():
    det = make(point_smoothing_frames=5)
    ys = [70, 72, 74, 76, 78, 130, 80, 82, 84, 86]  # one-frame detection spike
    assert run(det, 1, ys) == []


def test_new_id_after_crossing_does_not_double_count():
    det = make()
    events = run(det, 1, [90, 96, 104, 112])
    assert len(events) == 1
    # Track 1 is lost; track 2 appears right where it vanished and continues.
    events += run(det, 2, [116, 122, 128, 134], start_frame=6)
    assert len(events) == 1
    assert det.has_crossed(2)


def test_unrelated_new_track_far_away_still_counts():
    det = make()
    run(det, 1, [90, 96, 104, 112])
    events = run(det, 2, [90, 96, 104, 112], x=190, start_frame=6)  # >80px away
    assert len(events) == 1


def test_two_vehicles_side_by_side_both_count():
    det = make()
    events = []
    for i, y in enumerate([90, 96, 104, 112]):
        events += det.update(i, [(1, box(60, y)), (2, box(140, y))])
    assert sorted(e.track_id for e in events) == [1, 2]


def test_cooldown_applies_when_once_per_id_is_off():
    det = make(once_per_id=False, cooldown_frames=10)
    ys = [90, 96, 104, 110, 104, 96, 90, 96, 104, 110]  # re-crosses ~6 frames later
    assert len(run(det, 1, ys)) == 1
    det2 = make(once_per_id=False, cooldown_frames=3)
    assert len(run(det2, 1, ys)) == 2


def test_invalid_boxes_are_skipped():
    det = make()
    bad = [(1, (math.nan, 0, 10, 10)), (2, (10, 10, 5, 5)), (3, (0, 0, math.inf, 5)), (4, None)]
    assert det.update(0, bad) == []
    assert det.update(1, []) == []


def test_stale_state_is_pruned():
    det = make(state_ttl_frames=10)
    run(det, 1, [90])
    det.update(50, [])
    assert 1 not in det._states


def test_reset_clears_everything():
    det = make()
    run(det, 1, [90, 96, 104, 112])
    det.reset()
    assert det.crossed_count == 0
    assert len(run(det, 1, [90, 96, 104, 112], start_frame=100)) == 1


def test_zone_entry_triggers_once():
    zone = [[(50, 100), (150, 100), (150, 200), (50, 200)]]
    det = LineCrossingDetector(None, "positive", {"min_track_age_frames": 1, "point_smoothing_frames": 1}, zones=zone)
    events = run(det, 1, [80, 90, 105, 120, 140])
    assert len(events) == 1 and events[0].kind == "zone" and events[0].zone_index == 0


def test_no_line_and_no_zone_returns_nothing():
    det = LineCrossingDetector(None, "positive", {})
    assert run(det, 1, [90, 110]) == []


def test_bad_direction_and_degenerate_line_raise():
    with pytest.raises(ValueError):
        LineCrossingDetector(LINE, "sideways")
    with pytest.raises(ValueError):
        LineCrossingDetector(((5, 5), (5, 5)), "positive")


def test_diagonal_line_direction():
    line = ((0, 0), (200, 200))
    det = LineCrossingDetector(line, "positive", {"min_track_age_frames": 1, "point_smoothing_frames": 1})
    # side_of_line positive is below/left of a diagonal drawn top-left to bottom-right.
    events = []
    for i, (x, y) in enumerate([(120, 80), (110, 92), (100, 104), (90, 116)]):
        events += det.update(i, [(1, box(x, y))])
    assert len(events) == 1
