"""Unit tests for src.speed (two-line and homography speed estimation, limits)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.speed import (
    HomographySpeed,
    TwoLineSpeed,
    build_speed_estimator,
    is_overspeed,
    limit_for,
)

FPS = 30.0
LINE_A = ((0, 300), (640, 300))
LINE_B = ((0, 400), (640, 400))


def box(x: float, y: float):
    """Box whose bottom-centre is (x, y)."""
    return (x - 20, y - 40, x + 20, y)


def lines_est(**params):
    return TwoLineSpeed(LINE_A, LINE_B, 20.0, params)


def drive(est, ys, x=320.0, tid=1, start=0):
    """Feed one vehicle at the given bottom-y positions (30 fps). Returns all measurements."""
    out = []
    for i, y in enumerate(ys):
        f = start + i
        est.update(f, f / FPS, [(tid, box(x, y))])
        out += est.pop_measurements()
    return out


# ------------------------------------------------------------------ two lines
def test_two_lines_known_speed():
    # 5 px/frame over a 100 px gap = 20 frames = 0.667 s for 20 m -> 108 km/h
    m = drive(lines_est(), np.arange(250, 451, 5.0))
    assert len(m) == 1
    assert m[0].speed_kmh == pytest.approx(108.0, rel=0.02)
    assert m[0].method == "lines" and m[0].travel_m == 20.0


def test_two_lines_works_in_either_direction():
    m = drive(lines_est(), np.arange(450, 249, -5.0))
    assert len(m) == 1 and m[0].speed_kmh == pytest.approx(108.0, rel=0.02)


def test_two_lines_slow_vehicle():
    m = drive(lines_est(), np.arange(280, 421, 1.0))     # 1 px/frame -> 100 frames = 3.33 s -> 21.6 km/h
    assert len(m) == 1 and m[0].speed_kmh == pytest.approx(21.6, rel=0.03)


def test_two_lines_sub_frame_interpolation_beats_frame_rounding():
    # 7 px/frame: line crossings fall between frames; interpolation keeps the error small
    m = drive(lines_est(), np.arange(250, 451, 7.0))
    assert m[0].speed_kmh == pytest.approx(20.0 / (100 / 7 / FPS) * 3.6, rel=0.02)


def test_two_lines_tolerates_pixel_noise():
    rng = np.random.default_rng(3)
    ys = np.arange(250, 451, 5.0) + rng.normal(0, 0.8, 41)
    m = drive(lines_est(), ys)
    assert len(m) == 1 and m[0].speed_kmh == pytest.approx(108.0, rel=0.08)


def test_two_lines_measures_each_vehicle_once():
    est = lines_est()
    m = drive(est, np.arange(250, 451, 5.0))
    m += drive(est, np.arange(450, 249, -5.0), start=100)   # same track keeps going back: no second measurement
    assert len(m) == 1


def test_two_lines_no_measurement_if_second_line_not_reached():
    assert drive(lines_est(), np.arange(250, 391, 5.0)) == []


def test_two_lines_ignores_crossing_beside_the_segment():
    assert drive(lines_est(), np.arange(250, 451, 5.0), x=900.0) == []


def test_two_lines_discards_physically_implausible_speed():
    # 60 px/frame -> 0.056 s between lines: below min_travel_seconds, treated as a glitch
    assert drive(lines_est(), [250, 310, 370, 430, 490]) == []


def test_two_lines_two_vehicles_independent():
    est = lines_est()
    out = []
    for i in range(41):
        y1, y2 = 250 + 5 * i, 250 + 2.5 * i
        est.update(i, i / FPS, [(1, box(200, y1)), (2, box(440, y2))])
        out += est.pop_measurements()
    speeds = {m.track_id: m.speed_kmh for m in out}
    assert speeds[1] == pytest.approx(108.0, rel=0.02)
    assert 2 not in speeds or speeds[2] == pytest.approx(54.0, rel=0.03)


def test_two_lines_invalid_input():
    with pytest.raises(ValueError):
        TwoLineSpeed(LINE_A, LINE_B, 0)
    with pytest.raises(ValueError):
        TwoLineSpeed(((5, 5), (5, 5)), LINE_B, 10)
    est = lines_est()
    est.update(0, 0.0, [(1, (np.nan, 0, 1, 1)), (2, None), (3, (5, 5, 1, 1))])   # skipped, no crash


# ------------------------------------------------------------------ homography
WORLD = np.float32([[0, 0], [3.5, 0], [3.5, 12], [0, 12]])             # near-left, near-right, far-right, far-left (m)
IMAGE = np.float32([[300, 600], [500, 600], [460, 300], [340, 300]])    # perspective view of that rectangle
W2I = cv2.getPerspectiveTransform(WORLD, IMAGE)


def project(x_m: float, y_m: float):
    p = cv2.perspectiveTransform(np.array([[[x_m, y_m]]], np.float32), W2I)[0, 0]
    return float(p[0]), float(p[1])


def hom_est(**params):
    return HomographySpeed(IMAGE, WORLD, params)


def drive_world(est, y_start, y_end, speed_ms, x_m=1.75, tid=1, start=0):
    """Vehicle moving along the road at a constant speed (metres/second), projected into the image."""
    out, n = [], int(abs(y_end - y_start) / speed_ms * FPS) + 1
    for i in range(n):
        y = y_start + np.sign(y_end - y_start) * speed_ms * i / FPS
        px, py = project(x_m, y)
        est.update(start + i, (start + i) / FPS, [(tid, box(px, py))])
        out += est.pop_measurements()
    return out


@pytest.mark.parametrize("kmh", [30.0, 54.0, 90.0])
def test_homography_known_speed(kmh):
    est = hom_est()
    m = drive_world(est, -3.0, 20.0, kmh / 3.6)     # starts before the region, leaves after it
    m += est.flush(999, 999 / FPS)
    assert len(m) == 1
    assert m[0].speed_kmh == pytest.approx(kmh, rel=0.04)
    assert m[0].method == "homography" and m[0].travel_m > 5


def test_homography_other_direction_and_lane():
    est = hom_est()
    m = drive_world(est, 20.0, -3.0, 54 / 3.6, x_m=0.8)
    m += est.flush(999, 999 / FPS)
    assert len(m) == 1 and m[0].speed_kmh == pytest.approx(54.0, rel=0.04)


def test_homography_vehicle_outside_region_gives_nothing():
    est = hom_est()
    m = drive_world(est, 13.5, 30.0, 10.0)
    assert m + est.flush(999, 1.0) == []


def test_homography_flush_finishes_vehicle_still_inside():
    est = hom_est()
    drive_world(est, 0.5, 10.5, 54 / 3.6)            # stays inside the region
    m = est.flush(500, 500 / FPS)
    assert len(m) == 1 and m[0].speed_kmh == pytest.approx(54.0, rel=0.05)


def test_homography_stationary_vehicle_not_reported():
    est = hom_est()
    px, py = project(1.75, 5.0)
    for i in range(60):
        est.update(i, i / FPS, [(1, box(px, py))])
    assert est.flush(100, 3.0) == []


def test_homography_track_that_vanishes_is_finalized():
    est = hom_est(absent_frames=4)
    drive_world(est, 0.5, 9.0, 54 / 3.6)
    out = []
    for f in range(100, 110):                         # track no longer reported
        est.update(f, f / FPS, [])
        out += est.pop_measurements()
    assert len(out) == 1 and out[0].speed_kmh == pytest.approx(54.0, rel=0.05)


def test_homography_invalid_calibration():
    with pytest.raises(ValueError):
        HomographySpeed([[0, 0], [1, 1], [2, 2], [3, 3]], WORLD)          # collinear image points
    with pytest.raises(ValueError):
        HomographySpeed(IMAGE[:3], WORLD[:3])                             # needs 4 points
    with pytest.raises(ValueError):
        HomographySpeed(IMAGE, [[0, 0], [0, 0], [0, 0], [0, 0]])          # degenerate world points


def test_homography_projection_matches_known_geometry():
    est = hom_est()
    x, y = est.to_ground(project(2.0, 7.0))
    assert (x, y) == pytest.approx((2.0, 7.0), abs=1e-2)


# ------------------------------------------------------------------ limits / factory
CFG = {"speed": {"default_limit_kmh": 60, "tolerance_kmh": 3,
                 "limits_kmh": {"car": 60, "Motorcycle": 50, "rickshaw": 40}}}


def test_limits_are_per_class_case_insensitive_with_default():
    assert limit_for(CFG, "car") == 60
    assert limit_for(CFG, "motorcycle") == 50
    assert limit_for(CFG, "RICKSHAW") == 40
    assert limit_for(CFG, "tractor") == 60


def test_overspeed_uses_limit_plus_tolerance():
    assert not is_overspeed(CFG, "car", 63.0)
    assert is_overspeed(CFG, "car", 63.1)
    assert is_overspeed(CFG, "rickshaw", 50)
    assert not is_overspeed(CFG, "rickshaw", 42)


def test_factory():
    assert build_speed_estimator(CFG, {}) is None
    assert build_speed_estimator(CFG, {"speed": None}) is None
    assert build_speed_estimator({"speed": {"enabled": False}}, {"speed": {"method": "lines"}}) is None
    est = build_speed_estimator(CFG, {"speed": {"method": "lines", "lines": [LINE_A, LINE_B], "distance_m": 20}})
    assert isinstance(est, TwoLineSpeed)
    est = build_speed_estimator(CFG, {"speed": {"method": "homography", "image_points": IMAGE.tolist(),
                                                "world_points": WORLD.tolist()}})
    assert isinstance(est, HomographySpeed)
    with pytest.raises(ValueError):
        build_speed_estimator(CFG, {"speed": {"method": "radar"}})
    with pytest.raises(ValueError):
        build_speed_estimator(CFG, {"speed": {"method": "lines", "lines": [LINE_A], "distance_m": 5}})
    with pytest.raises(ValueError):
        build_speed_estimator(CFG, {"speed": {"method": "homography"}})
