"""Tests for ID stitching: heal fragmented tracks, never merge unrelated vehicles."""
from __future__ import annotations

from src.detector import Detection
from src.line_crossing import LineCrossingDetector
from src.speed import TwoLineSpeed
from src.track_stitcher import TrackStitcher

W, H = 1000, 800
P = {"max_gap_frames": 30, "max_distance_factor": 1.2}


def det(tid, cx, cy, w=60, h=40, cls="car"):
    return Detection((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.9, 2, cls, tid)


def run(stitcher, frames):
    """frames: {frame_no: [Detection, ...]}; returns {frame_no: [canonical ids]}."""
    return {f: [d.track_id for d in stitcher.apply(f, ds)] for f, ds in sorted(frames.items())}


def test_fragmented_vehicle_gets_its_old_id_back():
    frames = {f: [det(1, 200 + 5 * f, 400)] for f in range(20)}
    frames.update({f: [det(2, 200 + 5 * f, 400)] for f in range(24, 50)})        # ID switch after a 4-frame gap
    s = TrackStitcher(P, (W, H))
    ids = run(s, frames)
    assert set(sum(ids.values(), [])) == {1} and s.merges == 1


def test_two_vehicles_never_swap_or_merge_with_each_other():
    frames = {}
    for f in range(40):
        frames[f] = [det(1, 100 + 4 * f, 200), det(5, 100 + 4 * f, 600)]          # two parallel vehicles
    for f in range(44, 70):
        frames[f] = [det(9, 100 + 4 * f, 200), det(5, 100 + 4 * f, 600)]          # only the top one fragments
    ids = run(TrackStitcher(P, (W, H)), frames)
    assert ids[50] == [1, 5] and ids[10] == [1, 5]


def test_unrelated_new_vehicle_far_from_the_lost_one_is_not_merged():
    frames = {f: [det(1, 200 + 5 * f, 400)] for f in range(20)}
    frames.update({f: [det(2, 800, 150)] for f in range(24, 40)})
    ids = run(TrackStitcher(P, (W, H)), frames)
    assert ids[30] == [2]


def test_gap_too_long_is_not_merged():
    frames = {f: [det(1, 200 + 5 * f, 400)] for f in range(20)}
    frames.update({f: [det(2, 200 + 5 * f, 400)] for f in range(70, 90)})
    assert run(TrackStitcher(P, (W, H)), frames)[80] == [2]


def test_vehicles_leaving_or_entering_at_the_border_are_never_merged():
    leaving = {f: [det(1, 850 + 8 * f, 400)] for f in range(20)}                  # exits the right edge (x > 970)
    leaving.update({f: [det(2, 400, 400)] for f in range(22, 40)})                  # something new mid-scene
    assert run(TrackStitcher(P, (W, H)), leaving)[30] == [2]
    entering = {f: [det(1, 200 + 5 * f, 400)] for f in range(20)}
    entering.update({f: [det(2, 20 + 5 * (f - 24), 400)] for f in range(24, 40)})   # appears at the left border
    assert run(TrackStitcher(P, (W, H)), entering)[30] == [2]


def test_different_vehicle_kinds_are_not_merged_but_car_truck_flips_are():
    base = {f: [det(1, 200 + 5 * f, 400, cls="car")] for f in range(20)}
    moto = {**base, **{f: [det(2, 200 + 5 * f, 400, w=30, h=40, cls="motorcycle")] for f in range(24, 40)}}
    assert run(TrackStitcher(P, (W, H)), moto)[30] == [2]
    truck = {**base, **{f: [det(2, 200 + 5 * f, 400, cls="truck")] for f in range(24, 40)}}   # a rickshaw flipping car <-> truck
    assert run(TrackStitcher(P, (W, H)), truck)[30] == [1]


def test_size_must_be_similar():
    base = {f: [det(1, 200 + 5 * f, 400)] for f in range(20)}
    big = {**base, **{f: [det(2, 200 + 5 * f, 400, w=240, h=160)] for f in range(24, 40)}}
    assert run(TrackStitcher(P, (W, H)), big)[30] == [2]


def test_greedy_pairing_uses_the_nearest_lost_track():
    frames = {f: [det(1, 200 + 5 * f, 300), det(2, 200 + 5 * f, 330)] for f in range(20)}     # close, parallel
    frames.update({f: [det(3, 200 + 5 * f, 330)] for f in range(24, 40)})                      # reappears on the lower lane
    ids = run(TrackStitcher(P, (W, H)), frames)
    assert ids[30] == [2]


def test_untracked_detections_pass_through_and_reset_clears_state():
    s = TrackStitcher(P, (W, H))
    out = s.apply(0, [Detection((0, 0, 10, 10), 0.9, 2, "car", None), det(7, 300, 300)])
    assert out[0].track_id is None and out[1].track_id == 7
    s.reset()
    assert s.merges == 0


def test_stitching_lets_a_two_line_speed_measurement_survive_an_id_switch():
    lines = (((0, 300), (1000, 300)), ((0, 400), (1000, 400)))
    ys = [220 + 5 * f for f in range(50)]                       # crosses y=300 at f=16 and y=400 at f=36
    frames = {f: [det(1 if f < 25 else 2, 400, ys[f], w=60, h=40)] for f in range(50) if not 25 <= f < 28}   # ID switch between the lines
    for stitched in (False, True):
        est = TwoLineSpeed(*lines, 20.0, {})
        s = TrackStitcher(P, (W, H)) if stitched else None
        out = []
        for f, ds in sorted(frames.items()):
            ds = s.apply(f, ds) if s else ds
            est.update(f, f / 30.0, [(d.track_id, d.box) for d in ds])
            out += est.pop_measurements()
        assert (len(out) == 1) == stitched
        if stitched:
            assert abs(out[0].speed_kmh - 20.0 / (100 / 5 / 30.0) * 3.6) < 4          # ~108 km/h


def test_stitching_prevents_double_counting_a_crossing():
    line = LineCrossingDetector([[0, 400], [1000, 400]], "positive", {"min_track_age_frames": 1, "point_smoothing_frames": 1,
                                                                      "dedup_window_frames": 0})
    frames = {f: [det(1 if f < 30 else 2, 400, 300 + 5 * f)] for f in range(50) if not 30 <= f < 32}
    s = TrackStitcher(P, (W, H))
    events = []
    for f, ds in sorted(frames.items()):
        events += line.update(f, [(d.track_id, d.box) for d in s.apply(f, ds)])
    assert len(events) == 1 and events[0].track_id == 1
