"""Tests for multi-approach cameras (each stop line has its own signal)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from main import scale_camera
from src.utils import get_approaches, get_camera, load_config, remove_approach, save_approach_value

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_single_line_becomes_one_approach_named_main():
    cam = {"stop_line": [[0, 1], [5, 1]], "direction": "negative", "signal_roi": [1, 2, 3, 4], "violation_zones": []}
    aps = get_approaches(cam)
    assert len(aps) == 1 and aps[0]["name"] == "main"
    assert aps[0]["stop_line"] == [[0, 1], [5, 1]] and aps[0]["direction"] == "negative" and aps[0]["signal_roi"] == [1, 2, 3, 4]


def test_multiple_approaches_are_used_as_given():
    cam = {"stop_line": [[9, 9], [9, 9]], "approaches": [
        {"name": "left", "stop_line": [[0, 5], [10, 5]], "signal_roi": [1, 1, 5, 5]},
        {"name": "right", "stop_line": [[20, 5], [30, 5]], "direction": "negative"},
    ]}
    aps = get_approaches(cam)
    assert [a["name"] for a in aps] == ["left", "right"]
    assert aps[0]["direction"] == "positive" and aps[1]["direction"] == "negative"
    assert aps[1]["signal_roi"] is None and aps[0]["violation_zones"] == []


def test_unnamed_approaches_get_default_names_and_duplicates_are_rejected():
    assert [a["name"] for a in get_approaches({"approaches": [{"stop_line": 1}, {"stop_line": 2}]})] == ["approach1", "approach2"]
    with pytest.raises(ValueError):
        get_approaches({"approaches": [{"name": "a"}, {"name": "a"}]})


def test_scale_camera_scales_every_approach_but_not_the_originals():
    cam = {"frame_size": [1000, 2000], "approaches": [
        {"name": "a", "stop_line": [[100, 200], [300, 200]], "signal_roi": [10, 20, 30, 40], "violation_zones": [[[0, 0], [10, 0], [10, 10]]]},
    ]}
    out = scale_camera(cam, 1000, 2000, 500, 1000)
    ap = out["approaches"][0]
    assert ap["stop_line"] == [[50, 100], [150, 100]] and ap["signal_roi"] == [5, 10, 15, 20]
    assert ap["violation_zones"] == [[[0, 0], [5, 0], [5, 5]]]
    assert cam["approaches"][0]["stop_line"] == [[100, 200], [300, 200]]   # input untouched


def test_save_and_remove_approach_round_trip_keeps_comments(tmp_path):
    cfg = tmp_path / "c.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    save_approach_value(cfg, "junction_1", "left", "stop_line", [[10, 20], [300, 20]])
    save_approach_value(cfg, "junction_1", "left", "signal_roi", [5, 6, 7, 8])
    save_approach_value(cfg, "junction_1", "right", "stop_line", [[400, 20], [700, 20]])
    save_approach_value(cfg, "junction_1", "left", "direction", "negative")      # updates, does not duplicate
    cam = get_camera(load_config(cfg), "junction_1")
    assert [a["name"] for a in cam["approaches"]] == ["left", "right"]
    assert cam["approaches"][0] == {"name": "left", "stop_line": [[10, 20], [300, 20]], "signal_roi": [5, 6, 7, 8], "direction": "negative"}
    assert "# AI Traffic Violation" in cfg.read_text(encoding="utf-8")
    assert remove_approach(cfg, "junction_1", "left") is True
    assert remove_approach(cfg, "junction_1", "left") is False
    assert [a["name"] for a in get_camera(load_config(cfg), "junction_1")["approaches"]] == ["right"]
    assert remove_approach(cfg, "no_such_camera", "x") is False
