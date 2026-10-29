"""The dashboard Setup page API: serve a frame, validate and save drawn calibration into config.yaml."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dashboard.api.main import create_app
from src.utils import get_approaches, get_camera, load_config, save_config_value

ROOT = Path(__file__).resolve().parents[1]
H = {"Authorization": "Bearer tok"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "videos").mkdir()
    vid = tmp_path / "videos" / "clip.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1920, 1080))
    for n in range(6):
        f = np.full((1080, 1920, 3), 60, np.uint8)
        cv2.putText(f, str(n), (100, 200), 0, 4, (255, 255, 255), 6)
        w.write(f)
    w.release()
    cfg = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    save_config_value(cfg, "dashboard.api_token", "tok")
    save_config_value(cfg, "dashboard.dist_unused", True)
    app = create_app(str(cfg))
    c = TestClient(app)
    c.cfg_path = cfg
    return c


def cam(client, name="junction_1"):
    return get_camera(load_config(client.cfg_path), name)


def test_frame_is_a_downscaled_jpeg_with_the_real_size_in_headers(client):
    r = client.get("/api/setup/frame", params={"source": "clip.mp4", "index": 3, "max_width": 800}, headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert (r.headers["x-frame-width"], r.headers["x-frame-height"]) == ("1920", "1080")
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 800                                          # downscaled for transfer, coordinates stay in the original frame
    assert client.get("/api/setup/frame", params={"source": "nope.mp4"}, headers=H).status_code == 400
    assert client.get("/api/setup/frame", params={"source": "clip.mp4"}).status_code == 401


def test_save_line_roi_and_sensitivity_for_an_approach(client):
    body = {"camera": "junction_1", "approach": "left_lanes", "frame_size": [1920, 1080],
            "stop_line": [[100.4, 700], [900, 702.6]], "direction": "negative", "signal_roi": [1500, 100, 60, 150], "min_pixel_ratio": 0.004}
    r = client.post("/api/setup/save", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert set(r.json()["saved"]) == {"stop line", "signal ROI", "signal sensitivity"}
    ap = get_approaches(cam(client))[0]
    assert ap["name"] == "left_lanes" and ap["stop_line"] == [[100, 700], [900, 703]] and ap["direction"] == "negative"
    assert ap["signal_roi"] == [1500, 100, 60, 150] and ap["signal_overrides"] == {"min_pixel_ratio": 0.004}
    assert cam(client)["frame_size"] == [1920, 1080]
    assert "# AI Traffic Violation" in client.cfg_path.read_text(encoding="utf-8")        # comments survive
    # a second approach, and the page can read everything back
    client.post("/api/setup/save", json={"camera": "junction_1", "approach": "main_road", "frame_size": [1920, 1080],
                                         "stop_line": [[1000, 700], [1800, 700]]}, headers=H)
    info = client.get("/api/setup/camera/junction_1", headers=H).json()
    assert [a["name"] for a in info["approaches"]] == ["left_lanes", "main_road"] and info["exists"]


def test_save_legacy_single_line_without_an_approach(client):
    r = client.post("/api/setup/save", json={"camera": "old_style", "frame_size": [1280, 720], "stop_line": [[10, 300], [600, 300]]}, headers=H)
    assert r.status_code == 200 and cam(client, "old_style")["stop_line"] == [[10, 300], [600, 300]]
    assert get_approaches(cam(client, "old_style"))[0]["name"] == "main"


def test_save_speed_calibrations_both_methods_and_clear(client):
    lines = {"camera": "c", "frame_size": [1920, 1080], "speed": {"method": "lines", "points": [[100, 300], [900, 300], [80, 500], [920, 500]], "distance_m": 20}}
    assert client.post("/api/setup/save", json=lines, headers=H).status_code == 200
    assert cam(client, "c")["speed"] == {"method": "lines", "lines": [[[100, 300], [900, 300]], [[80, 500], [920, 500]]], "distance_m": 20.0}
    hom = {"camera": "c", "frame_size": [1920, 1080], "speed": {"method": "homography", "rect_m": [3.5, 12], "points": [[300, 900], [700, 900], [620, 400], [380, 400]]}}
    assert client.post("/api/setup/save", json=hom, headers=H).status_code == 200
    sp = cam(client, "c")["speed"]
    assert sp["method"] == "homography" and sp["world_points"][2] == [3.5, 12.0] and "lines" not in sp
    assert client.post("/api/setup/save", json={"camera": "c", "frame_size": [1920, 1080], "clear_speed": True}, headers=H).status_code == 200
    assert cam(client, "c")["speed"] is None


@pytest.mark.parametrize("patch,needle", [
    ({"stop_line": [[5, 5], [8, 6]]}, "shorter"),
    ({"signal_roi": [10, 10, 2, 2]}, "ROI must be"),
    ({"speed": {"method": "lines", "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}}, "distance"),           # no distance_m: rejected by validation
    ({"speed": {"method": "homography", "points": [[0, 0], [10, 10], [20, 20], [30, 30]], "rect_m": [3, 10]}}, "collinear"),
    ({"speed": {"method": "homography", "points": [[0, 0], [9, 0], [9, 9], [0, 9]]}}, "width and length"),
    ({}, "Nothing to save"),
    ({"camera": "bad name!"}, "names may only"),
    ({"frame_size": [5, 5], "stop_line": [[0, 0], [100, 100]]}, "frame_size"),
])
def test_invalid_input_is_rejected_and_the_config_is_untouched(client, patch, needle):
    before = client.cfg_path.read_text(encoding="utf-8")
    body = {"camera": "z", "frame_size": [1920, 1080], **patch}
    r = client.post("/api/setup/save", json=body, headers=H)
    assert r.status_code in (400, 422) and needle.lower() in r.text.lower(), r.text
    assert client.cfg_path.read_text(encoding="utf-8") == before


def test_editing_can_be_disabled_and_needs_the_token(client):
    assert client.post("/api/setup/save", json={"camera": "z", "frame_size": [1920, 1080], "stop_line": [[0, 0], [100, 0]]}).status_code == 401
    save_config_value(client.cfg_path, "dashboard.allow_config_edit", False)
    from dashboard.api.services.context import AppContext          # the running app re-reads the file only on a save, so reload it manually
    ctx: AppContext = client.app.state.ctx
    ctx.cfg.update(load_config(client.cfg_path))
    r = client.post("/api/setup/save", json={"camera": "z", "frame_size": [1920, 1080], "stop_line": [[0, 0], [100, 0]]}, headers=H)
    assert r.status_code == 403


def test_delete_approach_and_running_pipeline_sees_new_config(client):
    for name in ("a", "b"):
        client.post("/api/setup/save", json={"camera": "q", "approach": name, "frame_size": [1920, 1080], "stop_line": [[0, 5], [300, 5]]}, headers=H)
    assert client.delete("/api/setup/approach", params={"camera": "q", "name": "a"}, headers=H).json()["approaches"][0]["name"] == "b"
    assert client.delete("/api/setup/approach", params={"camera": "q", "name": "zzz"}, headers=H).status_code == 404
    # the LiveService shares the very same dict, so the next run starts with the saved setup
    assert [a["name"] for a in get_approaches(get_camera(client.app.state.ctx.live.cfg, "q"))] == ["b"]
