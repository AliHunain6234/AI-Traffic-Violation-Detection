"""Live-monitor source handling: file-name lookup, folder rules, camera permissions, URL masking."""
from __future__ import annotations

import os

import pytest

from dashboard.api.services.live_service import LiveService, RunError
from src.stream import is_live_source, mask_source, parse_source


@pytest.fixture()
def service(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "WhatsApp Video 2026-10-27 at 10.08.17 AM.mp4").write_bytes(b"x")
    (tmp_path / "videos" / "other.mp4").write_bytes(b"x")
    (tmp_path / "root clip.avi").write_bytes(b"x")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "annotated.mp4").write_bytes(b"x")               # generated output: never offered as input
    (tmp_path / "videos" / "notes.txt").write_text("not a video")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    return LiveService({"dashboard": {"allowed_video_dirs": ["."], "allow_live_sources": True}})


def test_bare_file_name_is_found_inside_videos_folder(service, tmp_path):
    got = service.check_video("WhatsApp Video 2026-10-27 at 10.08.17 AM.mp4")
    assert os.path.samefile(got, tmp_path / "videos" / "WhatsApp Video 2026-10-27 at 10.08.17 AM.mp4")
    assert os.path.samefile(service.check_video('"root clip.avi"'), tmp_path / "root clip.avi")      # quotes tolerated
    assert os.path.samefile(service.check_video("videos/other.mp4"), tmp_path / "videos" / "other.mp4")
    assert os.path.samefile(service.check_video("OTHER.MP4"), tmp_path / "videos" / "other.mp4")      # case-insensitive


def test_choices_list_only_input_videos_with_relative_paths(service):
    choices = service.video_choices()
    assert sorted(choices) == sorted(["videos/WhatsApp Video 2026-10-27 at 10.08.17 AM.mp4", "videos/other.mp4", "root clip.avi"])
    assert not any("outputs" in c or c.endswith(".txt") for c in choices)


def test_unknown_ambiguous_and_non_video_names_are_rejected_with_a_hint(service, tmp_path):
    with pytest.raises(RunError, match="Video not found"):
        service.check_video("missing.mp4")
    (tmp_path / "elsewhere" / "other.mp4").write_bytes(b"x")
    with pytest.raises(RunError, match="matches 2 files"):
        service.check_video("other.mp4")
    with pytest.raises(RunError):
        service.check_video("videos/notes.txt")
    with pytest.raises(RunError):
        service.check_video("C:/Windows/notepad.exe")


def test_files_outside_allowed_folders_are_refused(tmp_path, monkeypatch):
    (tmp_path / "allowed").mkdir()
    (tmp_path / "secret.mp4").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    svc = LiveService({"dashboard": {"allowed_video_dirs": ["allowed"], "allow_live_sources": False}})
    with pytest.raises(RunError):
        svc.check_video(str(tmp_path / "secret.mp4"))


def test_camera_sources_need_permission(service):
    assert service.check_video("0") == "0"
    assert service.check_video("rtsp://u:p@cam/s") == "rtsp://u:p@cam/s"
    off = LiveService({"dashboard": {"allowed_video_dirs": ["."], "allow_live_sources": False}})
    for src in ("0", "rtsp://cam/s", "http://cam/video"):
        with pytest.raises(RunError, match="disabled"):
            off.check_video(src)


def test_source_helpers():
    assert parse_source("0") == 0 and parse_source(" 3 ") == 3 and parse_source("a.mp4") == "a.mp4" and parse_source(2) == 2
    assert is_live_source(0) and is_live_source("RTSP://x") and is_live_source("http://x/video")
    assert not is_live_source("clip.mp4") and not is_live_source("C:/v/a.mp4")
    assert mask_source("rtsp://admin:s3cret@10.0.0.5:554/s1") == "rtsp://admin:****@10.0.0.5:554/s1"
    assert mask_source("rtsp://10.0.0.5/s1") == "rtsp://10.0.0.5/s1" and mask_source(0) == "0"


def _clip(tmp_path, frames=60, fps=30):
    import cv2
    import numpy as np
    path = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        out.write(np.full((48, 64, 3), i * 4 % 255, np.uint8))
    out.release()
    return path


def test_realtime_file_skips_frames_for_a_slow_consumer_and_keeps_video_time(tmp_path):
    import time
    from src.stream import FileSource
    src = FileSource(_clip(tmp_path), realtime=True)           # 2 s of video
    t0, seen = time.monotonic(), []
    while (item := src.read()) is not None:
        seen.append(item)
        time.sleep(0.25)                                       # a detector that manages 4 frames per second
    wall = time.monotonic() - t0
    assert wall < 3.5                                          # played at ~video speed (2 s clip), not 8 s of slow motion
    assert 4 <= len(seen) <= 12 and src.skipped > 40
    idx = [i for _, i, _ in seen]
    assert idx == sorted(set(idx))
    assert all(abs(s - i / 30) < 1e-9 for _, i, s in seen)     # timestamps stay the video's own clock
    src.close()


def test_live_view_paints_the_latest_detections_on_fresh_camera_frames(service):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    raw = np.zeros((48, 64, 3), np.uint8)
    annotated = raw.copy()
    annotated[10:20, 10:20] = (0, 255, 0)                                   # a drawn box
    service._pipeline = SimpleNamespace(last_processed=raw, live_source=True)
    service.min_interval = 0
    service._on_frame(annotated, {"fps": 3})
    assert service._jpeg is None                                            # a live camera is published per camera frame
    fresh = np.full((48, 64, 3), 90, np.uint8)                              # a newer camera picture
    service._on_camera_frame(fresh)
    shown = cv2.imdecode(np.frombuffer(service._jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert shown[15, 15, 1] > 200 and shown[15, 15, 2] < 60                 # the box is there (green)
    assert abs(int(shown[40, 55, 0]) - 90) < 12                             # the rest is the fresh frame


def test_plain_file_source_reads_every_frame(tmp_path):
    from src.stream import FileSource
    src = FileSource(_clip(tmp_path, frames=20))
    assert sum(1 for _ in iter(src.read, None)) == 20
