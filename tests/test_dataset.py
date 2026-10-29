"""Fine-tuning toolkit: label I/O, leak-free splits, editor logic, frame extraction, dataset preparation."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import extract_frames  # noqa: E402
import prelabel  # noqa: E402
import relabel  # noqa: E402
import train_detector  # noqa: E402
from src.dataset import (  # noqa: E402
    Box, LabelEditor, append_sources, dataset_stats, frame_number, load_classes, read_labels, read_sources,
    save_classes, split_by_video, write_labels,
)


# ----------------------------------------------------------------- label I/O
def test_label_round_trip_and_clipping(tmp_path):
    p = tmp_path / "a.txt"
    write_labels(p, [Box(1, 100, 50, 300, 250), Box(0, -20, -20, 40, 40), Box(2, 10, 10, 10.4, 10.4)], 640, 480)
    got = read_labels(p, 640, 480)
    assert len(got) == 2                                               # the sub-pixel box was dropped
    assert (got[0].cls, round(got[0].x1), round(got[0].y2)) == (1, 100, 250)
    assert (got[1].x1, got[1].y1) == pytest.approx((0.0, 0.0), abs=0.01)   # clipped to the image
    assert read_labels(tmp_path / "missing.txt", 640, 480) == []
    p.write_text("1 0.5 0.5 0.2 0.2\ngarbage line\n\n0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    assert [b.cls for b in read_labels(p, 100, 100)] == [1, 0]         # bad lines are skipped, not fatal


def test_classes_file(tmp_path):
    p = tmp_path / "classes.txt"
    save_classes(p, ["car", "rickshaw"])
    assert load_classes(p) == ["car", "rickshaw"]
    p.write_text("car\nCar\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_classes(p)
    with pytest.raises(FileNotFoundError):
        load_classes(tmp_path / "nope.txt")


# ----------------------------------------------------------------- splits
def _imgs(video: str, n: int):
    return [Path(f"{video}_{i * 30:06d}.jpg") for i in range(n)]


def test_split_by_video_keeps_each_video_on_one_side():
    imgs = _imgs("a", 40) + _imgs("b", 40) + _imgs("c", 20)
    src = {im.name: im.name.split("_")[0] for im in imgs}
    train, val = split_by_video(imgs, src, 0.2, seed=1)
    assert {src[i.name] for i in train}.isdisjoint({src[i.name] for i in val})
    assert train and val and len(train) + len(val) == len(imgs)


def test_single_video_split_is_a_late_block_with_a_gap():
    imgs = _imgs("only", 100)
    train, val = split_by_video(imgs, {}, 0.2, buffer=5)
    assert max(frame_number(i.name) for i in train) < min(frame_number(i.name) for i in val)
    assert len(val) == 20 and len(train) == 100 - 20 - 5
    with pytest.raises(ValueError):
        split_by_video(_imgs("tiny", 6), {}, 0.2, buffer=5)            # no room for a validation block after the gap
    with pytest.raises(ValueError):
        split_by_video(_imgs("x", 1), {}, 0.2)


def test_frame_number_and_sources_csv(tmp_path):
    assert frame_number("my-clip_000123.jpg") == 123 and frame_number("noframe.jpg") == 0
    csv = tmp_path / "sources.csv"
    append_sources(csv, [("a_000001.jpg", "a.mp4", 1)])
    append_sources(csv, [("b_000002.jpg", "b.mp4", 2)])                # appending must not repeat the header
    assert read_sources(csv) == {"a_000001.jpg": "a.mp4", "b_000002.jpg": "b.mp4"}


# ----------------------------------------------------------------- editor logic
def test_label_editor_select_relabel_delete_add_undo():
    ed = LabelEditor([Box(0, 0, 0, 100, 100), Box(0, 20, 20, 40, 40)], 3)
    assert ed.select_at(30, 30) == 1                                   # the small box inside the big one wins
    ed.press_class(2)
    assert ed.boxes[1].cls == 2 and ed.current_class == 2 and ed.dirty
    assert ed.select_at(90, 90) == 0 and ed.delete_selected() and len(ed.boxes) == 1
    assert ed.select_at(500, 500) is None and not ed.delete_selected()
    assert not ed.add_box(10, 10, 12, 12)                              # a click is not a box
    assert ed.add_box(200, 200, 150, 120) and ed.boxes[-1].cls == 2    # corners may come in any order
    assert (ed.boxes[-1].x1, ed.boxes[-1].y1, ed.boxes[-1].x2, ed.boxes[-1].y2) == (150, 120, 200, 200)
    ed.press_class(9)                                                  # out of range: ignored
    assert ed.current_class == 2
    assert ed.undo() and len(ed.boxes) == 1 and ed.undo() and len(ed.boxes) == 2 and ed.boxes[1].cls == 2
    assert ed.undo() and ed.boxes[1].cls == 0 and not ed.undo()


def test_relabel_render_runs_headlessly():
    img = np.zeros((300, 400, 3), np.uint8)
    ed = LabelEditor([Box(0, 20, 20, 120, 100), Box(1, 200, 100, 300, 200)], 2)
    ed.select_at(50, 50)
    out = relabel.render(img, ed, ["car", "rickshaw"], 1.0, 0, 5, "x.jpg", False, (10, 10, 60, 60), True)
    assert out.shape == img.shape and out.any()


# ----------------------------------------------------------------- tools
def _video(path: Path, frames: int = 90, fps: int = 30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    rng = np.random.default_rng(0)
    for n in range(frames):
        f = np.full((240, 320, 3), 60, np.uint8)
        if n >= 30:                                                    # nothing moves for the first second -> duplicates
            cv2.rectangle(f, (10 + 3 * n, 80), (50 + 3 * n, 140), (0, 0, 255), -1)
        f = cv2.add(f, rng.integers(0, 12, f.shape, dtype=np.uint8))
        w.write(f)
    w.release()


def test_extract_frames_skips_duplicates_and_records_sources(tmp_path):
    vid = tmp_path / "My Clip (1).mp4"
    _video(vid)
    out = tmp_path / "ds" / "images"
    rows = extract_frames.extract(str(vid), out, every_sec=0.2, max_frames=100, min_diff=6.0, min_sharpness=0, max_side=0)
    names = [r[0] for r in rows]
    assert names and all(n.startswith("my-clip-1") for n in names)
    assert len(rows) < 90 / 6                                          # the static first second was thinned out
    assert all((out / n).is_file() for n in names)
    assert extract_frames.slug("C:/x/WhatsApp Video 2026-10-27 at 10.08.17 AM.mp4").startswith("whatsapp-video")
    assert len(extract_frames.extract(str(vid), out, 0.2, 3, 0.0, 0, 0)) == 3          # max_frames honoured


def test_prelabel_map_parsing():
    assert prelabel.parse_map(["Truck=truck", "car = rickshaw"]) == {"truck": "truck", "car": "rickshaw"}
    with pytest.raises(ValueError):
        prelabel.parse_map(["truck"])


def _make_dataset(base: Path, n: int = 40, review: bool = True):
    (base / "images").mkdir(parents=True)
    save_classes(base / "classes.txt", ["car", "rickshaw"])
    rows, reviewed = [], []
    for i in range(n):
        vid = "a" if i < n // 2 else "b"
        name = f"{vid}_{i * 10:06d}.jpg"
        img = np.full((120, 160, 3), 50, np.uint8)
        cv2.rectangle(img, (20, 30), (70, 90), (0, 0, 255), -1)
        cv2.imwrite(str(base / "images" / name), img)
        write_labels(base / "labels" / name.replace(".jpg", ".txt"), [Box(i % 2, 20, 30, 70, 90)], 160, 120)
        rows.append((name, f"{vid}.mp4", i * 10))
        reviewed.append(name)
    append_sources(base / "sources.csv", rows)
    if review:
        (base / "reviewed.txt").write_text("\n".join(reviewed), encoding="utf-8")


def test_prepare_builds_a_leak_free_split_and_data_yaml(tmp_path):
    base = tmp_path / "ds"
    _make_dataset(base)
    yaml_path = train_detector.prepare(base, 0.25, use_unreviewed=False)
    text = yaml_path.read_text(encoding="utf-8")
    assert "0: car" in text and "1: rickshaw" in text and "images/train" in text
    train_imgs = {p.name for p in (base / "yolo" / "images" / "train").iterdir()}
    val_imgs = {p.name for p in (base / "yolo" / "images" / "val").iterdir()}
    assert train_imgs and val_imgs and not train_imgs & val_imgs
    assert {n.split("_")[0] for n in train_imgs}.isdisjoint({n.split("_")[0] for n in val_imgs})     # whole videos per side
    assert all((base / "yolo" / "labels" / "train" / (n[:-4] + ".txt")).is_file() for n in train_imgs)


def test_prepare_refuses_unreviewed_tiny_or_inconsistent_datasets(tmp_path):
    base = tmp_path / "ds"
    _make_dataset(base, review=False)
    with pytest.raises(ValueError, match="No image has been reviewed"):
        train_detector.prepare(base, 0.2, use_unreviewed=False)
    train_detector.prepare(base, 0.2, use_unreviewed=True)                                            # explicit opt-in works
    small = tmp_path / "small"
    _make_dataset(small, n=6)
    with pytest.raises(ValueError, match="Only 6 labelled images"):
        train_detector.prepare(small, 0.2, use_unreviewed=False)
    bad = tmp_path / "bad"
    _make_dataset(bad)
    write_labels(bad / "labels" / "a_000000.txt", [Box(7, 1, 1, 50, 50)], 160, 120)                   # class 7 does not exist
    with pytest.raises(ValueError, match="not in classes.txt"):
        train_detector.prepare(bad, 0.2, use_unreviewed=False)


def test_dataset_stats(tmp_path):
    base = tmp_path / "ds"
    _make_dataset(base, n=10)
    images = sorted((base / "images").iterdir())
    (base / "labels" / images[0].with_suffix(".txt").name).write_text("", encoding="utf-8")
    st = dataset_stats(base / "labels", images, 2)
    assert st["images"] == 10 and st["empty_images"] == 1 and sum(st["per_class"].values()) == 9
