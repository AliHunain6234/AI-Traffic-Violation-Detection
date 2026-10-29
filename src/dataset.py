"""Dataset helpers for fine-tuning the detector (rickshaw, plate, ...): YOLO label I/O, splits, editing.

Layout produced by the tools (``datasets/<name>/``)::

    images/            frames extracted from your videos (tools/extract_frames.py)
    labels/            one YOLO .txt per image: "class cx cy w h" (normalised)   (tools/prelabel.py, relabel.py)
    classes.txt        one class name per line; the line number is the class id
    sources.csv        image,video,frame   (used to split train/val BY VIDEO, never by frame)
    reviewed.txt       images a human has checked in tools/relabel.py
    yolo/              the train/val split + data.yaml for Ultralytics     (tools/train_detector.py prepare)
"""
from __future__ import annotations

import csv
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_BOX_PX = 6


@dataclass
class Box:
    """One labelled object in pixel coordinates."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


# ------------------------------------------------------------------ label I/O
def read_labels(path: Path, width: int, height: int) -> list[Box]:
    """Read a YOLO label file into pixel boxes (a missing file means no objects)."""
    boxes: list[Box] = []
    if not path.is_file():
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            log.warning("%s:%d is not a valid YOLO label line: %r", path, line_no, line)
            continue
        boxes.append(Box(c, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def write_labels(path: Path, boxes: Iterable[Box], width: int, height: int) -> None:
    """Write pixel boxes as a YOLO label file (clipped to the image; degenerate boxes are dropped)."""
    lines = []
    for b in boxes:
        x1, y1 = max(0.0, min(width, b.x1)), max(0.0, min(height, b.y1))
        x2, y2 = max(0.0, min(width, b.x2)), max(0.0, min(height, b.y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{b.cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_classes(path: Path) -> list[str]:
    """Class names, one per line (blank lines ignored)."""
    if not path.is_file():
        raise FileNotFoundError(f"classes file not found: {path}")
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(set(n.lower() for n in names)) != len(names):
        raise ValueError(f"duplicate class names in {path}: {names}")
    return names


def save_classes(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []


# ------------------------------------------------------------------ sources / split
def read_sources(path: Path) -> dict[str, str]:
    """``image file name -> source video`` from sources.csv (empty if the file is missing)."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["image"]: row["video"] for row in csv.DictReader(fh)}


def append_sources(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["image", "video", "frame"])
        w.writerows(rows)


def frame_number(name: str) -> int:
    """Frame index encoded at the end of a file name like ``clip_000123.jpg`` (0 if absent)."""
    m = re.search(r"_(\d+)\.[A-Za-z0-9]+$", name)
    return int(m.group(1)) if m else 0


def split_by_video(images: Sequence[Path], sources: dict[str, str], val_fraction: float = 0.2,
                   buffer: int = 5, seed: int = 0) -> tuple[list[Path], list[Path]]:
    """Split into train / val without leakage.

    With two or more videos, whole videos go to validation (so near-identical neighbouring frames never
    end up on both sides). With one video, the LAST ``val_fraction`` of its frames is validation and
    ``buffer`` frames in between are dropped.

    Raises:
        ValueError: If a non-empty validation split cannot be made.
    """
    if len(images) < 2:
        raise ValueError("need at least 2 images to make a train/val split")
    groups: dict[str, list[Path]] = {}
    for im in images:
        groups.setdefault(sources.get(im.name, im.name.rsplit("_", 1)[0]), []).append(im)
    for g in groups.values():
        g.sort(key=lambda p: frame_number(p.name))
    target = max(1, round(len(images) * val_fraction))
    if len(groups) >= 2:
        order = sorted(groups)
        random.Random(seed).shuffle(order)
        val: list[Path] = []
        for name in order:
            if len(val) >= target or len(val) + len(groups[name]) >= len(images):
                continue
            val += groups[name]
        if not val:                                       # every video is huge: fall back to the smallest one
            val = groups[min(groups, key=lambda k: len(groups[k]))]
        val_set = set(val)
        return [im for im in images if im not in val_set], sorted(val)
    only = next(iter(groups.values()))
    cut = len(only) - target
    if cut <= buffer:
        raise ValueError("too few frames in the single video to keep a validation block; extract more frames")
    return only[: cut - buffer], only[cut:]


# ------------------------------------------------------------------ statistics
def dataset_stats(labels_dir: Path, images: Sequence[Path], n_classes: int) -> dict:
    """Counts per class and how many images have no objects (background images)."""
    counts: Counter = Counter()
    empty = 0
    for im in images:
        lp = labels_dir / (im.stem + ".txt")
        ids = [int(ln.split()[0]) for ln in lp.read_text(encoding="utf-8").splitlines() if ln.split()] if lp.is_file() else []
        counts.update(ids)
        empty += 0 if ids else 1
    return {"images": len(images), "empty_images": empty, "per_class": {i: counts.get(i, 0) for i in range(n_classes)},
            "out_of_range": sorted(c for c in counts if c >= n_classes or c < 0)}


# ------------------------------------------------------------------ interactive editing (logic only)
class LabelEditor:
    """The state machine behind tools/relabel.py: boxes of one image, selection, edits and undo."""

    def __init__(self, boxes: Sequence[Box], n_classes: int) -> None:
        self.boxes: list[Box] = [Box(b.cls, b.x1, b.y1, b.x2, b.y2) for b in boxes]
        self.n_classes = n_classes
        self.selected: Optional[int] = None
        self.current_class = 0
        self._undo: list[list[Box]] = []
        self.dirty = False

    def _snapshot(self) -> None:
        self._undo.append([Box(b.cls, b.x1, b.y1, b.x2, b.y2) for b in self.boxes])
        self.dirty = True

    def hit_test(self, x: float, y: float) -> Optional[int]:
        """Index of the smallest box under the point (so a small box inside a big one is reachable)."""
        hits = [(b.area, i) for i, b in enumerate(self.boxes) if b.contains(x, y)]
        return min(hits)[1] if hits else None

    def select_at(self, x: float, y: float) -> Optional[int]:
        self.selected = self.hit_test(x, y)
        return self.selected

    def press_class(self, cls: int) -> None:
        """A class key: re-labels the selected box, otherwise sets the class for the next new box."""
        if not 0 <= cls < self.n_classes:
            return
        if self.selected is not None:
            self._snapshot()
            self.boxes[self.selected].cls = cls
        self.current_class = cls

    def delete_selected(self) -> bool:
        if self.selected is None:
            return False
        self._snapshot()
        del self.boxes[self.selected]
        self.selected = None
        return True

    def add_box(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Add a box of the current class; tiny drags (a click) are ignored."""
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            return False
        self._snapshot()
        self.boxes.append(Box(self.current_class, x1, y1, x2, y2))
        self.selected = len(self.boxes) - 1
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.boxes = self._undo.pop()
        self.selected = None
        return True
