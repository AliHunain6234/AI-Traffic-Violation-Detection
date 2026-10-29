"""Step 2: let the current model draw first-draft boxes, so you only have to correct them.

    python tools/prelabel.py --dataset rickshaw --classes car motorcycle bus truck rickshaw \\
        --map car=car motorcycle=motorcycle bus=bus truck=truck

The stock COCO model has no rickshaw class and calls rickshaws ``car`` / ``truck``. This writes them as
``car`` / ``truck`` in YOLO label files; you then re-label them as ``rickshaw`` in ``tools/relabel.py``
(one key press per box), which is much faster than drawing every box.

* ``--classes``  the class list of the NEW model, in order (this fixes the class ids).
* ``--map``      ``source_class=new_class`` pairs; source classes not mapped are dropped.
                 Default: every source class whose name is also in ``--classes``.
Images that already have a label file are left alone (your corrections are never overwritten) unless
``--overwrite`` is given.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.dataset import Box, list_images, load_classes, save_classes, write_labels  # noqa: E402
from src.logger import setup_logging  # noqa: E402
from src.utils import load_config  # noqa: E402

log = logging.getLogger("prelabel")


def parse_map(pairs: list[str]) -> dict[str, str]:
    """``["truck=truck", "car=car"]`` -> ``{"truck": "truck", "car": "car"}`` (names are lower-cased)."""
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--map entries look like source=target, got {p!r}")
        a, b = p.split("=", 1)
        out[a.strip().lower()] = b.strip().lower()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-label images with the current detector")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default="datasets")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default="models/yolo26n.pt", help="detector used for the first draft")
    p.add_argument("--classes", nargs="+", help="class list of the new model, in order (required the first time)")
    p.add_argument("--map", nargs="*", default=None, help="source=target pairs (default: same names)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--overwrite", action="store_true", help="redo images that already have labels")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging({"logging": {"level": "INFO"}})
    base = Path(args.root) / args.dataset
    images = list_images(base / "images")
    if not images:
        log.error("No images in %s (run tools/extract_frames.py first)", base / "images")
        return 2
    classes_path = base / "classes.txt"
    if args.classes:
        names = [c.strip() for c in args.classes]
        if classes_path.is_file() and load_classes(classes_path) != names:
            log.error("classes.txt already exists with different classes %s; delete it only if you also discard the labels", load_classes(classes_path))
            return 2
        save_classes(classes_path, names)
    try:
        names = load_classes(classes_path)
    except FileNotFoundError:
        log.error("Pass --classes the first time (e.g. --classes car motorcycle bus truck rickshaw)")
        return 2
    target = {n.lower(): i for i, n in enumerate(names)}
    try:
        mapping = parse_map(args.map) if args.map else {n.lower(): n.lower() for n in names}
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    bad = sorted(set(mapping.values()) - set(target))
    if bad:
        log.error("--map targets %s are not in the class list %s", bad, names)
        return 2

    from src.detector import VehicleDetector     # heavy import: only when we really run the model

    cfg = load_config(args.config, {"detector.model": args.model, "detector.conf": args.conf, "detector.imgsz": args.imgsz,
                                    "detector.classes": []})                  # [] = every class the model knows
    det = VehicleDetector(cfg)
    labels_dir = base / "labels"
    done = skipped = boxes_total = 0
    for im in images:
        lp = labels_dir / (im.stem + ".txt")
        if lp.is_file() and not args.overwrite:
            skipped += 1
            continue
        frame = cv2.imread(str(im))
        if frame is None:
            log.warning("Cannot read %s", im)
            continue
        h, w = frame.shape[:2]
        boxes = [Box(target[mapping[d.class_name.lower()]], *d.box) for d in det.detect(frame) if d.class_name.lower() in mapping]
        write_labels(lp, boxes, w, h)
        done += 1
        boxes_total += len(boxes)
    log.info("Pre-labelled %d images (%d boxes), left %d existing labels untouched. Next: python tools/relabel.py --dataset %s",
             done, boxes_total, skipped, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
