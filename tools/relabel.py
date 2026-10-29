"""Step 3: review and correct the labels with a keyboard/mouse editor (no external labelling tool needed).

    python tools/relabel.py --dataset rickshaw            # all images
    python tools/relabel.py --dataset rickshaw --todo     # only images you have not confirmed yet

Mouse   click a box ............ select it            drag on empty road ...... draw a new box
Keys    0-9 ................... re-label the selected box (or choose the class of the next new box)
        d / Delete ............ delete the selected box     u ............ undo
        n / Space / Right ..... save, mark as reviewed, next image     p / Left ... save, previous image
        s ..................... save      h ... show/hide help      q / Esc ... save and quit
Typical rickshaw session: select a box that the stock model called "car" or "truck" and press the rickshaw
key. Boxes that are wrong are deleted; vehicles the model missed are drawn.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.dataset import LabelEditor, list_images, load_classes, read_labels, write_labels  # noqa: E402
from src.logger import setup_logging  # noqa: E402

log = logging.getLogger("relabel")
WINDOW = "relabel"
PALETTE = [(60, 76, 231), (60, 160, 255), (80, 200, 90), (230, 160, 40), (200, 70, 190), (200, 200, 60), (40, 200, 230), (150, 150, 150), (90, 90, 230), (230, 120, 120)]


def color_for(cls: int) -> tuple[int, int, int]:
    return PALETTE[cls % len(PALETTE)]


def render(img: np.ndarray, ed: LabelEditor, names: list[str], scale: float, index: int, total: int, name: str,
           reviewed: bool, drag: Optional[tuple[int, int, int, int]], show_help: bool) -> np.ndarray:
    """Draw boxes, class legend and status on a copy of the (already scaled) image."""
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img.copy()
    for i, b in enumerate(ed.boxes):
        c = color_for(b.cls)
        p1, p2 = (int(b.x1 * scale), int(b.y1 * scale)), (int(b.x2 * scale), int(b.y2 * scale))
        cv2.rectangle(view, p1, p2, c, 3 if i == ed.selected else 2)
        label = names[b.cls] if b.cls < len(names) else f"?{b.cls}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(view, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 6, p1[1]), c, -1)
        cv2.putText(view, label, (p1[0] + 3, max(th, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if i == ed.selected:
            cv2.rectangle(view, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), (255, 255, 255), 1)
    if drag:
        cv2.rectangle(view, (drag[0], drag[1]), (drag[2], drag[3]), color_for(ed.current_class), 1)
    legend = [f"{i}: {n}" + ("   <- new boxes" if i == ed.current_class else "") for i, n in enumerate(names[:10])]
    status = f"{index + 1}/{total}  {name}  {'[reviewed]' if reviewed else '[not reviewed]'}  boxes: {len(ed.boxes)}"
    lines = [status] + legend + (["click=select drag=new 0-9=class d=delete u=undo n/space=next p=prev s=save q=quit"] if show_help else ["h = help"])
    y = 20
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(view, (4, y - th - 4), (tw + 12, y + 4), (25, 25, 25), -1)
        cv2.putText(view, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 10
    return view


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Review and correct YOLO labels")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default="datasets")
    p.add_argument("--todo", action="store_true", help="only images not yet marked as reviewed")
    p.add_argument("--start", type=int, default=0, help="start at this image number (1-based)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging({"logging": {"level": "INFO"}})
    base = Path(args.root) / args.dataset
    images = list_images(base / "images")
    try:
        names = load_classes(base / "classes.txt")
    except FileNotFoundError:
        log.error("%s has no classes.txt; run tools/prelabel.py --classes ... first", base)
        return 2
    if not images:
        log.error("No images in %s", base / "images")
        return 2
    reviewed_path = base / "reviewed.txt"
    reviewed = set(reviewed_path.read_text(encoding="utf-8").split()) if reviewed_path.is_file() else set()
    queue = [im for im in images if not (args.todo and im.name in reviewed)]
    if not queue:
        log.info("Everything is already reviewed.")
        return 0

    state = {"i": max(0, min(len(queue) - 1, args.start - 1)), "ed": None, "img": None, "scale": 1.0, "drag0": None, "drag": None, "help": False}

    def load(i: int) -> None:
        img = cv2.imread(str(queue[i]))
        if img is None:
            raise FileNotFoundError(queue[i])
        h, w = img.shape[:2]
        state["img"], state["scale"] = img, min(1.0, 1400 / w, 900 / h)
        cur = state["ed"].current_class if state["ed"] else 0
        state["ed"] = LabelEditor(read_labels(base / "labels" / (queue[i].stem + ".txt"), w, h), len(names))
        state["ed"].current_class = cur

    def save(mark: bool) -> None:
        ed, img, im = state["ed"], state["img"], queue[state["i"]]
        if ed.dirty or not (base / "labels" / (im.stem + ".txt")).is_file():
            write_labels(base / "labels" / (im.stem + ".txt"), ed.boxes, img.shape[1], img.shape[0])
            ed.dirty = False
        if mark and im.name not in reviewed:
            reviewed.add(im.name)
            with reviewed_path.open("a", encoding="utf-8") as fh:
                fh.write(im.name + "\n")

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        ed, s = state["ed"], state["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if ed.select_at(x / s, y / s) is None:
                state["drag0"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag0"]:
            state["drag"] = (*state["drag0"], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag0"]:
            x0, y0 = state["drag0"]
            ed.add_box(x0 / s, y0 / s, x / s, y / s)
            state["drag0"] = state["drag"] = None

    try:
        load(state["i"])
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            im = queue[state["i"]]
            cv2.imshow(WINDOW, render(state["img"], state["ed"], names, state["scale"], state["i"], len(queue), im.name,
                                      im.name in reviewed, state["drag"], state["help"]))
            key = cv2.waitKeyEx(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                save(False)
                break
            if key == -1:
                continue
            k = key & 0xFF
            if k in (ord("q"), 27):
                save(False)
                break
            if ord("0") <= k <= ord("9"):
                state["ed"].press_class(k - ord("0"))
            elif k in (ord("d"), 0x2E) or key == 0x2E0000:       # d or Delete
                state["ed"].delete_selected()
            elif k == ord("u"):
                state["ed"].undo()
            elif k == ord("s"):
                save(False)
            elif k == ord("h"):
                state["help"] = not state["help"]
            elif k in (ord("n"), ord(" ")) or key == 0x270000:    # next / space / right arrow
                save(True)
                if state["i"] + 1 >= len(queue):
                    log.info("Reached the last image.")
                    break
                state["i"] += 1
                load(state["i"])
            elif k == ord("p") or key == 0x250000:                # previous / left arrow
                save(False)
                state["i"] = max(0, state["i"] - 1)
                load(state["i"])
    except cv2.error as exc:
        log.error("No display available (%s). The labels are plain YOLO .txt files; any tool (CVAT, Roboflow) can edit them.", exc)
        return 3
    finally:
        cv2.destroyAllWindows()
    log.info("Reviewed %d of %d images. Next: python tools/train_detector.py all --dataset %s", len(reviewed), len(images), args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
