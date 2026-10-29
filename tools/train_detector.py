"""Step 4: train (fine-tune) the detector on your labelled dataset and install it.

    python tools/train_detector.py all --dataset rickshaw --epochs 60            # prepare + train + install
    python tools/train_detector.py prepare --dataset rickshaw                    # only build the train/val split
    python tools/train_detector.py train   --dataset rickshaw --device cuda      # only train (after prepare)

* ``prepare``  splits the images into train / validation BY VIDEO (never by frame, which would leak
               near-identical frames into validation and fake a great score) and writes ``data.yaml``.
* ``train``    fine-tunes from ``--base`` (default ``models/yolo26n.pt``). On a CPU this takes hours for a
               real dataset: use a GPU (Colab) with the same dataset folder zipped.
* the result   is copied to ``models/<name>.pt`` and the exact config lines to use it are printed.
By default only images you have confirmed in tools/relabel.py are used (``--use-unreviewed`` to include the rest).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import dataset_stats, list_images, load_classes, read_sources, split_by_video  # noqa: E402
from src.logger import setup_logging  # noqa: E402

log = logging.getLogger("train_detector")
MIN_BOXES_PER_CLASS = 50          # below this a class is unlikely to be learned well


def prepare(base: Path, val_fraction: float, use_unreviewed: bool, seed: int = 0) -> Path:
    """Build ``<dataset>/yolo`` (train/val folders + data.yaml); returns the data.yaml path.

    Raises:
        ValueError / FileNotFoundError: if the dataset is incomplete (messages say what to do).
    """
    names = load_classes(base / "classes.txt")
    images = list_images(base / "images")
    reviewed_path = base / "reviewed.txt"
    reviewed = set(reviewed_path.read_text(encoding="utf-8").split()) if reviewed_path.is_file() else set()
    if not use_unreviewed:
        if not reviewed:
            raise ValueError("No image has been reviewed yet. Run tools/relabel.py (press n after checking each image), "
                             "or add --use-unreviewed to train on the unchecked pre-labels.")
        images = [im for im in images if im.name in reviewed]
    labelled = [im for im in images if (base / "labels" / (im.stem + ".txt")).is_file()]
    if len(labelled) < len(images):
        log.warning("%d images have no label file and are skipped", len(images) - len(labelled))
    if len(labelled) < 10:
        raise ValueError(f"Only {len(labelled)} labelled images; a useful model needs hundreds (aim for 300+ boxes per class).")
    stats = dataset_stats(base / "labels", labelled, len(names))
    if stats["out_of_range"]:
        raise ValueError(f"Label files use class ids {stats['out_of_range']} that are not in classes.txt {names}")
    for i, n in enumerate(names):
        count = stats["per_class"][i]
        flag = "  <-- too few; label more of this class" if count < MIN_BOXES_PER_CLASS else ""
        log.info("class %d %-12s %6d boxes%s", i, n, count, flag)
    log.info("%d images, %d without objects", stats["images"], stats["empty_images"])

    train, val = split_by_video(labelled, read_sources(base / "sources.csv"), val_fraction, seed=seed)
    out = base / "yolo"
    if out.exists():
        shutil.rmtree(out)
    for split, items in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for im in items:
            shutil.copy2(im, out / "images" / split / im.name)
            shutil.copy2(base / "labels" / (im.stem + ".txt"), out / "labels" / split / (im.stem + ".txt"))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    log.info("Split: %d train / %d validation images -> %s", len(train), len(val), yaml_path)
    return yaml_path


def train(data_yaml: Path, base_model: str, epochs: int, imgsz: int, device: str, batch: int, name: str, models_dir: Path,
          patience: int) -> Path:
    """Run Ultralytics training and install the best weights as ``models/<name>.pt``."""
    from ultralytics import YOLO

    from src.utils import resolve_device

    dev = resolve_device(device)
    log.info("Training %s on %s: %d epochs, imgsz %d, batch %d, device %s", base_model, data_yaml, epochs, imgsz, batch, dev)
    if dev == "cpu" and epochs > 30:
        log.warning("Training on a CPU is slow (hours for a real dataset). Consider Colab's free GPU with the same dataset folder.")
    model = YOLO(base_model)
    result = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=dev, batch=batch, name=name, patience=patience,
                         project=str(Path("runs").resolve()), exist_ok=True, plots=False, verbose=False, workers=0)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but {best} was not produced")
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / f"{name}.pt"
    shutil.copy2(best, target)
    return target


def print_config_help(weights: Path, names: list[str]) -> None:
    keep = [n for n in names if n.lower() not in ("plate",)]
    log.info("Installed %s", weights)
    log.info("To use it, in config.yaml set:\n  detector:\n    model: %s\n    classes: [%s]%s", weights.as_posix(), ", ".join(keep),
             "\n  (a plate model is a SEPARATE model: set plate.detector_model: " + weights.as_posix() + ")" if "plate" in [n.lower() for n in names] else "")
    log.info("Speed limits are per class name: add each new class (e.g. rickshaw) under speed.limits_kmh.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a dataset and fine-tune the detector")
    p.add_argument("command", choices=["prepare", "train", "all"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default="datasets")
    p.add_argument("--base", default="models/yolo26n.pt", help="weights to start from")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=20, help="stop early after this many epochs without improvement")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--name", help="name of the installed model (default: the dataset name)")
    p.add_argument("--use-unreviewed", action="store_true", help="also train on images you have not confirmed in relabel.py")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging({"logging": {"level": "INFO"}})
    base = Path(args.root) / args.dataset
    name = args.name or args.dataset
    try:
        if args.command in ("prepare", "all"):
            data_yaml = prepare(base, args.val_fraction, args.use_unreviewed, args.seed)
        else:
            data_yaml = base / "yolo" / "data.yaml"
            if not data_yaml.is_file():
                raise FileNotFoundError(f"{data_yaml} not found; run the 'prepare' command first")
        if args.command in ("train", "all"):
            weights = train(data_yaml, args.base, args.epochs, args.imgsz, args.device, args.batch, name, Path("models"), args.patience)
            print_config_help(weights, load_classes(base / "classes.txt"))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
