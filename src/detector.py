"""Vehicle detection with Ultralytics YOLO (YOLO26 / YOLOv8, set in config).

``VehicleDetector`` owns the model. ``src.tracker.VehicleTracker`` reuses the same
model object for tracking, so only one set of weights is ever loaded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .utils import is_valid_frame, resolve_device

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detected (and possibly tracked) vehicle in a frame."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    conf: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


def merge_names(model_names: dict[int, str], overrides: dict[int, str]) -> dict[int, str]:
    """Class id -> name. A model's OWN names are used as they are (every YOLO .pt carries them, including
    models you fine-tune); ``detector.class_names`` is only a fallback for a model without names, because
    mixing the two would rename or duplicate classes of a fine-tuned model."""
    named = {int(k): str(v) for k, v in model_names.items()}
    return named if named else dict(overrides)


def resolve_classes(items, names: dict[int, str]) -> Optional[list[int]]:
    """Turn the ``detector.classes`` setting into class ids.

    Accepts ids (``[2, 3, 5, 7]``), class names (``[car, motorcycle, rickshaw]``) or a mix, so a
    fine-tuned model can be configured by name. ``None`` / empty means "every class the model has".

    Raises:
        ValueError: If a name is not a class of the model (the message lists the valid ones).
    """
    if items is None or (isinstance(items, (list, tuple)) and len(items) == 0):
        return None
    by_name = {str(v).lower(): int(k) for k, v in names.items()}
    out: list[int] = []
    for item in items:
        if isinstance(item, int) or (isinstance(item, str) and item.strip().lstrip("-").isdigit()):
            out.append(int(item))
        else:
            key = str(item).strip().lower()
            if key not in by_name:
                raise ValueError(f"detector.classes: {item!r} is not a class of this model. Available: {sorted(by_name)}")
            out.append(by_name[key])
    return sorted(set(out))


def openvino_gpu_available() -> bool:
    """True when OpenVINO is installed and sees an Intel GPU (e.g. Iris Xe)."""
    try:
        import openvino as ov
        return "GPU" in ov.Core().available_devices
    except Exception:
        return False


def prepare_openvino(model_name: str, imgsz: int) -> str:
    """Folder with an OpenVINO export of ``model_name`` at ``imgsz`` (exported once, ~20 s, then reused).

    The export has a fixed input size, so each size gets its own folder next to the weights:
    ``models/yolo26n.pt`` -> ``models/yolo26n_640_openvino_model/`` (Ultralytics recognises the format by
    that ``_openvino_model`` suffix).
    """
    import shutil
    from pathlib import Path

    from ultralytics import YOLO

    weights = Path(model_name)
    target = weights.with_name(f"{weights.stem}_{imgsz}_openvino_model")
    if (target / f"{weights.stem}.xml").is_file():
        return str(target)
    log.info("Exporting %s to OpenVINO at %d px (one-time, about 20 s)...", weights.name, imgsz)
    exported = YOLO(str(weights)).export(format="openvino", imgsz=imgsz, half=False, verbose=False)
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(exported), str(target))
    return str(target)


class VehicleDetector:
    """Wraps an Ultralytics YOLO model with the detection settings from config.

    Args:
        cfg: Full config (uses the ``detector`` section).

    Note:
        Constructing this loads the weights; Ultralytics downloads known model
        names (e.g. ``yolo26n.pt``) on first use if they are not on disk.
    """

    def __init__(self, cfg: dict) -> None:
        from ultralytics import YOLO  # imported lazily: keeps module import cheap

        det = cfg.get("detector", {})
        self.model_name: str = str(det.get("model", "yolo26n.pt"))
        self.device: str = resolve_device(det.get("device", "auto"))
        half = det.get("half", "auto")
        self.half: bool = (self.device.startswith("cuda")) if half == "auto" else bool(half)
        self.conf = float(det.get("conf", 0.35))
        self.iou = float(det.get("iou", 0.5))
        self.imgsz = int(det.get("imgsz", 960))
        self.max_det = int(det.get("max_det", 300))

        self.backend = "torch"
        self.model = None
        wanted = str(det.get("backend", "auto")).lower()           # auto | torch | openvino
        if (wanted in ("auto", "openvino") and self.device == "cpu" and self.model_name.endswith(".pt")
                and openvino_gpu_available()):
            try:                                                   # Intel GPU: ~2x faster than the CPU and leaves the CPU free
                self.model = YOLO(prepare_openvino(self.model_name, self.imgsz), task="detect")
                self.device, self.half, self.backend = "intel:gpu", False, "openvino-gpu"
            except Exception as exc:
                log.warning("OpenVINO GPU backend unavailable (%s); using PyTorch on the CPU", exc)
                self.model = None
        elif wanted == "openvino":
            log.warning("detector.backend is openvino but no Intel GPU / OpenVINO was found; using PyTorch")
        log.info("Loading model %s on %s (%s, half=%s)", self.model_name, self.device, self.backend, self.half)
        if self.model is None:
            self.model = YOLO(self.model_name)
        model_names: dict[int, str] = dict(getattr(self.model, "names", {}) or {})
        overrides = {int(k): str(v) for k, v in (det.get("class_names") or {}).items()}
        self.names: dict[int, str] = merge_names(model_names, overrides)
        self.classes: Optional[list[int]] = resolve_classes(det.get("classes", [2, 3, 5, 7]), self.names)

    # ------------------------------------------------------------------ public
    @property
    def predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments shared by ``predict`` and ``track`` calls."""
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.classes is not None:
            kwargs["classes"] = self.classes
        if self.half:  # passing half=False triggers a deprecation warning on newer Ultralytics
            kwargs["half"] = True
        return kwargs

    def detect(self, frame: Optional[np.ndarray]) -> list[Detection]:
        """Detect vehicles in one frame (no tracking). Empty/invalid frames give ``[]``."""
        if not is_valid_frame(frame):
            return []
        results = self.model.predict(frame, **self.predict_kwargs)
        return self.parse(results[0]) if results else []

    def warmup(self, frame: np.ndarray, runs: int = 2) -> None:
        """Run a few throw-away inferences so timing excludes one-off start-up cost."""
        for _ in range(runs):
            self.detect(frame)

    def class_name(self, class_id: int) -> str:
        """Display name for a class id."""
        return self.names.get(int(class_id), str(class_id))

    # ----------------------------------------------------------------- parsing
    def parse(self, result: Any) -> list[Detection]:
        """Convert an Ultralytics ``Results`` object into ``Detection`` objects.

        Works for both detect and track results; ``track_id`` is ``None`` when the
        tracker has not (yet) assigned an ID.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [
            Detection(
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                conf=float(c),
                class_id=int(k),
                class_name=self.class_name(int(k)),
                track_id=int(i) if i is not None else None,
            )
            for b, c, k, i in zip(xyxy, conf, cls, ids)
        ]
