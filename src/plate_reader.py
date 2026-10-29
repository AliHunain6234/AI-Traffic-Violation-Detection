"""Two-stage licence-plate reading: plate detector -> crop -> enhance -> EasyOCR.

Stage 1 (``PlateDetector``) finds the plate inside a *vehicle crop*. No general-purpose
plate model ships with this project, so the default is ``NullPlateDetector`` (finds
nothing) and the reader reports an empty result with a note instead of guessing.

How to plug in a real plate detector (recommended for Pakistani plates):
  1. Collect/label plate images (one class: ``plate``), e.g. in Roboflow or CVAT,
     covering the provincial formats and motorcycle plates you care about.
  2. Fine-tune: ``yolo detect train data=plates.yaml model=yolo26n.pt imgsz=640 epochs=100``.
  3. Set ``plate.detector_model: runs/detect/train/weights/best.pt`` in config.yaml.
  4. Add province regexes under ``plate.format_patterns`` to validate the OCR text.

OCR never raises into the pipeline: any failure yields an empty ``PlateResult``.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from .utils import is_valid_frame, pad_box, resolve_device, safe_crop, sharpness

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
# EasyOCR-style output: [(4-point polygon, text, confidence), ...]
OcrFn = Callable[[np.ndarray], list[tuple[Any, str, float]]]


@dataclass
class PlateResult:
    """Outcome of reading one vehicle crop. Empty ``text`` means nothing was read."""

    text: str = ""
    conf: float = 0.0
    box: Optional[Box] = None            # plate box in vehicle-crop coordinates
    crop: Optional[np.ndarray] = field(default=None, repr=False)
    enhanced: Optional[np.ndarray] = field(default=None, repr=False)
    format_name: str = ""                # name of the matching format pattern, if any
    note: str = ""                       # reason the result is empty / low quality

    @property
    def found(self) -> bool:
        return bool(self.text)


# --------------------------------------------------------------------------- #
# Stage 1: plate detectors
# --------------------------------------------------------------------------- #
class PlateDetector(ABC):
    """Finds licence plates inside a vehicle crop."""

    @abstractmethod
    def detect(self, vehicle_bgr: np.ndarray) -> list[tuple[Box, float]]:
        """Return ``[(x1, y1, x2, y2), confidence]`` for each plate found."""


class NullPlateDetector(PlateDetector):
    """Placeholder used when no plate model is configured: finds nothing."""

    def detect(self, vehicle_bgr: np.ndarray) -> list[tuple[Box, float]]:
        return []


class YoloPlateDetector(PlateDetector):
    """Ultralytics YOLO model fine-tuned to detect plates (see module docstring)."""

    def __init__(self, weights: str, conf: float = 0.30, device: str = "auto") -> None:
        from ultralytics import YOLO

        if not Path(weights).is_file():
            raise FileNotFoundError(f"Plate detector weights not found: {weights}")
        self.model = YOLO(weights)
        self.conf = conf
        self.device = resolve_device(device)

    def detect(self, vehicle_bgr: np.ndarray) -> list[tuple[Box, float]]:
        results = self.model.predict(vehicle_bgr, conf=self.conf, device=self.device, verbose=False)
        out: list[tuple[Box, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
        return out


# --------------------------------------------------------------------------- #
# Stage 2: enhancement
# --------------------------------------------------------------------------- #
def enhance_plate(
    crop_bgr: np.ndarray,
    upscale: float = 3.0,
    denoise: bool = True,
    clahe: bool = True,
    max_width: int = 640,
) -> np.ndarray:
    """Upscale, denoise and contrast-stretch a plate crop for OCR; returns grayscale."""
    h, w = crop_bgr.shape[:2]
    factor = max(1.0, min(float(upscale), max_width / max(1, w)))
    img = cv2.resize(crop_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
class PlateReader:
    """Detects, enhances and OCRs the plate of a vehicle crop.

    Args:
        cfg: Full config (uses the ``plate`` section).
        detector: Stage-1 plate detector (defaults to the one named in config).
        ocr_fn: Optional replacement for EasyOCR (tests, other OCR engines).
    """

    def __init__(
        self,
        cfg: dict,
        detector: Optional[PlateDetector] = None,
        ocr_fn: Optional[OcrFn] = None,
    ) -> None:
        p = cfg.get("plate", {})
        ocr = p.get("ocr", {})
        self.enabled = bool(p.get("enabled", True))
        self.fallback = bool(p.get("fallback_to_vehicle_crop", False))
        self.fallback_region = tuple(p.get("fallback_region", [0.55, 1.0]))
        self.upscale = float(p.get("upscale_factor", 3))
        self.denoise = bool(p.get("denoise", True))
        self.clahe = bool(p.get("clahe", True))
        self.min_width = int(p.get("min_plate_px_width", 40))
        self.languages = list(ocr.get("languages", ["en"]))
        self.gpu_pref = ocr.get("gpu", "auto")
        self.allowlist = ocr.get("allowlist") or None
        self.min_conf = float(ocr.get("min_conf", 0.30))
        self.patterns = self._compile_patterns(p.get("format_patterns") or [])
        self.detector = detector or self._build_detector(p, cfg)
        self._ocr_fn = ocr_fn
        self._easyocr_reader: Any = None

    # --------------------------------------------------------------- building
    @staticmethod
    def _build_detector(p: dict, cfg: dict) -> PlateDetector:
        weights = p.get("detector_model")
        if not weights:
            log.info("No plate detector configured: plate reading uses the placeholder hook")
            return NullPlateDetector()
        device = cfg.get("detector", {}).get("device", "auto")
        return YoloPlateDetector(str(weights), float(p.get("detector_conf", 0.30)), device)

    @staticmethod
    def _compile_patterns(items: Sequence[Any]) -> list[tuple[str, re.Pattern]]:
        """Accept ``"regex"`` or ``{name: ..., regex: ...}`` entries."""
        out: list[tuple[str, re.Pattern]] = []
        for i, item in enumerate(items):
            name, rx = (item.get("name", f"pattern{i}"), item["regex"]) if isinstance(item, dict) else (f"pattern{i}", item)
            out.append((str(name), re.compile(rx)))
        return out

    def _ocr(self, image: np.ndarray) -> list[tuple[Any, str, float]]:
        if self._ocr_fn is not None:
            return self._ocr_fn(image)
        if self._easyocr_reader is None:
            import easyocr  # lazy: first use downloads OCR models

            gpu = (resolve_device(self.gpu_pref) == "cuda") if self.gpu_pref == "auto" else bool(self.gpu_pref)
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
        kwargs: dict[str, Any] = {"detail": 1, "paragraph": False}
        if self.allowlist:
            kwargs["allowlist"] = self.allowlist
        return self._easyocr_reader.readtext(image, **kwargs)

    # ------------------------------------------------------------------- read
    def read(self, vehicle_bgr: Optional[np.ndarray]) -> PlateResult:
        """Read the plate of one vehicle crop. Never raises."""
        if not self.enabled:
            return PlateResult(note="plate reading disabled")
        if not is_valid_frame(vehicle_bgr):
            return PlateResult(note="empty vehicle crop")
        try:
            return self._read(vehicle_bgr)  # type: ignore[arg-type]
        except Exception:  # OCR/model failures must never break the pipeline
            log.exception("Plate reading failed")
            return PlateResult(note="plate reading error")

    def read_best(self, crops: Sequence[np.ndarray], max_crops: int = 3) -> PlateResult:
        """Read several crops of the same vehicle (sharpest first); keep the best result."""
        valid = [c for c in crops if is_valid_frame(c)]
        valid.sort(key=sharpness, reverse=True)
        best = PlateResult(note="no crops")
        for crop in valid[:max_crops]:
            res = self.read(crop)
            if res.found and (not best.found or res.conf > best.conf):
                best = res
            elif not best.found:
                best = res
        return best

    def _read(self, vehicle: np.ndarray) -> PlateResult:
        h, w = vehicle.shape[:2]
        found = sorted(self.detector.detect(vehicle), key=lambda d: d[1], reverse=True)
        if found:
            box = found[0][0]
        elif self.fallback:
            lo, hi = self.fallback_region
            box = (0.0, h * lo, float(w), h * hi)
        else:
            return PlateResult(note="no plate detected (no plate model or plate not visible)")

        x1, y1, x2, y2 = pad_box(box, 0.10, w, h)
        crop = vehicle[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < self.min_width:
            return PlateResult(box=box, note=f"plate too small (<{self.min_width}px wide)")

        enhanced = enhance_plate(crop, self.upscale, self.denoise, self.clahe)
        text, conf = self._join(self._ocr(enhanced))
        if not text:
            return PlateResult(box=box, crop=crop.copy(), enhanced=enhanced, note="OCR returned nothing")
        res = PlateResult(text=text, conf=conf, box=box, crop=crop.copy(), enhanced=enhanced)
        res.format_name = self._match_format(text)
        if self.patterns and not res.format_name:
            res.note = "text does not match any configured plate format"
        return res

    # --------------------------------------------------------------- parsing
    def _join(self, items: Sequence[tuple[Any, str, float]]) -> tuple[str, float]:
        """Order OCR fragments (lines top-to-bottom, left-to-right) and merge them."""
        frags = []
        for poly, text, conf in items:
            text = self.normalize(text)
            if not text or float(conf) < self.min_conf:
                continue
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            frags.append((pts[:, 0].mean(), pts[:, 1].mean(), np.ptp(pts[:, 1]), text, float(conf)))
        if not frags:
            return "", 0.0
        frags.sort(key=lambda f: f[1])
        median_h = float(np.median([f[2] for f in frags])) or 1.0
        lines: list[list[tuple]] = []
        for f in frags:
            if lines and abs(f[1] - np.mean([g[1] for g in lines[-1]])) <= 0.6 * median_h:
                lines[-1].append(f)
            else:
                lines.append([f])
        ordered = [f for line in lines for f in sorted(line, key=lambda g: g[0])]
        text = " ".join(f[3] for f in ordered)
        weights = np.array([len(f[3]) for f in ordered], dtype=float)
        conf = float(np.average([f[4] for f in ordered], weights=weights))
        return text, conf

    @staticmethod
    def normalize(text: str) -> str:
        """Upper-case, strip stray characters and collapse whitespace."""
        text = re.sub(r"[^A-Za-z0-9\- ]", "", text or "").upper()
        return re.sub(r"\s+", " ", text).strip()

    def _match_format(self, text: str) -> str:
        compact = re.sub(r"[\s\-]", "", text)
        for name, rx in self.patterns:
            if rx.fullmatch(text) or rx.fullmatch(compact):
                return name
        return ""
