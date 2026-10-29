"""Validation of what a user draws for a camera: stop line, signal ROI, speed calibration.

One place for the rules, shared by the command-line tools (tools/define_*.py) and the dashboard's
Setup page, so both accept and reject exactly the same inputs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .speed import HomographySpeed, TwoLineSpeed

MIN_LINE_PX = 10
MIN_ROI_PX = 6


def validate_line(points: Sequence[Sequence[float]]) -> None:
    """Raise ``ValueError`` unless the points form a usable stop line."""
    if len(points) != 2:
        raise ValueError("A stop line needs exactly two points")
    (x1, y1), (x2, y2) = points
    if float(np.hypot(x2 - x1, y2 - y1)) < MIN_LINE_PX:
        raise ValueError(f"Line is shorter than {MIN_LINE_PX}px; pick two distinct points")


def validate_roi(roi: Sequence[float], frame_size: tuple[int, int]) -> list[int]:
    """Clip ``[x, y, w, h]`` to the frame; raise ``ValueError`` if what is left is unusable."""
    if len(roi) != 4:
        raise ValueError("A signal ROI is [x, y, w, h]")
    w_f, h_f = frame_size
    x, y, w, h = (int(round(v)) for v in roi)
    x, y = max(0, x), max(0, y)
    w, h = min(w, w_f - x), min(h, h_f - y)
    if w < MIN_ROI_PX or h < MIN_ROI_PX:
        raise ValueError(f"ROI must be at least {MIN_ROI_PX}x{MIN_ROI_PX}px inside the frame, got {list(roi)}")
    return [x, y, w, h]


def rect_world_points(width: float, length: float) -> list[list[float]]:
    """World points for corners near-left, near-right, far-right, far-left of a width x length rectangle (metres)."""
    if width <= 0 or length <= 0:
        raise ValueError("Rectangle width and length must be positive")
    return [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]]


def build_calibration(method: str, points: Sequence[Sequence[float]], distance_m: Optional[float],
                      world: Optional[Sequence[Sequence[float]]]) -> dict:
    """Validate the inputs with the real estimators and return the dict stored under ``speed:`` in the config.

    Raises:
        ValueError: If the inputs cannot form a valid calibration.
    """
    pts = [[int(round(x)), int(round(y))] for x, y in points]
    if method == "lines":
        if len(pts) != 4:
            raise ValueError("The lines method needs 4 points (2 per line)")
        cal = {"method": "lines", "lines": [pts[:2], pts[2:]], "distance_m": float(distance_m or 0)}
        TwoLineSpeed(cal["lines"][0], cal["lines"][1], cal["distance_m"])
        return cal
    if method == "homography":
        if len(pts) != 4 or not world or len(world) != 4:
            raise ValueError("The homography method needs 4 image points and 4 world points")
        cal = {"method": "homography", "image_points": pts, "world_points": [[float(a), float(b)] for a, b in world]}
        HomographySpeed(cal["image_points"], cal["world_points"])
        return cal
    raise ValueError(f"Unknown method {method!r}")
