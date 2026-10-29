"""Camera-shake compensation: align every frame to the first (reference) frame.

Stop lines, signal ROIs, violation zones and speed calibrations are fixed pixel coordinates, so a
camera that drifts or shakes (handheld clip, windy pole, loose bracket) moves the scene under them.
``CameraStabilizer`` estimates the camera motion against the reference frame and warps each frame
back, so all geometry stays valid.

Method: corners found once on the reference frame are tracked into each new frame with pyramidal
Lucas-Kanade optical flow; a RANSAC similarity transform (shift + small rotation + zoom) is fitted.
Moving vehicles are outliers and are ignored, and boxes passed in ``ignore_boxes`` are excluded
explicitly. If too few points agree, the last good transform is reused.

Limits: it corrects small to moderate shake, not a camera that is re-aimed or driven around.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraStabilizer:
    """Warps frames so the static background stays where it was in the first frame.

    Args:
        params: The ``stabilization`` config section.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        p = params or {}
        self.max_width = int(p.get("max_width", 640))          # tracking is done on a downscaled copy for speed
        self.min_inliers = int(p.get("min_inliers", 12))
        self.max_shift_frac = float(p.get("max_shift_frac", 0.30))
        self.max_scale_change = float(p.get("max_scale_change", 0.12))
        self.max_corners = int(p.get("max_corners", 400))
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_pts: Optional[np.ndarray] = None
        self._scale = 1.0
        self._M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)     # current -> reference, in small-image pixels
        self.shift_px = 0.0                                          # last estimated shift in full-frame pixels
        self.inliers = 0
        self.failures = 0

    # ------------------------------------------------------------------ public
    def stabilize(self, frame: np.ndarray, ignore_boxes: Optional[Sequence[Sequence[float]]] = None) -> np.ndarray:
        """Return ``frame`` aligned to the reference frame (the first frame passed in)."""
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        gray = self._small_gray(frame)
        if self._ref_gray is None:
            self._set_reference(gray)
            return frame
        if self._ref_pts is None or len(self._ref_pts) < self.min_inliers:
            return frame

        M = self._estimate(gray, ignore_boxes)
        if M is None:
            self.failures += 1
            M = self._M                                          # keep the last good transform
        else:
            self._M = M
        full = M.copy()
        full[:, 2] /= self._scale                                # translation back to full-resolution pixels
        self.shift_px = float(np.hypot(full[0, 2], full[1, 2]))
        if self.shift_px < 0.3 and abs(full[0, 1]) < 1e-4 and abs(full[0, 0] - 1) < 1e-4:
            return frame
        return cv2.warpAffine(frame, full, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def reset(self) -> None:
        self._ref_gray = self._ref_pts = None
        self._M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        self.shift_px, self.inliers, self.failures = 0.0, 0, 0

    # ---------------------------------------------------------------- internals
    def _small_gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = min(1.0, self.max_width / w)
        small = cv2.resize(frame, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA) if self._scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _set_reference(self, gray: np.ndarray) -> None:
        self._ref_gray = gray
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=8, blockSize=7)
        self._ref_pts = pts
        if pts is None or len(pts) < self.min_inliers:
            log.warning("Stabilizer: too few features in the reference frame; stabilization is inactive")

    def _estimate(self, gray: np.ndarray, ignore_boxes) -> Optional[np.ndarray]:
        """Fit the current->reference similarity transform (small-image pixels), or ``None`` if unreliable."""
        assert self._ref_gray is not None and self._ref_pts is not None
        cur, status, _ = cv2.calcOpticalFlowPyrLK(self._ref_gray, gray, self._ref_pts, None, winSize=(21, 21), maxLevel=3)
        if cur is None:
            return None
        good = status.reshape(-1) == 1
        ref_pts, cur_pts = self._ref_pts.reshape(-1, 2)[good], cur.reshape(-1, 2)[good]
        if ignore_boxes:                                          # points on vehicles are not background
            keep = np.ones(len(cur_pts), bool)
            for b in ignore_boxes:
                x1, y1, x2, y2 = (float(v) * self._scale for v in b[:4])
                keep &= ~((cur_pts[:, 0] >= x1) & (cur_pts[:, 0] <= x2) & (cur_pts[:, 1] >= y1) & (cur_pts[:, 1] <= y2))
            ref_pts, cur_pts = ref_pts[keep], cur_pts[keep]
        if len(cur_pts) < self.min_inliers:
            return None
        M, inl = cv2.estimateAffinePartial2D(cur_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=500, confidence=0.99)
        if M is None or inl is None or int(inl.sum()) < self.min_inliers:
            return None
        self.inliers = int(inl.sum())
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        sw = self._ref_gray.shape[1]
        if abs(scale - 1) > self.max_scale_change or np.hypot(M[0, 2], M[1, 2]) > self.max_shift_frac * sw:
            return None                                           # implausible: the camera was probably re-aimed
        return M
