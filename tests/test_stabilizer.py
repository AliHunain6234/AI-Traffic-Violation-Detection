"""CameraStabilizer: removes known camera shake, ignores moving vehicles, and never crashes."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.stabilizer import CameraStabilizer

W, H = 640, 480


@pytest.fixture(scope="module")
def background() -> np.ndarray:
    rng = np.random.default_rng(5)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2.0)
    for _ in range(40):                                          # some structure: rectangles like buildings / lane marks
        x, y = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 60))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(15, 60)), y + int(rng.integers(15, 60))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def shaken(img: np.ndarray, dx: float, dy: float, angle: float = 0.0) -> np.ndarray:
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def inner_error(a: np.ndarray, b: np.ndarray, margin: int = 70) -> float:
    return float(np.abs(a[margin:-margin, margin:-margin].astype(np.float32) - b[margin:-margin, margin:-margin]).mean())


def test_first_frame_is_returned_untouched_and_defines_the_reference(background):
    s = CameraStabilizer()
    out = s.stabilize(background)
    assert out is background


@pytest.mark.parametrize("dx,dy,angle", [(12, -9, 0), (-25, 18, 0), (7, 5, 1.5), (-30, -22, -2.0)])
def test_known_shake_is_removed(background, dx, dy, angle):
    s = CameraStabilizer()
    s.stabilize(background)
    frame = shaken(background, dx, dy, angle)
    before = inner_error(frame, background)
    after = inner_error(s.stabilize(frame), background)
    assert after < before * 0.35 and after < 6.0, (before, after)
    assert s.inliers >= 12


def test_a_moving_vehicle_does_not_pull_the_estimate(background):
    s = CameraStabilizer()
    s.stabilize(background)
    frame = shaken(background, 14, -10)
    cv2.rectangle(frame, (250, 200), (390, 330), (0, 0, 255), -1)             # a big vehicle in the middle
    out = s.stabilize(frame, ignore_boxes=[(250, 200, 390, 330)])
    ref = background.copy()
    mask = np.ones((H, W), bool); mask[190:340, 240:400] = False
    err = float(np.abs(out.astype(np.float32) - ref)[mask][::7].mean())
    assert err < 8.0 and s.shift_px == pytest.approx(np.hypot(14, 10), abs=1.5)


def test_estimated_shift_matches_the_real_one(background):
    s = CameraStabilizer()
    s.stabilize(background)
    s.stabilize(shaken(background, -18, 11))
    assert s.shift_px == pytest.approx(np.hypot(18, 11), abs=1.0)


def test_unreliable_frames_fall_back_to_the_last_good_transform(background):
    s = CameraStabilizer()
    s.stabilize(background)
    s.stabilize(shaken(background, 10, 6))
    good = s._M.copy()
    out = s.stabilize(np.full((H, W, 3), 128, np.uint8))                       # featureless frame: no match possible
    assert out.shape == (H, W, 3) and s.failures == 1 and np.allclose(s._M, good)


def test_implausible_jump_is_rejected(background):
    s = CameraStabilizer(params={"max_shift_frac": 0.1})
    s.stabilize(background)
    s.stabilize(shaken(background, 120, 0))                                    # > 10 % of the width: probably re-aimed
    assert s.failures == 1


def test_degenerate_inputs_do_not_crash(background):
    s = CameraStabilizer()
    assert s.stabilize(np.zeros((0, 0, 3), np.uint8)).size == 0
    blank = CameraStabilizer()
    blank.stabilize(np.full((H, W, 3), 90, np.uint8))                          # featureless reference: inactive, frames pass through
    assert blank.stabilize(background) is background
    s.stabilize(background); s.reset()
    assert s.stabilize(background) is background                                # after reset the next frame is the new reference
