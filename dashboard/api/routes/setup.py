"""Camera setup from the browser: show a frame of a video / camera, save the lines and calibration drawn on it.

The same validation as the command-line tools (src/calibration.py) is used, and the same config
keys are written (comments in config.yaml are preserved).
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from src.calibration import build_calibration, rect_world_points, validate_line, validate_roi
from src.utils import get_approaches, get_camera, load_config, read_frame, remove_approach, save_approach_value, save_config_value

from ..services.context import AppContext, require_token
from ..services.live_service import RunError

log = logging.getLogger("dashboard.setup")
router = APIRouter(prefix="/api/setup")
NAME = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def _editable(ctx: AppContext) -> None:
    if not ctx.cfg.get("dashboard", {}).get("allow_config_edit", False):
        raise HTTPException(status_code=403, detail="Editing the config from the dashboard is disabled (set dashboard.allow_config_edit: true)")


def _reload(ctx: AppContext) -> None:
    """Re-read config.yaml into the SAME dict object so services holding it see the change."""
    fresh = load_config(ctx.config_path)
    ctx.cfg.clear()
    ctx.cfg.update(fresh)


class SpeedPayload(BaseModel):
    method: Literal["lines", "homography"]
    points: list[list[float]] = Field(min_length=4, max_length=4)
    distance_m: Optional[float] = Field(default=None, gt=0, le=2000)       # lines: real distance between the two lines
    rect_m: Optional[list[float]] = None                                    # homography: [width, length] of the rectangle
    world: Optional[list[list[float]]] = None                              # homography: 4 explicit world points


class SaveRequest(BaseModel):
    camera: str
    approach: Optional[str] = None
    frame_size: list[int] = Field(min_length=2, max_length=2)
    stop_line: Optional[list[list[float]]] = None
    direction: Optional[Literal["positive", "negative"]] = None
    signal_roi: Optional[list[float]] = None
    min_pixel_ratio: Optional[float] = Field(default=None, gt=0, lt=1)
    speed: Optional[SpeedPayload] = None
    clear_speed: bool = False

    @field_validator("camera", "approach")
    @classmethod
    def _safe_name(cls, v):
        if v is not None and not NAME.match(v):
            raise ValueError("names may only contain letters, digits, '-' and '_' (max 40)")
        return v


@router.get("/frame")
def frame(source: str = Query(min_length=1), index: int = Query(0, ge=0, le=200000), max_width: int = Query(1600, ge=200, le=4000),
          ctx: AppContext = Depends(require_token)):
    """One frame of a video file / webcam / camera URL as JPEG; the real size is in ``X-Frame-Width/Height``."""
    try:
        resolved = ctx.live.check_video(source)
    except RunError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if ctx.live.running and not resolved.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
        raise HTTPException(status_code=409, detail="A live source is being processed; stop it before opening the same camera here")
    try:
        img = read_frame(resolved, index)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, None, fx=max_width / w, fy=max_width / w, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the frame")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Frame-Width": str(w), "X-Frame-Height": str(h), "Cache-Control": "no-store",
                             "Access-Control-Expose-Headers": "X-Frame-Width, X-Frame-Height"})


@router.get("/camera/{camera_id}")
def camera_config(camera_id: str, ctx: AppContext = Depends(require_token)):
    """Current setup of a camera (so the page can draw what is already configured)."""
    if not NAME.match(camera_id):
        raise HTTPException(status_code=400, detail="Invalid camera name")
    try:
        cam = get_camera(ctx.cfg, camera_id)
    except KeyError:
        return {"id": camera_id, "exists": False, "approaches": [], "speed": None, "frame_size": None}
    return {"id": camera_id, "exists": True, "frame_size": cam.get("frame_size"), "speed": cam.get("speed"),
            "approaches": get_approaches(cam), "uses_legacy_single_line": not cam.get("approaches")}


@router.post("/save")
def save(req: SaveRequest, ctx: AppContext = Depends(require_token)):
    """Validate and save whatever was drawn; returns the camera's configuration afterwards."""
    _editable(ctx)
    w, h = req.frame_size
    if w < 16 or h < 16:
        raise HTTPException(status_code=400, detail="frame_size looks wrong")
    path = ctx.config_path
    base = f"camera.sites.{req.camera}"
    try:
        saved: list[str] = []
        if req.stop_line is not None:
            validate_line(req.stop_line)
            line = [[int(round(x)), int(round(y))] for x, y in req.stop_line]
            if req.approach:
                save_approach_value(path, req.camera, req.approach, "stop_line", line)
                if req.direction:
                    save_approach_value(path, req.camera, req.approach, "direction", req.direction)
            else:
                save_config_value(path, f"{base}.stop_line", line)
                if req.direction:
                    save_config_value(path, f"{base}.direction", req.direction)
            saved.append("stop line")
        if req.signal_roi is not None:
            roi = validate_roi(req.signal_roi, (w, h))
            if req.approach:
                save_approach_value(path, req.camera, req.approach, "signal_roi", roi)
            else:
                save_config_value(path, f"{base}.signal_roi", roi)
            saved.append("signal ROI")
        if req.min_pixel_ratio is not None:
            if req.approach:
                save_approach_value(path, req.camera, req.approach, "signal", {"min_pixel_ratio": req.min_pixel_ratio})
            else:
                save_config_value(path, f"{base}.signal", {"min_pixel_ratio": req.min_pixel_ratio})
            saved.append("signal sensitivity")
        if req.speed is not None:
            sp = req.speed
            world = sp.world
            if sp.method == "homography" and world is None:
                if not sp.rect_m or len(sp.rect_m) != 2:
                    raise ValueError("homography needs the real width and length (metres) of the rectangle you marked")
                world = rect_world_points(*sp.rect_m)
            save_config_value(path, f"{base}.speed", build_calibration(sp.method, sp.points, sp.distance_m, world))
            saved.append("speed calibration")
        if req.clear_speed:
            save_config_value(path, f"{base}.speed", None)
            saved.append("speed calibration removed")
        if not saved:
            raise ValueError("Nothing to save: draw a stop line, a signal ROI or a speed calibration first")
        save_config_value(path, f"{base}.frame_size", [w, h])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _reload(ctx)
    log.info("Setup saved for camera %s%s: %s", req.camera, f" / {req.approach}" if req.approach else "", ", ".join(saved))
    return {"saved": saved, **camera_config(req.camera, ctx)}


@router.delete("/approach")
def delete_approach(camera: str = Query(), name: str = Query(), ctx: AppContext = Depends(require_token)):
    _editable(ctx)
    if not NAME.match(camera) or not NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    if not remove_approach(ctx.config_path, camera, name):
        raise HTTPException(status_code=404, detail="No such approach")
    _reload(ctx)
    return camera_config(camera, ctx)
