"""Live monitor: run control, MJPEG stream and the WebSocket feed."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services.context import AppContext, EventFilters, require_token, token_ok
from ..services.live_service import RunError, placeholder_jpeg

router = APIRouter(prefix="/api")
BOUNDARY = "frame"


class RunRequest(BaseModel):
    video: str = Field(min_length=1)
    camera: Optional[str] = None
    model: Optional[str] = None
    tracker: Optional[str] = Field(default=None, pattern="^(botsort|bytetrack)$")
    assume_signal: Optional[str] = Field(default=None, pattern="^(red|yellow|green)$")


@router.get("/live/status")
def status(ctx: AppContext = Depends(require_token)):
    snap = ctx.live.snapshot()
    snap["stats"] = snap["stats"] or ctx.events.live_stats()
    return snap


@router.get("/live/videos")
def videos(ctx: AppContext = Depends(require_token)):
    """Video files the Live monitor can start (relative paths) and whether camera sources are allowed."""
    cam_cfg = ctx.cfg.get("camera", {})
    return {"videos": ctx.live.video_choices(), "live_sources": ctx.live.allow_live,
            "cameras": list((cam_cfg.get("sites") or {}).keys()), "active_camera": cam_cfg.get("active", "default")}


@router.post("/live/start")
def start(req: RunRequest, ctx: AppContext = Depends(require_token)):
    try:
        ctx.live.start(req.video, req.camera, req.model, req.tracker, req.assume_signal)
    except RunError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ctx.live.snapshot()


@router.post("/live/stop")
def stop(ctx: AppContext = Depends(require_token)):
    ctx.live.stop()
    return {"stopping": ctx.live.running}


@router.get("/live/stream.mjpg")
async def stream(request: Request, ctx: AppContext = Depends(require_token)):
    """Multipart MJPEG of the latest annotated frame (a placeholder when idle)."""
    interval = ctx.live.min_interval
    idle = placeholder_jpeg()

    async def frames():
        while not await request.is_disconnected():
            jpg = ctx.live.latest_jpeg() or idle
            yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n").encode() + jpg + b"\r\n"
            await asyncio.sleep(interval)

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store"})


@router.websocket("/ws/live")
async def live_socket(ws: WebSocket):
    """Pushes stats, run status and newly logged events every ``websocket_push_interval_s``."""
    ctx: AppContext = ws.app.state.ctx
    if not token_ok(ctx, ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    interval = float(ctx.cfg.get("dashboard", {}).get("websocket_push_interval_s", 0.5))
    sent = len(ctx.events.frame())
    try:
        while True:
            df = ctx.events.frame()
            new_items = []
            if len(df) < sent:          # file was reset by a new run
                sent = 0
            if len(df) > sent:
                new_items = [ctx.events._row(r) for r in df.iloc[sent:].to_dict("records")]
                sent = len(df)
            snap = ctx.live.snapshot()
            await ws.send_json({
                "type": "tick", "run": {k: v for k, v in snap.items() if k != "stats"},
                "stats": snap["stats"] or ctx.events.live_stats(),
                "kpis": ctx.events.kpis(EventFilters()), "new_events": new_items,
            })
            await asyncio.sleep(interval)
    except (WebSocketDisconnect, RuntimeError):  # client left (a send after close raises RuntimeError)
        return
