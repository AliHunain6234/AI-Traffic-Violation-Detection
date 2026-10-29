"""Shared application context and the optional token check."""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query, Request

from .events_service import EventFilters, EventsService
from .live_service import LiveService


@dataclass
class AppContext:
    cfg: dict
    events: EventsService
    live: LiveService
    token: Optional[str]
    config_path: str = "config.yaml"      # the file the Setup page edits


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def token_ok(ctx: AppContext, provided: Optional[str]) -> bool:
    """True if no token is configured or ``provided`` matches it (constant-time compare)."""
    if not ctx.token:
        return True
    return bool(provided) and hmac.compare_digest(str(provided), ctx.token)


def require_token(
    ctx: AppContext = Depends(get_ctx),
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> AppContext:
    """Dependency: accept ``Authorization: Bearer <token>`` or ``?token=`` (needed by <img>/EventSource)."""
    bearer = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
    if not token_ok(ctx, bearer or token):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return ctx


def filters(
    event_type: Optional[str] = None, vehicle_class: Optional[str] = None, signal_state: Optional[str] = None,
    camera_id: Optional[str] = None, plate: Optional[str] = None, q: Optional[str] = None,
    date_from: Optional[str] = None, date_to: Optional[str] = None,
) -> EventFilters:
    """Dependency turning query parameters into ``EventFilters``."""
    return EventFilters(event_type, vehicle_class, signal_state, camera_id, plate, q, date_from, date_to)
