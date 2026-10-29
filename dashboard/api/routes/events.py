"""Events: list, detail, CSV/PDF export and evidence files."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..services import report
from ..services.context import AppContext, EventFilters, filters, require_token

router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


@router.get("/events")
def list_events(
    f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token),
    sort: str = "time", order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
):
    """Filtered, sorted, paginated events plus the facet values for the filter UI."""
    return ctx.events.list(f, sort, order, limit, offset)


@router.get("/events/export.csv")
def export_csv(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    df = ctx.events.filtered(f)
    return Response(report.events_csv(df), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="events.csv"'})


@router.get("/report.pdf")
def export_pdf(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    df = ctx.events.filtered(f)
    text = ", ".join(f"{k}={v}" for k, v in vars(f).items() if v)
    pdf = report.violation_report_pdf(df.sort_values("time", ascending=False) if len(df) else df,
                                      ctx.events.evidence_root, text)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="violation_report.pdf"'})


@router.get("/events/{event_id}")
def event_detail(event_id: str, ctx: AppContext = Depends(require_token)):
    event = ctx.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.get("/evidence/{event_id}/{name}")
def evidence_file(event_id: str, name: str, ctx: AppContext = Depends(require_token)):
    path = ctx.events.evidence_file(event_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
