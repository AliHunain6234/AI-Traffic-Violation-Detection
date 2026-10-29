"""Aggregates for the Overview and Analytics pages."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

from ..services.context import AppContext, EventFilters, filters, require_token

router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/kpis")
def kpis(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    return ctx.events.kpis(f)


@router.get("/analytics/timeseries")
def timeseries(bucket: Optional[str] = None, f: EventFilters = Depends(filters),
               ctx: AppContext = Depends(require_token)):
    return ctx.events.timeseries(f, bucket)


@router.get("/analytics/by-class")
def by_class(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    return ctx.events.by_class(f)


@router.get("/analytics/by-signal")
def by_signal(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    return ctx.events.by_signal(f)


@router.get("/analytics/heatmap")
def heatmap(f: EventFilters = Depends(filters), ctx: AppContext = Depends(require_token)):
    return ctx.events.heatmap(f)


@router.get("/analytics/benchmark")
def benchmark(ctx: AppContext = Depends(require_token)):
    """Latest benchmark.py results, or ``{"results": []}`` if it has not been run."""
    return ctx.events.json_file("benchmark") or {"results": []}


@router.get("/analytics/evaluation")
def evaluation(ctx: AppContext = Depends(require_token)):
    """Latest evaluate.py report, or ``{"overall": null}`` if it has not been run."""
    return ctx.events.json_file("evaluation") or {"overall": None}
