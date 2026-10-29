"""Read-only access to events.csv: filtering, sorting and the aggregates the dashboard shows."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

FLAGGED = ("red_light_violation", "amber_crossing", "overspeed")
EVENT_TYPES = ("red_light_violation", "amber_crossing", "legal_crossing", "overspeed")
CROSSING_TYPES = ("red_light_violation", "amber_crossing", "legal_crossing")   # overspeed is not a line crossing
SORTABLE = {"event_id", "track_id", "vehicle_class", "event_type", "signal_state", "frame_no",
            "video_timestamp", "wall_clock_time", "confidence", "plate_text", "plate_conf", "time", "speed_kmh"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_\-.]+$")


@dataclass
class EventFilters:
    """Query filters shared by the list, export, report and analytics endpoints."""

    event_type: Optional[str] = None      # comma separated
    vehicle_class: Optional[str] = None   # comma separated
    signal_state: Optional[str] = None    # comma separated
    camera_id: Optional[str] = None
    plate: Optional[str] = None           # substring of plate text
    q: Optional[str] = None               # free-text search
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class EventsService:
    """Loads events.csv (re-read when it changes) and answers queries on it."""

    def __init__(self, cfg: dict) -> None:
        self.csv_path = Path(cfg["paths"]["events_csv"])
        self.evidence_root = Path(cfg["paths"]["evidence_dir"])
        self.stats_path = Path(cfg["paths"]["stats_json"])
        self.benchmark_path = Path(cfg["paths"]["benchmark_json"])
        self.evaluation_path = Path(cfg["paths"]["evaluation_json"])
        self._lock = threading.Lock()
        self._sig: Optional[tuple[float, int]] = None
        self._df = pd.DataFrame()

    # ----------------------------------------------------------------- loading
    def frame(self) -> pd.DataFrame:
        """The full events table with a parsed ``time`` column; empty if there is no file."""
        with self._lock:
            try:
                st = self.csv_path.stat()
            except OSError:
                self._sig, self._df = None, pd.DataFrame()
                return self._df
            sig = (st.st_mtime, st.st_size)
            if sig != self._sig:
                self._df = self._load()
                self._sig = sig
            return self._df

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path, keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame()
        if df.empty:
            return df
        for col in ("track_id", "frame_no"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ("confidence", "plate_conf"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for col in ("speed_kmh", "limit_kmh"):       # only overspeed rows have values
            df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else float("nan")
        for col in ("plate_text", "evidence_path", "camera_id"):
            if col not in df:
                df[col] = ""
        # wall_clock_time is stored as local ISO time; keep the local clock, drop the offset.
        df["time"] = pd.to_datetime(df["wall_clock_time"].astype(str).str[:19], errors="coerce")
        return df

    # --------------------------------------------------------------- filtering
    def filtered(self, f: EventFilters) -> pd.DataFrame:
        df = self.frame()
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        for col, value in (("event_type", f.event_type), ("vehicle_class", f.vehicle_class),
                           ("signal_state", f.signal_state)):
            if value:
                mask &= df[col].isin([v.strip() for v in value.split(",") if v.strip()])
        if f.camera_id:
            mask &= df["camera_id"] == f.camera_id
        if f.plate:
            mask &= df["plate_text"].astype(str).str.contains(f.plate, case=False, regex=False)
        if f.q:
            hay = (df["event_id"].astype(str) + " " + df["plate_text"].astype(str) + " "
                   + df["vehicle_class"].astype(str) + " " + df["track_id"].astype(str))
            mask &= hay.str.contains(f.q, case=False, regex=False)
        start = _to_ts(f.date_from)
        if start is not None:
            mask &= df["time"] >= start
        end = _to_ts(f.date_to, end_of_day=True)
        if end is not None:
            mask &= df["time"] <= end
        return df[mask]

    def list(self, f: EventFilters, sort: str = "time", order: str = "desc",
             limit: int = 50, offset: int = 0) -> dict[str, Any]:
        df = self.filtered(f)
        total = len(df)
        if total:
            key = sort if sort in SORTABLE else "time"
            df = df.sort_values(key, ascending=(order == "asc"), kind="stable")
            df = df.iloc[offset: offset + limit]
        return {"total": total, "items": [self._row(r) for r in df.to_dict("records")],
                "facets": self.facets()}

    def facets(self) -> dict[str, list[str]]:
        df = self.frame()
        if df.empty:
            return {"vehicle_class": [], "signal_state": [], "event_type": list(EVENT_TYPES), "camera_id": []}
        return {
            "vehicle_class": sorted(df["vehicle_class"].astype(str).unique()),
            "signal_state": sorted(df["signal_state"].astype(str).unique()),
            "event_type": list(EVENT_TYPES),
            "camera_id": sorted(c for c in df["camera_id"].astype(str).unique() if c),
        }

    def _row(self, rec: dict) -> dict[str, Any]:
        t = rec.get("time")
        out = {k: (None if _isnull(v) else v) for k, v in rec.items() if k != "time"}
        out["time"] = None if _isnull(t) else pd.Timestamp(t).isoformat()
        out["has_evidence"] = bool(rec.get("evidence_path"))
        return out

    def get(self, event_id: str) -> Optional[dict[str, Any]]:
        """One event with its evidence file URLs and metadata (``None`` if unknown)."""
        df = self.frame()
        if df.empty or not SAFE_ID.match(event_id):
            return None
        rows = df[df["event_id"] == event_id]
        if rows.empty:
            return None
        event = self._row(rows.iloc[0].to_dict())
        folder = self.evidence_root / event_id
        files, meta = {}, {}
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    files[p.stem] = f"/api/evidence/{event_id}/{p.name}"
            mp = folder / "metadata.json"
            if mp.is_file():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
        event["evidence"] = files
        event["metadata"] = meta
        return event

    def evidence_file(self, event_id: str, name: str) -> Optional[Path]:
        """Resolve an evidence file safely inside the evidence root, or ``None``."""
        if not SAFE_ID.match(event_id) or not SAFE_FILE.match(name):
            return None
        root = self.evidence_root.resolve()
        path = (root / event_id / name).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            return None
        return path

    # -------------------------------------------------------------- aggregates
    def live_stats(self) -> dict[str, Any]:
        if self.stats_path.is_file():
            try:
                return json.loads(self.stats_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {}

    def kpis(self, f: EventFilters) -> dict[str, Any]:
        df = self.filtered(f)
        stats = self.live_stats()
        red = int((df["event_type"] == "red_light_violation").sum()) if len(df) else 0
        amber = int((df["event_type"] == "amber_crossing").sum()) if len(df) else 0
        legal = int((df["event_type"] == "legal_crossing").sum()) if len(df) else 0
        over = int((df["event_type"] == "overspeed").sum()) if len(df) else 0
        plates = int((df["plate_text"].astype(str) != "").sum()) if len(df) else 0
        top = float(df["speed_kmh"].max()) if len(df) and df["speed_kmh"].notna().any() else 0.0
        return {
            "vehicles_tracked": int(stats.get("vehicles_tracked", 0)),
            "crossings": red + amber + legal,
            "red_light_violations": red,
            "amber_crossings": amber,
            "legal_crossings": legal,
            "overspeed_events": over,
            "max_speed_kmh": round(max(top, float(stats.get("max_speed_kmh", 0.0))), 1),
            "plates_read": plates,
            "fps": float(stats.get("fps", 0.0)),
            "status": stats.get("status", "idle"),
        }

    def timeseries(self, f: EventFilters, bucket: Optional[str] = None) -> dict[str, Any]:
        df = self.filtered(f).dropna(subset=["time"]) if len(self.frame()) else self.frame()
        if df.empty:
            return {"bucket": bucket or "hour", "points": []}
        if bucket not in ("minute", "hour", "day"):
            span = df["time"].max() - df["time"].min()
            bucket = "minute" if span < pd.Timedelta(hours=2) else "hour" if span < pd.Timedelta(days=3) else "day"
        freq = {"minute": "min", "hour": "h", "day": "D"}[bucket]
        grouped = (df.assign(t=df["time"].dt.floor(freq))
                   .groupby(["t", "event_type"]).size().unstack(fill_value=0))
        for t in EVENT_TYPES:
            if t not in grouped:
                grouped[t] = 0
        grouped = grouped.reindex(pd.date_range(grouped.index.min(), grouped.index.max(), freq=freq), fill_value=0)
        points = [{"t": ts.isoformat(), **{k: int(row[k]) for k in EVENT_TYPES}} for ts, row in grouped.iterrows()]
        return {"bucket": bucket, "points": points}

    def by_class(self, f: EventFilters) -> list[dict[str, Any]]:
        df = self.filtered(f)
        df = df[df["event_type"].isin(FLAGGED)] if len(df) else df
        if df.empty:
            return []
        counts = df.groupby("vehicle_class").size().sort_values(ascending=False)
        return [{"name": str(k), "value": int(v)} for k, v in counts.items()]

    def by_signal(self, f: EventFilters) -> list[dict[str, Any]]:
        df = self.filtered(f)
        if len(df):
            df = df[df["event_type"].isin(CROSSING_TYPES)]     # overspeed events are not signal crossings
        counts = df["signal_state"].value_counts() if len(df) else pd.Series(dtype=int)
        return [{"name": s, "value": int(counts.get(s, 0))} for s in ("red", "yellow", "green")]

    def heatmap(self, f: EventFilters) -> list[dict[str, int]]:
        """Violations per (weekday 0=Mon, hour); only non-empty cells."""
        df = self.filtered(f)
        df = df[df["event_type"].isin(FLAGGED)].dropna(subset=["time"]) if len(df) else df
        if df.empty:
            return []
        g = df.groupby([df["time"].dt.weekday, df["time"].dt.hour]).size()
        return [{"day": int(d), "hour": int(h), "count": int(c)} for (d, h), c in g.items()]

    def json_file(self, which: str) -> Optional[dict]:
        path = self.benchmark_path if which == "benchmark" else self.evaluation_path
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def _isnull(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_ts(value: Optional[str], end_of_day: bool = False) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value[:19])
    except (ValueError, TypeError):
        return None
    if end_of_day and len(value) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return ts
