"""Logging setup, the events.csv writer and the live stats.json writer."""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .utils import RollingFPS, ensure_dir

log = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(cfg: dict) -> None:
    """Configure root logging to the console and (optionally) a file.

    Safe to call more than once: existing handlers are replaced.
    """
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_path = log_cfg.get("file")
    if file_path:
        ensure_dir(Path(file_path).parent)
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)


@dataclass
class EventRecord:
    """One crossing event; field names match the events.csv columns."""

    event_id: str
    track_id: int
    vehicle_class: str
    event_type: str
    signal_state: str
    frame_no: int
    video_timestamp: str
    wall_clock_time: str
    confidence: float
    plate_text: str = ""
    plate_conf: float = 0.0
    evidence_path: str = ""
    camera_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self, columns: list[str]) -> list[Any]:
        """Order the record's values by ``columns``; unknown columns come out empty."""
        data = {**asdict(self), **self.extra}
        row: list[Any] = []
        for col in columns:
            value = data.get(col, "")
            if isinstance(value, float):
                value = round(value, 4)
            row.append(value)
        return row


class EventIdFactory:
    """Generates unique, filesystem-safe event ids such as ``EVT-20260314-083000-0007``."""

    def __init__(self, run_tag: Optional[str] = None) -> None:
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._seq = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            self._seq += 1
            return f"EVT-{self.run_tag}-{self._seq:04d}"


class EventLogger:
    """Thread-safe, append-only writer for ``events.csv``.

    The header is written once; every event is flushed immediately so the
    dashboard (and a crashed run) always sees complete rows.
    """

    def __init__(self, cfg: dict) -> None:
        log_cfg = cfg.get("logging", {})
        self.path = Path(cfg["paths"]["events_csv"])
        self.columns: list[str] = list(log_cfg.get("csv_columns", [])) + list(
            log_cfg.get("extra_columns", [])
        )
        if not self.columns:
            raise ValueError("logging.csv_columns is empty in config")
        self._lock = threading.Lock()
        ensure_dir(self.path.parent)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                existing = list(reader.fieldnames or [])
                rows = list(reader)
            if existing != self.columns:
                # The config gained (or lost) columns since this file was created: migrate it in place,
                # keeping every old row (new columns stay empty) instead of silently dropping new data.
                merged = self.columns + [c for c in existing if c not in self.columns]
                log.warning("events.csv header differs from config; migrating %d existing rows to the new columns", len(rows))
                with self.path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=merged, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                self.columns = merged
            return
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(self.columns)

    def log(self, record: EventRecord) -> None:
        """Append one event and flush."""
        with self._lock, self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(record.to_row(self.columns))
        log.info(
            "%s | track %s %s | signal=%s | frame %d",
            record.event_type, record.track_id, record.vehicle_class,
            record.signal_state, record.frame_no,
        )

    def read(self):
        """Return all logged events as a pandas DataFrame (empty if none)."""
        import pandas as pd

        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.path, keep_default_na=False)


class StatsWriter:
    """Keeps live counters and writes them atomically to ``stats.json``.

    The dashboard polls this file for KPI cards and the processing FPS.
    """

    def __init__(self, cfg: dict, camera_id: str = "", persist: bool = True) -> None:
        self.path = Path(cfg["paths"]["stats_json"])
        self.interval = int(cfg.get("logging", {}).get("stats_interval_frames", 15))
        self.persist = persist  # False keeps counters in memory only (benchmark runs)
        if persist:
            ensure_dir(self.path.parent)
        self._fps = RollingFPS()
        self._since_write = 0
        self.stats: dict[str, Any] = {
            "camera_id": camera_id,
            "status": "idle",
            "frame_no": 0,
            "total_frames": 0,
            "video_timestamp": "00:00:00.000",
            "signal_state": "unknown",
            "signals": {},                 # approach name -> signal state (one entry per stop line)
            "vehicles_tracked": 0,
            "active_tracks": 0,
            "crossings": 0,
            "red_light_violations": 0,
            "amber_crossings": 0,
            "legal_crossings": 0,
            "plates_read": 0,
            "speed_measurements": 0,
            "overspeed_events": 0,
            "max_speed_kmh": 0.0,
            "fps": 0.0,
            "updated_at": "",
        }

    def tick(self) -> float:
        """Register a processed frame; returns the rolling FPS and flushes periodically."""
        self.stats["fps"] = round(self._fps.tick(), 2)
        self._since_write += 1
        if self._since_write >= self.interval:
            self.write()
        return self.stats["fps"]

    def update(self, **values: Any) -> None:
        """Merge values into the stats (unknown keys are accepted)."""
        self.stats.update(values)

    def count_event(self, event_type: str, plate_read: bool = False) -> None:
        """Increment the counters for one finished event."""
        self.stats["crossings"] += 1
        key = {
            "red_light_violation": "red_light_violations",
            "amber_crossing": "amber_crossings",
            "legal_crossing": "legal_crossings",
        }.get(event_type)
        if key:
            self.stats[key] += 1
        if plate_read:
            self.stats["plates_read"] += 1

    def count_speed(self, speed_kmh: float, overspeed: bool) -> None:
        """Record one finished speed measurement."""
        self.stats["speed_measurements"] += 1
        self.stats["max_speed_kmh"] = round(max(self.stats["max_speed_kmh"], float(speed_kmh)), 1)
        if overspeed:
            self.stats["overspeed_events"] += 1

    def write(self) -> None:
        """Atomically write stats.json (temp file + replace)."""
        self._since_write = 0
        self.stats["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if not self.persist:
            return
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:  # a reader may briefly hold the file open on Windows
            log.warning("Could not write %s: %s", self.path, exc)
