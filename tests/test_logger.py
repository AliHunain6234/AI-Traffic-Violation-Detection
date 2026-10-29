"""Regression tests: events.csv schema changes must never drop or truncate data."""
from __future__ import annotations

import csv
from pathlib import Path

from src.logger import EventIdFactory, EventLogger, EventRecord
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config(ROOT / "config.yaml")
    cfg["paths"]["events_csv"] = str(tmp_path / "events.csv")
    return cfg


def _record(ids: EventIdFactory, **extra) -> EventRecord:
    return EventRecord(ids.next(), 1, "car", "red_light_violation", "red", 10, "00:00:01.000", "2026-01-01T00:00:00+05:00",
                       0.9, camera_id="cam", extra=extra)


def test_old_events_csv_is_migrated_not_truncated(tmp_path):
    cfg = _cfg(tmp_path)
    old_columns = cfg["logging"]["csv_columns"] + ["camera_id"]            # the schema before approach/speed were added
    with open(cfg["paths"]["events_csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(old_columns)
        w.writerow(["EVT-OLD-1", 5, "bus", "legal_crossing", "green", 7, "t", "w", 0.7, "", 0, "", "cam"])
    EventLogger(cfg).log(_record(EventIdFactory("T"), approach="left_lanes", speed_kmh=88.5, limit_kmh=60.0))
    rows = list(csv.DictReader(open(cfg["paths"]["events_csv"], encoding="utf-8")))
    assert [r["event_id"] for r in rows][0] == "EVT-OLD-1" and len(rows) == 2          # old row kept
    assert rows[0]["approach"] == "" and rows[0]["track_id"] == "5"                     # new columns empty for old rows
    assert rows[1]["approach"] == "left_lanes" and rows[1]["speed_kmh"] == "88.5"       # new data NOT dropped


def test_header_only_file_is_rewritten_and_matching_file_untouched(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg["paths"]["events_csv"]).write_text(",".join(cfg["logging"]["csv_columns"]) + "\n", encoding="utf-8")
    EventLogger(cfg).log(_record(EventIdFactory("T"), approach="a"))
    EventLogger(cfg).log(_record(EventIdFactory("U"), approach="b"))                    # second run appends, no migration
    rows = list(csv.DictReader(open(cfg["paths"]["events_csv"], encoding="utf-8")))
    assert [r["approach"] for r in rows] == ["a", "b"]
