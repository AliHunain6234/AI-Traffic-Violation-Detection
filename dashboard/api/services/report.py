"""CSV and PDF exports of (filtered) events."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

EXPORT_COLUMNS = ["event_id", "track_id", "vehicle_class", "event_type", "signal_state", "frame_no",
                  "video_timestamp", "wall_clock_time", "confidence", "plate_text", "plate_conf", "evidence_path",
                  "speed_kmh", "limit_kmh", "approach"]
MAX_TABLE_ROWS = 300
MAX_THUMBNAILS = 12


def events_csv(df: pd.DataFrame) -> str:
    """Filtered events as CSV text (original columns only)."""
    cols = [c for c in EXPORT_COLUMNS + ["camera_id"] if c in df.columns]
    return df[cols].to_csv(index=False) if len(df) else ",".join(EXPORT_COLUMNS) + "\n"


def violation_report_pdf(df: pd.DataFrame, evidence_root: Path, filters_text: str = "") -> bytes:
    """A landscape A4 violation report: summary, evidence thumbnails and an event table."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Traffic Violation Report")
    styles = getSampleStyleSheet()
    story = [Paragraph("Traffic Violation Report", styles["Title"]),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}"
                       + (f" &nbsp;|&nbsp; Filters: {filters_text}" if filters_text else ""), styles["Normal"]),
             Spacer(1, 6 * mm)]

    red = int((df["event_type"] == "red_light_violation").sum()) if len(df) else 0
    amber = int((df["event_type"] == "amber_crossing").sum()) if len(df) else 0
    legal = int((df["event_type"] == "legal_crossing").sum()) if len(df) else 0
    plates = int((df["plate_text"].astype(str) != "").sum()) if len(df) else 0
    over = int((df["event_type"] == "overspeed").sum()) if len(df) else 0
    summary = Table([["Events", "Red-light", "Amber", "Legal", "Overspeed", "Plates read"],
                     [len(df), red, amber, legal, over, plates]], hAlign="LEFT")
    summary.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                 ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                 ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [summary, Spacer(1, 6 * mm)]
    story.append(Paragraph("Disclaimer: automated detections for human review; not a legal determination.",
                           styles["Italic"]))
    story.append(Spacer(1, 4 * mm))

    flagged = df[df["event_type"].isin(("red_light_violation", "amber_crossing", "overspeed"))] if len(df) else df
    thumbs = []
    for _, r in flagged.head(MAX_THUMBNAILS).iterrows():
        p = evidence_root / str(r["event_id"]) / "crossing.jpg"
        if p.is_file():
            thumbs.append([Image(str(p), width=60 * mm, height=34 * mm),
                           Paragraph(f"<b>{r['event_id']}</b><br/>{r['vehicle_class']} #{r['track_id']}<br/>"
                                     f"{r['event_type'].replace('_', ' ')}<br/>plate: {r['plate_text'] or '-'}",
                                     styles["BodyText"])])
    if thumbs:
        story.append(Paragraph("Evidence (crossing frames)", styles["Heading2"]))
        rows = [sum(thumbs[i:i + 2], []) for i in range(0, len(thumbs), 2)]
        for row in rows:
            row += [""] * (4 - len(row))
        story += [Table(rows, colWidths=[62 * mm, 62 * mm, 62 * mm, 62 * mm]), Spacer(1, 4 * mm)]

    story.append(Paragraph("Events", styles["Heading2"]))
    header = ["Event", "Time", "Class", "Type", "Signal", "Speed / limit", "Plate", "Conf"]
    data = [header]
    for _, r in df.head(MAX_TABLE_ROWS).iterrows():
        speed = f"{r['speed_kmh']:.0f} / {r['limit_kmh']:.0f}" if pd.notna(r.get("speed_kmh")) else "-"
        data.append([r["event_id"], str(r["wall_clock_time"])[:19].replace("T", " "), r["vehicle_class"],
                     str(r["event_type"]).replace("_", " "), r["signal_state"], speed, r["plate_text"] or "-",
                     f"{float(r['confidence']):.2f}"])
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    story.append(table)
    if len(df) > MAX_TABLE_ROWS:
        story.append(Paragraph(f"... {len(df) - MAX_TABLE_ROWS} more events not shown.", styles["Italic"]))
    doc.build(story)
    return buf.getvalue()
