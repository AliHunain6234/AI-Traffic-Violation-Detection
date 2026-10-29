"""Compare predicted events (events.csv) with a hand-labelled ground-truth CSV.

    python evaluate.py --pred outputs/events.csv --gt ground_truth/ground_truth.csv --tolerance 15

A prediction matches a ground-truth event when the event types agree (unless
``--ignore-type``) and the frames differ by at most the tolerance. Matching is
one-to-one and minimises total frame error (Hungarian assignment).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.logger import setup_logging
from src.utils import ensure_dir, load_config

log = logging.getLogger("evaluate")

EVENT_TYPES = ("red_light_violation", "amber_crossing", "legal_crossing", "overspeed")
FLAGGED_TYPES = ("red_light_violation", "amber_crossing", "overspeed")
FORBIDDEN = 10**9


@dataclass
class MatchResult:
    """Outcome of matching predictions to ground truth."""

    pairs: list[tuple[int, int]] = field(default_factory=list)   # (gt_index, pred_index)
    missed: list[int] = field(default_factory=list)              # unmatched gt indices
    false_positives: list[int] = field(default_factory=list)     # unmatched pred indices


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load and validate the ground-truth CSV (``#`` comment lines are ignored).

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: On missing columns, non-integer frames or unknown event types.
    """
    df = pd.read_csv(path, comment="#", skip_blank_lines=True, dtype={"vehicle_description": str})
    df.columns = [c.strip() for c in df.columns]
    missing = {"frame_no", "event_type"} - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")
    if "vehicle_description" not in df.columns:
        df["vehicle_description"] = ""
    df["vehicle_description"] = df["vehicle_description"].fillna("")
    df["event_type"] = df["event_type"].astype(str).str.strip()
    bad = sorted(set(df["event_type"]) - set(EVENT_TYPES))
    if bad:
        raise ValueError(f"Unknown event_type in ground truth: {bad}; expected {list(EVENT_TYPES)}")
    try:
        df["frame_no"] = df["frame_no"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_no in ground truth must be integers") from exc
    return df.reset_index(drop=True)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load events.csv predictions (an empty or header-only file gives an empty frame)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    for col in ("frame_no", "event_type"):
        if col not in df.columns:
            raise ValueError(f"Predictions are missing column '{col}'")
    df["frame_no"] = pd.to_numeric(df["frame_no"], errors="coerce").fillna(-1).astype(int)
    return df.reset_index(drop=True)


def match_events(gt: pd.DataFrame, pred: pd.DataFrame, tolerance: int, match_type: bool = True) -> MatchResult:
    """One-to-one matching minimising total frame distance within ``tolerance``."""
    result = MatchResult()
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 or n_pred == 0:
        result.missed = list(range(n_gt))
        result.false_positives = list(range(n_pred))
        return result
    gf = gt["frame_no"].to_numpy()[:, None]
    pf = pred["frame_no"].to_numpy()[None, :]
    cost = np.abs(gf - pf).astype(np.int64)
    allowed = cost <= tolerance
    if match_type:
        allowed &= gt["event_type"].to_numpy()[:, None] == pred["event_type"].to_numpy()[None, :]
    cost = np.where(allowed, cost, FORBIDDEN)
    rows, cols = linear_sum_assignment(cost)
    matched_gt, matched_pred = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < FORBIDDEN:
            result.pairs.append((int(r), int(c)))
            matched_gt.add(int(r))
            matched_pred.add(int(c))
    result.missed = [i for i in range(n_gt) if i not in matched_gt]
    result.false_positives = [j for j in range(n_pred) if j not in matched_pred]
    return result


def _ratio(num: float, den: float) -> Optional[float]:
    return round(num / den, 4) if den else None


def metrics(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1 (``None`` where undefined, e.g. no predictions)."""
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * p * r, p + r) if p is not None and r is not None else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, tolerance: int, match_type: bool = True,
             flagged_only: bool = False) -> dict:
    """Full evaluation: overall + per event type metrics and the error lists."""
    if flagged_only:
        gt = gt[gt["event_type"].isin(FLAGGED_TYPES)].reset_index(drop=True)
        pred = pred[pred["event_type"].isin(FLAGGED_TYPES)].reset_index(drop=True)
    m = match_events(gt, pred, tolerance, match_type)
    offsets = [int(pred.loc[j, "frame_no"] - gt.loc[i, "frame_no"]) for i, j in m.pairs]
    type_mismatch = sum(1 for i, j in m.pairs if gt.loc[i, "event_type"] != pred.loc[j, "event_type"])
    per_type = {}
    for t in EVENT_TYPES:
        g_idx = set(gt.index[gt["event_type"] == t])
        p_idx = set(pred.index[pred["event_type"] == t])
        tp = sum(1 for i, j in m.pairs if i in g_idx and j in p_idx)
        per_type[t] = metrics(tp, len(p_idx) - tp, len(g_idx) - tp) if (g_idx or p_idx) else None
    return {
        "tolerance_frames": tolerance,
        "match_event_type": match_type,
        "ground_truth_events": len(gt),
        "predicted_events": len(pred),
        "overall": metrics(len(m.pairs), len(m.false_positives), len(m.missed)),
        "per_type": per_type,
        "mean_frame_offset": round(float(np.mean(offsets)), 2) if offsets else None,
        "mean_abs_frame_error": round(float(np.mean(np.abs(offsets))), 2) if offsets else None,
        "type_mismatches": type_mismatch,
        "false_positives": [
            {k: _plain(pred.loc[j, k]) for k in ("event_id", "track_id", "vehicle_class", "event_type", "frame_no")
             if k in pred.columns} for j in m.false_positives],
        "missed_events": [
            {"frame_no": int(gt.loc[i, "frame_no"]), "event_type": gt.loc[i, "event_type"],
             "vehicle_description": gt.loc[i, "vehicle_description"]} for i in m.missed],
    }


def _plain(v):
    return v.item() if hasattr(v, "item") else v


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_report(res: dict) -> None:
    """Log a readable report."""
    o = res["overall"]
    log.info("Ground truth: %d | Predictions: %d | tolerance: +/-%d frames | match type: %s",
             res["ground_truth_events"], res["predicted_events"], res["tolerance_frames"], res["match_event_type"])
    log.info("OVERALL  precision %s  recall %s  F1 %s  (TP %d, FP %d, FN %d)",
             _fmt(o["precision"]), _fmt(o["recall"]), _fmt(o["f1"]), o["tp"], o["fp"], o["fn"])
    for t, m in res["per_type"].items():
        if m:
            log.info("  %-20s P %s  R %s  F1 %s  (TP %d, FP %d, FN %d)",
                     t, _fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["f1"]), m["tp"], m["fp"], m["fn"])
    if res["mean_abs_frame_error"] is not None:
        log.info("Mean frame offset %+.1f (abs %.1f)", res["mean_frame_offset"], res["mean_abs_frame_error"])
    for fp in res["false_positives"]:
        log.info("FALSE POSITIVE: %s", fp)
    for miss in res["missed_events"]:
        log.info("MISSED: %s", miss)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate predicted events against ground truth")
    p.add_argument("--pred", help="predictions CSV (default: paths.events_csv)")
    p.add_argument("--gt", help="ground-truth CSV (default: paths.ground_truth_csv)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--tolerance", type=int, help="frame tolerance (default: evaluation.frame_tolerance)")
    p.add_argument("--ignore-type", action="store_true", help="match on frame only, not on event type")
    p.add_argument("--flagged-only", action="store_true", help="only red-light and amber events")
    p.add_argument("--output", help="JSON report path (default: paths.evaluation_json)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging({"logging": {"level": "INFO"}})
    ev_cfg = cfg.get("evaluation", {})
    tol = args.tolerance if args.tolerance is not None else int(ev_cfg.get("frame_tolerance", 15))
    match_type = not args.ignore_type and bool(ev_cfg.get("match_event_type", True))
    try:
        gt = load_ground_truth(args.gt or cfg["paths"]["ground_truth_csv"])
        pred = load_predictions(args.pred or cfg["paths"]["events_csv"])
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 2
    res = evaluate(gt, pred, tol, match_type, args.flagged_only)
    print_report(res)
    out = Path(args.output or cfg["paths"]["evaluation_json"])
    ensure_dir(out.parent)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("Report written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
