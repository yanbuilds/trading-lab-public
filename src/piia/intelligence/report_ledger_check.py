"""Check local report/ledger coverage consistency, not correctness of reported P&L."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from piia.intelligence.paths import ROOT

APP_ID = 0x46505631
POLICY_START = datetime(2026, 11, 18, tzinfo=timezone.utc)


def _timestamp(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timezone required")
    return t.astimezone(timezone.utc)


def check_report_ledger(report: dict, *, ledger: Path | None = None, now: datetime | None = None) -> dict:
    """Only attest coverage metadata against the current local database, read-only."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone required")
    path = Path(ledger) if ledger is not None else ROOT / "data" / "piia" / "frozen_policy_v1.sqlite"
    issues = []
    try:
        if not path.is_file():
            raise FileNotFoundError("ledger missing")
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
                raise ValueError("unrelated ledger")
            rows = db.execute("SELECT cutoff, created_at FROM observations ORDER BY cutoff").fetchall()
        if len(rows) < 3:
            raise ValueError("insufficient observations")
        parsed = [(_timestamp(c), _timestamp(created)) for c, created in rows]
        cutoffs = [c for c, _ in parsed]
        if (cutoffs[0] < POLICY_START or any(c.minute or c.second or c.microsecond for c in cutoffs)
                or any(b - a != timedelta(hours=1) for a, b in zip(cutoffs, cutoffs[1:]))
                or any(not timedelta(0) <= created - cutoff <= timedelta(minutes=50) for cutoff, created in parsed)):
            raise ValueError("invalid or discontinuous observation times")
        generated = _timestamp(report["generated_at_utc"])
        if type(report.get("observations")) is not int or report["observations"] != len(rows):
            issues.append("observation_count_mismatch")
        if (_timestamp(report["first_decision_utc"]) != cutoffs[0]
                or _timestamp(report["last_observation_cutoff_utc"]) != cutoffs[-1]):
            issues.append("observation_window_mismatch")
        if not max(created for _, created in parsed) <= generated <= now.astimezone(timezone.utc):
            issues.append("report_generation_time_invalid")
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError, AttributeError, OverflowError):
        issues.append("ledger_or_report_metadata_unavailable_or_invalid")
    return {"ok": not issues, "issues": issues, "pnl_recomputed": False,
            "independent_provenance_verified": False,
            "note": "Coverage metadata consistency does not prove that report values were computed from the ledger."}
