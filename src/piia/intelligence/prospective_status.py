"""Read-only collection status, never an assessment of strategy returns."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from piia.intelligence.paths import ROOT

APP_ID = 0x46505631
START = datetime(2026, 11, 18, tzinfo=timezone.utc)
STEPS = {
    "funding": ("funding_observation_claims", "funding_observations", {"complete", "incomplete"}),
    "ai_shadow": ("ai_shadow_claims", "ai_shadow_events", {"completed", "no_candidate", "model_failed"}),
    "chronos_tiny": ("chronos_shadow_claims", "chronos_shadow_events", {"completed", "no_candidate", "model_failed"}),
    "chronos2": ("chronos2_shadow_claims", "chronos2_shadow_events", {"completed", "no_candidate", "model_failed"}),
}


def _time(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone required")
    return value.astimezone(timezone.utc)


def collection_status(*, path: Path | None = None, now: datetime | None = None) -> dict:
    """Inspect current ledger and latest step records without creating or updating it."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone required")
    now = now.astimezone(timezone.utc)
    cutoff = now.replace(minute=0, second=0, microsecond=0)
    path = Path(path) if path is not None else ROOT / "data" / "piia" / "frozen_policy_v1.sqlite"
    result = {"status": "unavailable", "checked_at_utc": now.isoformat(), "current_cutoff_utc": cutoff.isoformat(),
              "order_permissions": False, "performance_calculated": False,
              "note": "Local coverage status only; does not independently validate evidence, model outputs or profitability."}
    if not path.is_file():
        result.update(status="no_ledger", observations=0,
                      current_hour_status="not_started" if cutoff < START else
                      "open_unrecorded" if now - cutoff <= timedelta(minutes=50) else "window_closed")
        return result
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=2) as db:
            db.execute("BEGIN")
            if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
                raise ValueError("unrelated ledger")
            rows = db.execute("SELECT id,cutoff,created_at FROM observations ORDER BY cutoff").fetchall()
            if len(rows) > 100_000:
                raise ValueError("too many observations")
            parsed = [(ident, _time(t), _time(created)) for ident, t, created in rows]
            if any(t < START or t != t.replace(minute=0, second=0, microsecond=0)
                   or not timedelta(0) <= created - t <= timedelta(minutes=50) or created > now
                   for _, t, created in parsed):
                raise ValueError("invalid observation timing")
            times = [t for _, t, _ in parsed]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("duplicate or unordered observation")
            if not parsed:
                result.update(status="ok", observations=0, current_hour_status="not_started" if cutoff < START else
                              "open_unrecorded" if now - cutoff <= timedelta(minutes=50) else "window_closed")
                return result
            ident, first, last = parsed[-1][0], times[0], times[-1]
            source = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
            if not source:
                raise ValueError("missing observation")
            snapshot = json.loads(source[0])
            if (not isinstance(snapshot, dict) or snapshot.get("cutoff_utc") != rows[-1][1]
                    or snapshot.get("created_at_utc") != rows[-1][2] or snapshot.get("order_permissions") is not False):
                raise ValueError("observation metadata mismatch")
            source_hash = hashlib.sha256(source[0].encode("utf-8")).hexdigest()
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            steps = {}
            for label, (claim_table, event_table, allowed) in STEPS.items():
                claim = db.execute(f"SELECT started_at,observation_sha256 FROM {claim_table} WHERE observation_id=?", (ident,)).fetchone() if claim_table in tables else None
                event_row = db.execute(f"SELECT created_at,payload_json FROM {event_table} WHERE observation_id=?", (ident,)).fetchone() if event_table in tables else None
                if not claim and not event_row:
                    steps[label] = "not_attempted"
                elif claim and not event_row:
                    try:
                        started = _time(claim[0])
                        steps[label] = ("attempted_without_result" if claim[1] == source_hash
                                        and parsed[-1][2] <= started <= now and started < last + timedelta(hours=1)
                                        else "invalid_binding")
                    except (TypeError, ValueError, AttributeError):
                        steps[label] = "invalid_binding"
                elif not claim:
                    steps[label] = "invalid_binding"
                else:
                    try:
                        event = json.loads(event_row[1])
                        started, created = _time(claim[0]), _time(event_row[0])
                        if (claim[1] != source_hash or event.get("observation_sha256") != source_hash
                                or event.get("observation_id") != ident or event.get("created_at_utc") != event_row[0]
                                or event.get("order_permissions") is not False or event.get("status") not in allowed
                                or not parsed[-1][2] <= started <= created < last + timedelta(hours=1) or created > now):
                            raise ValueError("invalid step binding")
                        steps[label] = "recorded_" + event["status"]
                    except (KeyError, TypeError, ValueError, AttributeError):
                        steps[label] = "invalid_binding"
            missing_inside = int((last - first).total_seconds() // 3600 + 1 - len(times))
            segments = []
            segment_start = times[0]
            count = 1
            for previous, current in zip(times, times[1:]):
                if current - previous == timedelta(hours=1):
                    count += 1
                else:
                    segments.append({"start_utc": segment_start.isoformat(), "end_utc": previous.isoformat(), "observations": count})
                    segment_start, count = current, 1
            segments.append({"start_utc": segment_start.isoformat(), "end_utc": times[-1].isoformat(), "observations": count})
            result.update(status="ok", observations=len(times), first_cutoff_utc=first.isoformat(),
                          last_cutoff_utc=last.isoformat(), missing_hours_before_first=int((first - START).total_seconds() // 3600),
                          missing_hours_within_recorded_span=missing_inside,
                          current_hour_status="recorded" if cutoff in times else
                          "not_started" if cutoff < START else
                          "open_unrecorded" if now - cutoff <= timedelta(minutes=50) else "window_closed",
                          latest_steps=steps, minimum_three_contiguous=(len(times) >= 3 and missing_inside == 0),
                          contiguous_segments=segments, latest_consecutive_hours=segments[-1]["observations"],
                          segmentation_note="Coverage only; no per-segment accounts or returns are calculated, and a gap cannot be backfilled.")
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError, AttributeError, OverflowError):
        result.update(status="unavailable", reason="ledger_or_metadata_invalid_or_unavailable")
    return result
