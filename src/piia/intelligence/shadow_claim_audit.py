"""Read-only audit of prospective shadow claims; never proves model accuracy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from piia.intelligence.report_ledger_check import APP_ID

TABLES = {
    "ai_shadow": ("ai_shadow_claims", "ai_shadow_events"),
    "chronos_shadow": ("chronos_shadow_claims", "chronos_shadow_events"),
    "chronos2_shadow": ("chronos2_shadow_claims", "chronos2_shadow_events"),
}


def _time(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timezone required")
    return t.astimezone(timezone.utc)


def audit_claims(path: Path, arm: str, *, now: datetime | None = None, cutoffs: list[str] | None = None) -> dict:
    """Require one matching prior claim and event for every recorded observation."""
    if arm not in TABLES:
        raise ValueError("unknown shadow arm")
    result = {"ok": False, "arm": arm, "observations_checked": 0,
              "independent_provenance_verified": False, "model_output_verified": False,
              "note": "Checks local claim binding and timing only; not model accuracy or trustworthy timestamp proof."}
    try:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("timezone required")
        path = Path(path)
        if not path.is_file():
            raise ValueError("ledger unavailable")
        claim, event = TABLES[arm]
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=2) as db:
            db.execute("BEGIN")
            if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
                raise ValueError("unrelated ledger")
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"observations", claim, event}.issubset(tables):
                raise ValueError("observation, claim, or event table missing")
            rows = db.execute(f"SELECT o.id,o.cutoff,o.created_at,o.payload_json,c.started_at,c.observation_sha256,"
                              f"e.created_at,e.payload_json FROM observations o "
                              f"LEFT JOIN {claim} c ON c.observation_id=o.id "
                              f"LEFT JOIN {event} e ON e.observation_id=o.id ORDER BY o.cutoff").fetchall()
        if not rows or len(rows) > 100_000:
            raise ValueError("observation count invalid")
        if cutoffs is not None:
            if not isinstance(cutoffs, list) or not cutoffs or len(set(cutoffs)) != len(cutoffs):
                raise ValueError("invalid requested cutoffs")
            chosen = set(cutoffs)
            rows = [row for row in rows if row[1] in chosen]
            if len(rows) != len(chosen) or {row[1] for row in rows} != chosen:
                raise ValueError("requested observation missing")
        for ident, cutoff_text, observed_text, raw, started_text, claim_hash, created_text, event_raw in rows:
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            observation = json.loads(raw)
            payload = json.loads(event_raw)
            cutoff, observed, started, created = map(_time, (cutoff_text, observed_text, started_text, created_text))
            if (observation.get("cutoff_utc") != cutoff_text or observation.get("created_at_utc") != observed_text
                    or claim_hash != digest or payload.get("observation_sha256") != digest
                    or payload.get("observation_id") != ident or payload.get("created_at_utc") != created_text
                    or payload.get("order_permissions") is not False
                    or not cutoff <= observed <= started <= created < cutoff + timedelta(hours=1)
                    or created > now):
                raise ValueError("claim or event binding invalid")
        result.update(ok=True, observations_checked=len(rows))
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, AttributeError, OverflowError):
        result["reason"] = "missing_invalid_or_unbound_claim_or_event"
    return result
