"""Read previously imported chat research without creating or modifying a database."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def recent_chat_research(path: Path, limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("history limit must be between 1 and 100")
    if not path.is_file():
        return []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT created_at, input_hash, payload_json, result_json FROM research_runs "
            "WHERE mode = ? AND status = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            ("chat_assisted", "completed", limit),
        ).fetchall()
    records = []
    for created, input_hash, raw_payload, raw_result in rows:
        payload, result = json.loads(raw_payload), json.loads(raw_result)
        records.append({"recorded_at_utc": created, "snapshot_cutoff_utc": payload.get("cutoff_at_utc"),
                        "symbol": payload.get("symbol"), "request_hash": input_hash,
                        "independent_llm_review": result.get("independent_llm_review"),
                        "researcher": (result.get("researcher") or {}).get("summary"),
                        "critic": (result.get("critic") or {}).get("summary"),
                        "synthesizer": (result.get("synthesizer") or {}).get("summary"),
                        "suggested_action": (result.get("synthesizer") or {}).get("suggested_action")})
    return records
