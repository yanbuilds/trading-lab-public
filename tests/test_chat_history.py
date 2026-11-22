import json
import sqlite3

from piia.intelligence.chat_history import recent_chat_research


def test_chat_history_reads_only_completed_chat_records_without_creating_db(tmp_path):
    path = tmp_path / "history.sqlite"
    assert recent_chat_research(path) == []
    assert not path.exists()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE research_runs (id TEXT, created_at TEXT, input_hash TEXT, "
                   "payload_json TEXT, result_json TEXT, mode TEXT, status TEXT)")
        payload = json.dumps({"symbol": "BTCUSDT", "cutoff_at_utc": "2026-01-01T00:00:00Z"})
        result = json.dumps({"independent_llm_review": "NOT_WIRED", "critic": {"summary": "uncertain"},
                             "synthesizer": {"summary": "wait", "suggested_action": "stand_aside"}})
        db.execute("INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                   ("1", "2026-01-01T00:01:00Z", "hash", payload, result, "chat_assisted", "completed"))
        db.execute("INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                   ("2", "2026-01-01T00:02:00Z", "hash", payload, result, "fixture", "completed"))
    before = path.read_bytes()
    records = recent_chat_research(path)
    assert len(records) == 1
    assert records[0]["snapshot_cutoff_utc"] == "2026-01-01T00:00:00Z"
    assert records[0]["critic"] == "uncertain"
    assert records[0]["independent_llm_review"] == "NOT_WIRED"
    assert path.read_bytes() == before
