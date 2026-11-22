from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from piia.intelligence.db import make_engine, session_factory
from piia.intelligence.hashing import blob_path
from piia.intelligence.historical_phase2 import (
    EXPECTED_DECISIONS_SHA256,
    FORBIDDEN_DB,
    PACKET_DIR,
    load_frozen,
    sha256_file,
    validate_decisions,
)
from piia.intelligence.models import Analysis, Counterargument, EvidenceItem, ResearchRun, Thesis

pytestmark = pytest.mark.skipif(
    not (PACKET_DIR / "CHAT_DECISIONS_FROZEN.json").exists(),
    reason="frozen historical packets not present",
)


def test_frozen_decisions_hash_and_schema_are_locked():
    decisions, parent, sealed, digest = load_frozen()
    assert digest == EXPECTED_DECISIONS_SHA256
    assert digest == sha256_file(PACKET_DIR / "CHAT_DECISIONS_FROZEN.json")
    validate_decisions(decisions, [p["packet_id"] for p in parent["packets"]])
    dirs = [d["synthesizer"]["direction"] for d in decisions["decisions"]]
    assert dirs.count("flat") == 4
    assert sum(d in {"long", "short"} for d in dirs) == 6
    assert sealed["contains_outcomes"] is False
    assert parent["held_out"] is False
    assert parent["independent_llm_review"] == "NOT_WIRED"


def test_phase2_results_invariants_and_no_holdout_claim():
    path = PACKET_DIR / "PHASE2_RESULTS.json"
    pytest.skip("phase-2 results missing") if not path.exists() else None
    report = json.loads(path.read_text())
    assert report["decisions_sha256"] == EXPECTED_DECISIONS_SHA256
    assert report["decisions_file_unmodified"] is True
    assert report["held_out"] is False
    assert report["already_seen_dataset"] is True
    assert report["independent_llm_review"] == "NOT_WIRED"
    assert report["authorship"]["user_approvals"] is False
    assert report["authorship"]["live_orders"] is False
    assert report["authorship"]["approval_kind"] == "harness_sandbox_only_not_user"
    assert report["reconciliation_ok"] is True
    assert report["ledger_ok"] is True
    assert abs(report["reconciliation_error"]) <= 1e-6
    assert report["directional_candidates"] == 6
    assert report["flat_decisions"] == 4
    assert report["closed_trades"] + len(report["unfilled"]) == 6
    assert report["open_positions"] == 0
    assert report["wins"] + report["losses"] + report["breakeven"] == report["closed_trades"]
    # Turtle is same window but not the same entry/horizon rule.
    note = report["turtle_s1_same_window"]["comparison_note"].lower()
    assert "not an apples-to-apples" in note
    assert report["flat_cash_benchmark"]["return_pct"] == 0.0


def test_evidence_in_sandbox_db_has_strict_cutoff():
    from piia.intelligence.historical_phase2 import DB_PATH, EVIDENCE_STORE
    pytest.skip("sandbox db missing") if not DB_PATH.exists() else None
    session = session_factory(make_engine(f"sqlite:///{DB_PATH.as_posix()}"))()
    try:
        theses = list(session.scalars(select(Thesis)))
        assert len(theses) == 10
        runs = list(session.scalars(select(ResearchRun)))
        assert len(runs) == 10
        assert {r.mode for r in runs} == {"chat_assisted"}
        assert {r.independent_llm_review for r in runs} == {"NOT_WIRED"}
        roles = {a.role for a in session.scalars(select(Analysis))}
        assert roles == {"researcher", "critic", "synthesizer"}
        assert len(list(session.scalars(select(Counterargument)))) == 10
        for ev in session.scalars(select(EvidenceItem)):
            raw = json.loads(blob_path(EVIDENCE_STORE, ev.content_hash).read_bytes())
            if isinstance(raw, str):
                raw = json.loads(raw)
            decision = pd.Timestamp(raw["decision_at_utc"])
            for last in raw["last_closed_bar_open_utc"].values():
                assert pd.Timestamp(last) < decision
            if ev.observed_at is not None:
                assert pd.Timestamp(ev.observed_at) <= decision
            blob = json.dumps(raw)
            assert "net_usd" not in blob
            assert "exit_reason" not in blob
    finally:
        session.close()


def test_live_research_db_was_not_used():
    assert FORBIDDEN_DB.exists()
    # mtime still the prospective thesis snapshot from earlier today
    mtime = FORBIDDEN_DB.stat().st_mtime
    # The live DB must remain older than the phase-2 result file if both exist.
    result = PACKET_DIR / "PHASE2_RESULTS.json"
    if result.exists():
        assert mtime < result.stat().st_mtime
    assert sha256_file(PACKET_DIR / "CHAT_DECISIONS_FROZEN.json") == EXPECTED_DECISIONS_SHA256
