from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from piia.intelligence.db import make_engine, session_factory
from piia.intelligence.hashing import blob_path
from piia.intelligence.historical_independent import (
    EXPECTED_PACKET_BYTES_SHA256,
    FORBIDDEN_DB,
    OUT_DIR,
    PRIOR_DECISIONS,
    PRIOR_PHASE2_DB,
    RAW_DIR,
    REVIEW_MARKER,
    ROLE_FILES,
    IndependentRoleError,
    _load_role,
    funding_inclusive_entry_prepare,
    load_independent_roles,
)
from piia.intelligence.historical_phase2 import sha256_file
from piia.intelligence.models import Analysis, EvidenceItem, ResearchRun, Thesis
from trading_lab.perps.risk import floor_step, size_order

FIXTURE_LABEL = "FIXTURE_NOT_REAL_MODEL_TRADE"


def _legacy_independent_artifacts_available() -> bool:
    from piia.intelligence.historical_phase2 import PACKET_DIR
    packet = PACKET_DIR / "PACKETS_FOR_REVIEW.json"
    return ROLE_FILES["researcher"]["default"].exists() and packet.is_file()


legacy_independent = pytest.mark.skipif(
    not _legacy_independent_artifacts_available(),
    reason="legacy historical independent role files or PACKETS_FOR_REVIEW.json are not in this worktree; opt-in only, artifacts are not fabricated",
)


def test_fixture_role_hash_mismatch_and_rename_fail_closed(tmp_path):
    researcher = tmp_path / "researcher.json"
    researcher.write_text('{"role": "researcher", "decisions": [{"packet_id": "P-01", "direction": "flat", "summary": "s", "invalidation": "i"}], "label": "%s"}' % FIXTURE_LABEL, encoding="utf-8")
    from piia.intelligence.historical_phase2 import sha256_file
    digest = sha256_file(researcher)
    payload, got = _load_role(researcher, "researcher", digest)
    assert payload["role"] == "researcher"
    assert got == digest
    with pytest.raises(IndependentRoleError, match="hash mismatch"):
        _load_role(researcher, "researcher", "0" * 64)
    with pytest.raises(IndependentRoleError, match="role field"):
        _load_role(researcher, "critic", digest)


def test_fixture_funding_inclusive_gate_does_not_round_up():
    decision = {"synthesizer": {"direction": "short", "asset": "Asset-C", "suggested_action": "review_paper_plan"}}
    event = {"label": FIXTURE_LABEL, "packet_id": "FIXTURE-P06"}
    gate = funding_inclusive_entry_prepare(
        raw_open=28232.0, stop_distance=198.18802419906413, side="short",
        symbol="NDX100USDT", packet_id="FIXTURE-P06", decision=decision, event=event,
    )
    assert gate["proceed"] is True
    assert gate["max_qty"] is not None
    assert gate["max_qty"] < gate["notes"]["unconstrained_qty"]
    assert gate["max_qty"] <= gate["notes"]["unconstrained_qty"]
    assert gate["notes"]["known_cost_plus_reserve"] <= 0.30 + 1e-9
    assert gate["notes"]["reserve_is_not_a_future_funding_forecast"] is True


def test_fixture_protected_database_path_is_refused(tmp_path, monkeypatch):
    from piia.intelligence import historical_independent as hi
    forbidden = tmp_path / "piia.dev.sqlite"
    forbidden.write_bytes(b"fixture-not-live-db")
    monkeypatch.setattr(hi, "FORBIDDEN_DB", forbidden)
    with pytest.raises(ValueError, match="refusing a protected research database"):
        hi.run_independent(db_path=forbidden, packet_dir=tmp_path)


@legacy_independent
def test_role_hashes_and_source_packet_match():
    loaded = load_independent_roles()
    for role, meta in ROLE_FILES.items():
        assert loaded["hashes"][role] == meta["sha256"]
    assert loaded["packet_bytes_sha256"] == EXPECTED_PACKET_BYTES_SHA256
    assert loaded["decisions"]["independent_llm_review"] == REVIEW_MARKER
    assert loaded["decisions"]["official_model_api"] == "not_configured"
    dirs = [d["synthesizer"]["direction"] for d in loaded["decisions"]["decisions"]]
    assert dirs.count("flat") == 8
    assert sum(d in {"long", "short"} for d in dirs) == 2
    shorts = [d for d in loaded["decisions"]["decisions"] if d["synthesizer"]["direction"] == "short"]
    assert {(s["packet_id"], s["synthesizer"]["asset"]) for s in shorts} == {("P-01", "Asset-A"), ("P-06", "Asset-C")}


@legacy_independent
def test_role_rename_cannot_fake_independence():
    critic = ROLE_FILES["critic"]["default"]
    # Same bytes can pass the critic hash, but cannot be loaded as researcher.
    with pytest.raises(IndependentRoleError, match="role field"):
        _load_role(critic, "researcher", ROLE_FILES["critic"]["sha256"])


@legacy_independent
def test_incomplete_roles_fail_closed(tmp_path):
    raw = json.loads(ROLE_FILES["synthesizer"]["default"].read_text())
    raw["decisions"] = [d for d in raw["decisions"] if d["packet_id"] != "P-08"]
    path = tmp_path / "synth_incomplete.json"
    path.write_text(json.dumps(raw))
    # hash will also fail first; either error is fail-closed
    with pytest.raises(IndependentRoleError):
        load_independent_roles(synthesizer_path=path)


@legacy_independent
def test_wrong_bytes_hash_fail_closed(tmp_path):
    src = ROLE_FILES["researcher"]["default"]
    path = tmp_path / "researcher_tampered.json"
    path.write_text(src.read_text() + "\n")
    with pytest.raises(IndependentRoleError, match="hash mismatch"):
        load_independent_roles(researcher_path=path)


@legacy_independent
def test_funding_inclusive_gate_does_not_round_up():
    # P-06 at the known next open would exceed 0.30 if sized to the unconstrained 1% budget;
    # the gate must cap quantity downward or stay flat, never enlarge a lot.
    from piia.intelligence.historical_packets import ASSET_MAP, PAIR, STOP_ATR_MULT
    from trading_lab.perps.compare import load_cache
    from piia.intelligence.paths import ROOT

    sealed = json.loads((ROOT / "artifacts/research_packets/historical_v1/BACKEND_SEALED.json").read_text())
    ev = next(e for e in sealed["events"] if e["packet_id"] == "P-06")
    ts = pd.Timestamp(ev["decision_at_utc"])
    frames, *_ = load_cache(ROOT / "data/perps-extended")
    raw_open = float(frames[PAIR["NDX100USDT"]].at[ts, "open"])
    atr = float(ev["assets"]["Asset-C"]["atr14"])
    decision = {
        "synthesizer": {"direction": "short", "asset": "Asset-C", "suggested_action": "review_paper_plan"},
    }
    gate = funding_inclusive_entry_prepare(
        raw_open=raw_open, stop_distance=STOP_ATR_MULT * atr, side="short",
        symbol="NDX100USDT", packet_id="P-06", decision=decision, event=ev,
    )
    assert gate["proceed"] is True
    assert gate["max_qty"] == 0.0009
    assert gate["max_qty"] < gate["notes"]["unconstrained_qty"]
    assert gate["notes"]["known_cost_plus_reserve"] <= 0.30 + 1e-9


@legacy_independent
def test_independent_results_invariants():
    path = OUT_DIR / "PHASE2_RESULTS.json"
    pytest.skip("independent results missing") if not path.exists() else None
    report = json.loads(path.read_text())
    assert report["independent_llm_review"] == REVIEW_MARKER
    assert report["official_model_api"] == "not_configured"
    assert report["held_out"] is False
    assert report["already_seen_dataset"] is True
    assert report["authorship"]["user_approvals"] is False
    assert report["authorship"]["live_orders"] is False
    assert report["isolation_boundary"]["statistically_independent"] is False
    assert report["isolation_boundary"]["program_enforced_inference_isolation"] is False
    assert report["isolation_boundary"]["tradingagents_library_integrated"] is False
    assert report["reconciliation_ok"] is True
    assert report["ledger_ok"] is True
    assert report["closed_trades"] == 2
    assert report["wins"] + report["losses"] + report["breakeven"] == report["closed_trades"]
    assert report["role_files_unmodified_after_run"] is True
    assert report["prior_decisions_unmodified_after_run"] is True
    warning = report["prior_single_assistant_comparison"]["warning"].lower()
    assert "not independent statistical validation" in warning
    assert report["prior_single_assistant_comparison"]["window_match"] is True


@legacy_independent
def test_independent_evidence_has_strict_cutoff():
    from piia.intelligence.historical_independent import DB_PATH, EVIDENCE_STORE
    pytest.skip("independent db missing") if not DB_PATH.exists() else None
    session = session_factory(make_engine(f"sqlite:///{DB_PATH.as_posix()}"))()
    try:
        assert len(list(session.scalars(select(Thesis)))) == 10
        runs = list(session.scalars(select(ResearchRun)))
        assert len(runs) == 10
        assert {r.independent_llm_review for r in runs} == {REVIEW_MARKER}
        assert {a.role for a in session.scalars(select(Analysis))} == {"researcher", "critic", "synthesizer"}
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


@legacy_independent
def test_protected_databases_and_prior_decisions_untouched():
    assert FORBIDDEN_DB.exists()
    result = OUT_DIR / "PHASE2_RESULTS.json"
    if result.exists():
        assert FORBIDDEN_DB.stat().st_mtime < result.stat().st_mtime
        assert PRIOR_DECISIONS.stat().st_mtime < result.stat().st_mtime
        if PRIOR_PHASE2_DB.exists():
            assert PRIOR_PHASE2_DB.stat().st_mtime < result.stat().st_mtime
    assert sha256_file(PRIOR_DECISIONS) == "d881a688aba853b7c8cd05f2028ba0bfe4a1ffcedfa0bb8abc42fe61624888ae"
    if RAW_DIR.exists():
        for role, meta in ROLE_FILES.items():
            assert sha256_file(RAW_DIR / f"{role}.json") == meta["sha256"]


def test_engine_max_qty_never_rounds_up():
    from piia.intelligence.paper.engine import PaperEngine, spec_for
    from trading_lab.perps.risk import floor_step
    spec = spec_for("NDX100USDT")
    qty, reason = size_order(30, 30, 28232.0, 198.18802419906413, spec, leverage=2, risk_fraction=0.01, cash_floor=0.2, max_margin_fraction=0.5, slippage=0.0005)
    assert reason == "ok"
    cap = floor_step(0.0009, spec.qty_step)
    assert cap <= qty
    assert cap == 0.0009
