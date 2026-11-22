import pandas as pd
import pytest

from piia.intelligence import historical_phase2


def test_incomplete_historical_funding_stops_before_replay_database_is_touched(tmp_path, monkeypatch):
    symbol = "BTC/USDT:USDT"
    provenance = {"symbols": {symbol: {"funding_interval_hours": 8}}}
    monkeypatch.setattr(historical_phase2, "load_cache", lambda _: ({symbol: pd.DataFrame()}, {}, {symbol: []}, provenance))
    first, last = "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z"
    sealed = {"events": [{"packet_id": "x", "sequence": 0, "decision_at_utc": first,
                          "horizon_exit_at_utc": last}]}
    bundle = ({"decisions": [{"packet_id": "x"}]}, {"packets": [{"packet_id": "x"}]}, sealed, "digest")
    db = tmp_path / "replay.sqlite"
    with pytest.raises(historical_phase2.FrozenDecisionError, match="funding missing"):
        historical_phase2.run_phase2(data_dir=tmp_path, db_path=db, evidence_store=tmp_path / "evidence",
                                     decisions_bundle=bundle)
    assert not db.exists()


def test_existing_historical_replay_database_is_never_overwritten(tmp_path):
    db = tmp_path / "replay.sqlite"
    db.write_bytes(b"existing replay data")
    with pytest.raises(FileExistsError, match="choose a new path"):
        historical_phase2.run_phase2(db_path=db, evidence_store=tmp_path / "evidence")
    assert db.read_bytes() == b"existing replay data"


def test_gate_preflights_both_replay_paths_before_starting(tmp_path):
    from piia.intelligence.htf_alignment_gate import run_experiment
    existing = tmp_path / "existing.sqlite"
    existing.write_bytes(b"keep")
    fresh = tmp_path / "fresh.sqlite"
    with pytest.raises(FileExistsError, match="new paths"):
        run_experiment(manual_db=fresh, independent_db=existing)
    assert not fresh.exists()
    assert existing.read_bytes() == b"keep"
    with pytest.raises(ValueError, match="distinct"):
        run_experiment(manual_db=fresh, independent_db=fresh)
