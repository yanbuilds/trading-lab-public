"""Synthetic raw snapshots test binding, not the truth of on-chain data."""
from copy import deepcopy

import pytest

from piia.intelligence import onchain_btc
from piia.intelligence.onchain_integrity import verify_onchain_snapshot
from tests.test_public_research_feeds import CUTOFF, FakeClient, _onchain_payloads


def test_raw_onchain_snapshot_reproduces_metrics_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(onchain_btc.time, "sleep", lambda *_: None)
    mapping, _, _ = _onchain_payloads()
    context = onchain_btc.fetch_btc_onchain(now=CUTOFF, client=FakeClient(mapping), snapshot_dir=tmp_path)
    assert context["usable_for_btc_research"] is True
    verify_onchain_snapshot(context, tmp_path)
    changed = deepcopy(context)
    changed["metrics"]["mempool"]["count"] += 1
    with pytest.raises(ValueError, match="does not match"):
        verify_onchain_snapshot(changed, tmp_path)
    (tmp_path / "mempool.bin").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_onchain_snapshot(context, tmp_path)


def test_chat_gate_blocks_missing_saved_raw(monkeypatch, tmp_path):
    from piia.intelligence import chat_decision_gate as gate
    monkeypatch.setattr(onchain_btc.time, "sleep", lambda *_: None)
    mapping, _, _ = _onchain_payloads()
    context = onchain_btc.fetch_btc_onchain(now=CUTOFF, client=FakeClient(mapping), snapshot_dir=tmp_path / "onchain")
    request = {"snapshot_ok": True, "cutoff_at_utc": CUTOFF.isoformat(), "request_hash": "fixture",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"}, "onchain_context": context,
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": CUTOFF.isoformat()}}}}
    state = {"request_hash": "fixture", "run_id": "fixture", "paths": {"run_dir": str(tmp_path)},
             "packet_recommendation_blockers": [], "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False, "created_at_utc": CUTOFF.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check():
        return gate.assess_chat_evidence(request, "BTCUSDT", now=CUTOFF, state=state, restricted={},
                                         archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "btc_onchain_raw_snapshot_invalid" not in check()["blockers"]
    (tmp_path / "onchain" / "mempool.bin").unlink()
    assert "btc_onchain_raw_snapshot_invalid" in check()["blockers"]
