"""Synthetic issuer-derived estimate fixtures; not observed market flows."""
from datetime import datetime, timezone
import hashlib

import httpx
import pytest

from piia.intelligence.etf_xoomar import fetch_partial_etf, verify_partial_etf

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)


def client(payload):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload, request=request)))


def test_partial_funds_are_never_full_market_flows(tmp_path):
    payload = {"updatedAt": "2026-11-17T22:00:00Z", "data": [
        {"date": "2026-11-17", "ticker": "ARKB", "asset": "btc", "issuer": "ARK", "flowUsd": "-2.50"},
        {"date": "2026-11-16", "ticker": "IBIT", "asset": "btc", "issuer": "iShares", "flowUsd": None}]}
    out = fetch_partial_etf(client=client(payload), now=NOW, snapshot_dir=tmp_path)
    assert out["available"] and not out["complete"] and not out["usable_for_btc_research"]
    assert out["missing_expected_funds"] == ["BITB"]
    assert out["latest_by_fund"]["IBIT"]["estimated_flow_usd"] is None
    assert "total" not in out
    assert hashlib.sha256((tmp_path / out["raw_snapshot_file"]).read_bytes()).hexdigest() == out["raw_sha256"]
    verify_partial_etf(out, tmp_path)
    bad = {**out, "latest_by_fund": {}}
    with pytest.raises(ValueError, match="reproduce"):
        verify_partial_etf(bad, tmp_path)
    (tmp_path / out["raw_snapshot_file"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_partial_etf(out, tmp_path)


def test_future_or_duplicate_data_fails_closed():
    row = {"date": "2026-11-17", "ticker": "IBIT", "asset": "btc", "issuer": "iShares", "flowUsd": "4"}
    for payload in ({"updatedAt": "2026-11-19T00:00:00Z", "data": [row]},
                    {"updatedAt": "2026-11-17T22:00:00Z", "data": [row, row]}):
        out = fetch_partial_etf(client=client(payload), now=NOW)
        assert not out["available"] and not out["usable_for_btc_research"]
