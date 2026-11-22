from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytest.importorskip("piia")

import httpx

from piia.intelligence.etf_flows import (
    EXPECTED_FUNDS,
    fetch_us_spot_btc_etf_flows,
    parse_farside_btc_html,
    parse_flow_value,
)
from piia.intelligence.evidence_gates import apply_external_missing, recommendation_blockers
from piia.intelligence.onchain_btc import REQUIRED_PARTS, fetch_btc_onchain
from piia.intelligence.research_packet import build_request


CUTOFF = datetime(2026, 11, 17, 2, 20, tzinfo=timezone.utc)

FARSIDE_HTML = """<!doctype html><html><body>
<p>Today's date is 17 November 2026</p>
<table>
<tr><th>Date</th><th>IBIT</th><th>FBTC</th><th>BITB</th><th>ARKB</th><th>BTCO</th><th>EZBC</th><th>BRRR</th><th>HODL</th><th>BTCW</th><th>MSBT</th><th>GBTC</th><th>BTC</th><th>Total</th></tr>
<tr><td>16 Nov 2026</td><td>-</td><td>(11.3)</td><td>(22.9)</td><td>(16.6)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>(7.4)</td><td>0.0</td><td>0.0</td><td>(17.8)</td><td>7.4</td><td>(68.6)</td></tr>
<tr><td>13 Nov 2026</td><td>12.8</td><td>(29.3)</td><td>(44.2)</td><td>(31.7)</td><td>0.0</td><td>0.0</td><td>(2.9)</td><td>(16.2)</td><td>0.0</td><td>0.0</td><td>(70.9)</td><td>(3.0)</td><td>(185.4)</td></tr>
<tr><td>Total</td><td>64193.4</td><td>19212.1</td><td>2460.9</td><td>2242.8</td><td>218.8</td><td>357.8</td><td>80.6</td><td>357.1</td><td>244.9</td><td>1.5</td><td>(23694.8)</td><td>242.6</td><td>65917.7</td></tr>
</table>
<p>Automatically generated. Farside Investors is not responsible or liable for any errors, omissions or inaccuracies.</p>
</body></html>
"""


class FakeResponse:
    def __init__(self, content, url, status=200):
        self.content = content if isinstance(content, bytes) else content.encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.status_code = status
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("GET", str(self.url)), response=httpx.Response(self.status_code))

    def json(self):
        return json.loads(self.content)


class FakeClient:
    def __init__(self, mapping, default_url="https://mempool.space"):
        self.mapping = mapping
        self.default_url = default_url

    def get(self, url):
        path = urlparse(url).path
        if path not in self.mapping:
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        body = self.mapping[path]
        if isinstance(body, FakeResponse):
            return body
        return FakeResponse(body, url)


def _onchain_payloads(*, height=976207, tip_hash="ab" * 32, ts=1794881691, drop=None):
    drop = drop or []
    block = {
        "id": tip_hash, "height": height, "timestamp": ts, "tx_count": 10,
        "weight": 1000, "version": 1, "bits": 1, "difficulty": 1,
        "mediantime": ts, "merkle_root": "00", "nonce": 1, "previousblockhash": "cd" * 32, "size": 100,
    }
    mapping = {
        "/api/blocks/tip/height": str(height),
        "/api/blocks/tip/hash": tip_hash,
        f"/api/block/{tip_hash}": json.dumps(block),
        "/api/mempool": json.dumps({"count": 100, "vsize": 2000, "total_fee": 3000}),
        "/api/v1/fees/recommended": json.dumps({
            "fastestFee": 1, "halfHourFee": 1, "hourFee": 1, "economyFee": 1, "minimumFee": 1,
        }),
        "/api/v1/difficulty-adjustment": json.dumps({
            "progressPercent": 10.0, "difficultyChange": -1.2, "remainingBlocks": 1800,
            "nextRetargetHeight": 978336, "estimatedRetargetDate": 1795746998816, "previousTime": 1794632570,
        }),
        "/api/v1/mining/hashrate/3d": json.dumps({
            "currentHashrate": 9.675e20, "currentDifficulty": 1.348e14,
        }),
    }
    for path in drop:
        mapping.pop(path, None)
    return mapping, tip_hash, height


def test_flow_parser_treats_dash_as_unreported_not_zero():
    assert parse_flow_value("-") is None
    assert parse_flow_value("0.0") == 0.0
    assert parse_flow_value("(11.3)") == pytest.approx(-11.3)
    assert parse_flow_value("12.8") == pytest.approx(12.8)
    parsed = parse_farside_btc_html(FARSIDE_HTML)
    assert parsed["page_date"] == "2026-11-17"
    assert parsed["funds"] == list(EXPECTED_FUNDS)
    latest = parsed["latest_row"]
    assert latest["flow_date"] == "2026-11-16"
    assert latest["partial"] is True
    assert latest["fund_flows_millions_usd"]["IBIT"] is None
    assert latest["fund_flows_millions_usd"]["FBTC"] == pytest.approx(-11.3)
    assert parsed["latest_complete_row"]["flow_date"] == "2026-11-13"
    assert parsed["latest_complete_row"]["complete"] is True
    assert parsed["latest_complete_row"]["fund_flows_millions_usd"]["IBIT"] == pytest.approx(12.8)
    # cumulative Total row is not a daily flow
    assert all(day["flow_date"] != "Total" for day in parsed["days"])



def test_live_shaped_two_row_header_parses_dates():
    html = """<!doctype html><html><body>
    <p>Today's date is 17 November 2026</p>
    <table>
    <tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Total</td></tr>
    <tr><td></td><td>IBIT</td><td>FBTC</td><td>BITB</td><td>ARKB</td><td>BTCO</td><td>EZBC</td><td>BRRR</td><td>HODL</td><td>BTCW</td><td>MSBT</td><td>GBTC</td><td>BTC</td><td></td></tr>
    <tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>0.20%</td><td>0.21%</td><td>0.25%</td><td>0.19%</td><td>0.25%</td><td>0.20%</td><td>0.25%</td><td>0.14%</td><td>1.50%</td><td>0.15%</td><td></td></tr>
    <tr><td>16 Nov 2026</td><td>-</td><td>(11.3)</td><td>(22.9)</td><td>(16.6)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>(7.4)</td><td>0.0</td><td>0.0</td><td>(17.8)</td><td>7.4</td><td>(68.6)</td></tr>
    <tr><td>13 Nov 2026</td><td>12.8</td><td>(29.3)</td><td>(44.2)</td><td>(31.7)</td><td>0.0</td><td>0.0</td><td>(2.9)</td><td>(16.2)</td><td>0.0</td><td>0.0</td><td>(70.9)</td><td>(3.0)</td><td>(185.4)</td></tr>
    </table></body></html>
    """
    parsed = parse_farside_btc_html(html)
    assert parsed["latest_row"]["flow_date"] == "2026-11-16"
    assert parsed["latest_row"]["fund_flows_millions_usd"]["IBIT"] is None
    assert parsed["latest_complete_row"]["flow_date"] == "2026-11-13"
    assert parsed["latest_complete_row"]["table_total_millions_usd"] == pytest.approx(-185.4)


def test_etf_fetch_uses_last_complete_day_and_not_aum(tmp_path):
    client = FakeClient({
        "/btc/": FakeResponse(FARSIDE_HTML, "https://farside.co.uk/btc/"),
    })
    result = fetch_us_spot_btc_etf_flows(now=CUTOFF, client=client, snapshot_dir=tmp_path)
    assert result["complete"] is True
    assert result["usable_for_btc_research"] is True
    assert result["latest_row_is_partial"] is True
    assert result["latest_complete_flow_date"] == "2026-11-13"
    assert result["measure"] == "net_fund_flows"
    assert "aum" in result["not_measure"]
    assert result["official_sec_daily_filing"] is False
    assert result["units"] == "millions_of_usd"
    assert (tmp_path / "farside_btc.html").is_file()
    assert result["raw_sha256"] == __import__("hashlib").sha256(FARSIDE_HTML.encode()).hexdigest()


def test_etf_future_and_unclosed_complete_rows_are_not_usable():
    from piia.intelligence.etf_flows import _quality
    from datetime import datetime, timezone
    row = {"flow_date": "2026-11-17", "complete": True}
    parsed = {"latest_complete_row": row, "latest_row": row, "coverage_missing_expected_funds": []}
    # 15:00 UTC is before New York's ordinary close in November.
    assert not _quality(parsed, datetime(2026, 11, 17, 15, tzinfo=timezone.utc))["usable_for_btc_research"]
    row["flow_date"] = "2026-11-18"
    outcome = _quality(parsed, datetime(2026, 11, 17, 23, tzinfo=timezone.utc))
    assert not outcome["usable_for_btc_research"]
    assert outcome["future_or_unclosed_session"] is True


def test_etf_incomplete_coverage_is_not_complete():
    html = """<!doctype html><html><body>
    <p>Today's date is 17 November 2026</p>
    <table>
    <tr><th>Date</th><th>FBTC</th><th>BITB</th><th>ARKB</th><th>BTCO</th><th>EZBC</th><th>BRRR</th><th>HODL</th><th>BTCW</th><th>MSBT</th><th>GBTC</th><th>BTC</th><th>Total</th></tr>
    <tr><td>13 Nov 2026</td><td>(29.3)</td><td>(44.2)</td><td>(31.7)</td><td>0.0</td><td>0.0</td><td>(2.9)</td><td>(16.2)</td><td>0.0</td><td>0.0</td><td>(70.9)</td><td>(3.0)</td><td>(198.2)</td></tr>
    </table></body></html>
    """
    client = FakeClient({"/btc/": FakeResponse(html, "https://farside.co.uk/btc/")})
    result = fetch_us_spot_btc_etf_flows(now=CUTOFF, client=client)
    assert "IBIT" in result["coverage_missing_expected_funds"]
    assert result["complete"] is False
    assert result["usable_for_btc_research"] is False


def test_onchain_complete_requires_full_set_and_correct_units(tmp_path):
    mapping, tip_hash, height = _onchain_payloads()
    result = fetch_btc_onchain(now=CUTOFF, client=FakeClient(mapping), snapshot_dir=tmp_path)
    assert result["complete"] is True
    assert result["usable_for_btc_research"] is True
    assert result["metrics"]["tip_height"]["value"] == height
    assert result["metrics"]["tip_height"]["unit"] == "blocks"
    assert result["metrics"]["last_block"]["id"] == tip_hash
    assert result["metrics"]["last_block"]["observation_at_utc"].startswith("2026-11-17T02:14:51")
    assert result["metrics"]["mempool"]["unit_total_fee"] == "satoshis"
    assert result["metrics"]["recommended_fees"]["unit"] == "sat_per_vbyte"
    assert result["metrics"]["hashrate"]["unit_hashrate"] == "hashes_per_second"
    assert result["metrics"]["hashrate"]["current_hashrate_exahash_per_second_derived"] == pytest.approx(967.5)
    # millisecond retarget dates are converted, not stored raw
    assert result["metrics"]["difficulty_adjustment"]["estimated_retarget_at_utc"].startswith("2026-")
    assert set(result["present_parts"]) == set(REQUIRED_PARTS)
    assert (tmp_path / "tip_height.bin").read_text() == "976207"

    dropped, tip_hash, height = _onchain_payloads(drop=["/api/v1/mining/hashrate/3d"])
    incomplete = fetch_btc_onchain(now=CUTOFF, client=FakeClient(dropped))
    assert incomplete["complete"] is False
    assert incomplete["partial"] is True
    assert "hashrate" in incomplete["missing_parts"]
    assert incomplete["usable_for_btc_research"] is False


def test_onchain_stale_tip_is_not_usable():
    old = int((CUTOFF - timedelta(hours=5)).timestamp())
    mapping, _, _ = _onchain_payloads(ts=old)
    result = fetch_btc_onchain(now=CUTOFF, client=FakeClient(mapping))
    assert result["stale"] is True
    assert result["complete"] is False
    assert result["usable_for_btc_research"] is False


def test_gates_are_instrument_specific_and_do_not_fake_completeness():
    request = {
        "snapshot_ok": True,
        "missing_datasets": ["onchain", "btc_etf_flows", "equities_fundamentals"],
        "instruments": {s: {"last_closed_close_at_utc": CUTOFF.isoformat(),
                            "indicators_prefix_only": {"last_close": 2.0}}
                        for s in ("BTCUSDT", "XAUUSDT", "NDX100USDT")},
    }
    item = {"ok": True, "weights_ran": True, "prediction_length": 1,
            "last_close_at_utc": CUTOFF.isoformat(), "last_close": 2.0, "timeframe": "1H_closed",
            "quantile_10": [1.0], "quantile_50": [2.0], "quantile_90": [3.0]}
    forecast = {"weights_ran": True, "forecasts": {s: dict(item) for s in request["instruments"]}}
    macro = {"series": {
        "DFII10": {"stale": False}, "DTWEXBGS": {"stale": False},
        "RRPONTSYD": {"stale": False}, "WDTGAL": {"stale": False},
    }}
    missing = apply_external_missing(request["missing_datasets"], None, None)
    assert "equities_fundamentals" not in missing
    assert "nasdaq100_official_aggregate_valuation_and_breadth" in missing
    assert "gold_etf_and_physical_demand_flows" in missing
    request["missing_datasets"] = missing
    gate = recommendation_blockers(request, forecast, macro, onchain=None, etf=None)
    assert "missing_critical_evidence:BTCUSDT:onchain" in gate["instruments"]["BTCUSDT"]
    assert "missing_critical_evidence:BTCUSDT:btc_etf_flows" in gate["instruments"]["BTCUSDT"]
    assert gate["instruments"]["XAUUSDT"] == []
    assert gate["instruments"]["NDX100USDT"] == []
    assert gate["packet"] == []
    assert gate["trading_recommendation_blocked"] is True

    onchain = {"usable_for_btc_research": True, "complete": True}
    etf = {"usable_for_btc_research": True, "complete": True}
    request["missing_datasets"] = apply_external_missing(["onchain", "btc_etf_flows", "equities_fundamentals"], onchain, etf)
    assert "onchain" not in request["missing_datasets"]
    assert "btc_etf_flows" not in request["missing_datasets"]
    assert "onchain_exchange_flows_utxo_realized_cap" in request["missing_datasets"]
    open_gate = recommendation_blockers(request, forecast, macro, onchain=onchain, etf=etf)
    assert open_gate["instruments"]["BTCUSDT"] == []
    assert open_gate["trading_recommendation_blocked"] is False

    trivial = {"usable_for_btc_research": False, "complete": False, "metrics": {"tip_height": 1}}
    still_blocked = recommendation_blockers(request, forecast, macro, onchain=trivial, etf=etf)
    assert "missing_critical_evidence:BTCUSDT:onchain" in still_blocked["instruments"]["BTCUSDT"]


def test_onchain_and_etf_contexts_are_bound_into_request_hash():
    from tests.test_research_cycle import _snapshot
    snap = _snapshot()
    snap["onchain_context"] = {"retrieved_at_utc": CUTOFF.isoformat(), "complete": True, "metrics": {"tip_height": {"value": 1}}}
    snap["etf_flow_context"] = {"retrieved_at_utc": CUTOFF.isoformat(), "complete": True, "asof_complete_flow_date": "2026-11-13"}
    req = build_request(snap)
    assert req["instruments"]["BTCUSDT"]["onchain_context_hash"]
    assert req["instruments"]["BTCUSDT"]["etf_flow_context_hash"]
    original = req["request_hash"]
    req["onchain_context"]["metrics"]["tip_height"]["value"] = 2
    # recompute would change; stored hash must not match mutated context
    from piia.intelligence.hashing import content_hash as ch
    from piia.intelligence.research_packet import _canonical
    assert ch(_canonical(req["onchain_context"])) != req["instruments"]["BTCUSDT"]["onchain_context_hash"]
    # restore and confirm hash stable when untouched
    req["onchain_context"]["metrics"]["tip_height"]["value"] = 1
    assert req["request_hash"] == original


def test_live_public_feeds_real_fetch(tmp_path):
    if os.environ.get("PIIA_SKIP_NETWORK") == "1":
        pytest.skip("network fetch skipped")
    now = datetime.now(timezone.utc)
    onchain = fetch_btc_onchain(now=now, snapshot_dir=tmp_path / "onchain")
    etf = fetch_us_spot_btc_etf_flows(now=now, snapshot_dir=tmp_path / "etf")
    proof = tmp_path / "live_proof.json"
    proof.write_text(json.dumps({
        "retrieved_at_utc": now.isoformat(),
        "onchain_complete": onchain.get("complete"),
        "onchain_usable": onchain.get("usable_for_btc_research"),
        "onchain_tip": (onchain.get("metrics") or {}).get("tip_height"),
        "onchain_last_block": (onchain.get("metrics") or {}).get("last_block"),
        "onchain_errors": onchain.get("errors"),
        "onchain_source": onchain.get("source_docs"),
        "etf_complete": etf.get("complete"),
        "etf_usable": etf.get("usable_for_btc_research"),
        "etf_page_date": etf.get("page_date"),
        "etf_latest_flow_date": etf.get("latest_flow_date"),
        "etf_latest_complete_flow_date": etf.get("latest_complete_flow_date"),
        "etf_latest_row_is_partial": etf.get("latest_row_is_partial"),
        "etf_funds": etf.get("funds"),
        "etf_coverage_missing": etf.get("coverage_missing_expected_funds"),
        "etf_source": etf.get("source_url"),
        "etf_errors": etf.get("errors"),
        "etf_raw_sha256": etf.get("raw_sha256"),
    }, indent=2, default=str), encoding="utf-8")
    assert onchain.get("errors") in ({}, None) or onchain.get("metrics")
    assert "tip_height" in (onchain.get("metrics") or {})
    assert etf.get("source_url") == "https://farside.co.uk/btc/"
    assert "IBIT" in (etf.get("funds") or [])
    # The live proof must exist; completeness depends on contemporaneous reporting lag.
    assert proof.is_file()
