"""Synthetic public data; these fixtures are not a historical performance result."""
from datetime import timedelta
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import httpx
import pandas as pd
import pytest

from experiments import funding_history, funding_observer, frozen_policy_capture as capture
from experiments import prospective_funding_realization as realization
from tests.test_prospective_paper_report import synthetic_record, specs


def payload(symbol, t, next_update, interval=8, rate="0.0001"):
    return {"code": "00000", "requestTime": int(t.timestamp() * 1000), "data": [{"symbol": symbol,
            "fundingRate": rate, "fundingRateInterval": str(interval), "nextUpdate": str(int(next_update.timestamp() * 1000))}]}


def test_current_snapshot_rejects_bad_symbol_clock_and_unaligned_update():
    t = capture.START
    received = t + timedelta(minutes=2)
    p = payload("BTCUSDT", received, t + timedelta(hours=8))
    raw = json.dumps(p).encode()
    item = funding_observer.parse_snapshot(p, symbol="BTCUSDT", cutoff=t, received=received, raw=raw)
    assert item["reported_interval_hours"] == 8 and item["next_update_is_verified_settlement"] is False
    for bad in (payload("XAUUSDT", received, t + timedelta(hours=8)),
                payload("BTCUSDT", received - timedelta(minutes=3), t + timedelta(hours=8)),
                payload("BTCUSDT", received, t + timedelta(hours=8, minutes=1))):
        with pytest.raises(ValueError):
            funding_observer.parse_snapshot(bad, symbol="BTCUSDT", cutoff=t, received=received, raw=json.dumps(bad).encode())
    for key, value in (("fundingRate", True), ("fundingRateInterval", 8.5), ("nextUpdate", True)):
        bad = copy.deepcopy(p)
        bad["data"][0][key] = value
        with pytest.raises(ValueError):
            funding_observer.parse_snapshot(bad, symbol="BTCUSDT", cutoff=t, received=received, raw=json.dumps(bad).encode())


def test_observer_is_one_attempt_and_does_not_write_real_ledger(tmp_path, monkeypatch):
    t = capture.START
    created = t + timedelta(minutes=2, seconds=1)
    record = synthetic_record(t)
    record["capture_script_sha256"] = hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(capture, "now_utc", lambda: created)
    monkeypatch.setattr(funding_observer.ai_shadow, "now_utc", lambda: created)
    path = tmp_path / "synthetic.sqlite"
    capture.save_observation(path, t, created, record)
    calls = []
    def respond(req):
        calls.append(str(req.url))
        symbol = req.url.params["symbol"]
        return httpx.Response(200, json=payload(symbol, created, t + timedelta(hours=8)), request=req)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    assert funding_observer.run(path, client=client)["status"] == "complete"
    with pytest.raises(sqlite3.IntegrityError):
        funding_observer.run(path, client=client)
    assert len(calls) == 3
    events = realization.read_funding_events(path)
    assert len(events) == 1 and events[t.isoformat()]["actual_funding_paid"] is None
    with pytest.raises(ValueError, match="cannot write"):
        funding_observer.run(funding_observer.LEDGER, client=client)


def history_payload(symbol, t, rates):
    return {"code": "00000", "requestTime": int((t + timedelta(hours=20)).timestamp() * 1000),
            "data": [{"symbol": symbol, "fundingTime": str(int((t + timedelta(hours=h)).timestamp() * 1000)),
                      "fundingRate": str(rate)} for h, rate in rates]}


def test_history_reconstructs_raw_pages_and_detects_revision():
    t = capture.START
    start, end = t + timedelta(hours=1), t + timedelta(hours=10)
    p = history_payload("BTCUSDT", t, [(16, .0002), (8, .0001), (0, -.0001)])
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=p, request=req)))
    history = funding_history.fetch_history("BTCUSDT", start, end, client=client)
    assert funding_history.validate_history(history, "BTCUSDT", start, end) == [{"timestamp": int((t + timedelta(hours=8)).timestamp() * 1000), "rate": .0001}]
    bad = copy.deepcopy(history); bad["records"][0]["rate"] = .7
    with pytest.raises(ValueError, match="selected records mismatch"):
        funding_history.validate_history(bad, "BTCUSDT", start, end)
    bad = copy.deepcopy(history); bad["pages"][0]["raw_json"] += " "
    with pytest.raises(ValueError, match="integrity"):
        funding_history.validate_history(bad, "BTCUSDT", start, end)


def test_shifted_history_page_refuses_to_report():
    t = capture.START
    calls = []
    def handler(req):
        calls.append(1)
        rate = .0001 if len(calls) == 1 else .0002
        return httpx.Response(200, json=history_payload("BTCUSDT", t, [(8, rate), (0, 0.0)]), request=req)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="shifted or revised"):
        funding_history.fetch_history("BTCUSDT", t + timedelta(hours=1), t + timedelta(hours=8), client=client)


def test_history_paginates_boundary_and_rejects_overlapping_pages():
    t = capture.START
    start, end = t + timedelta(hours=1), t + timedelta(hours=150)
    calls = []
    def handler(req):
        page = int(req.url.params["pageNo"])
        calls.append(page)
        hours = range(200, 100, -1) if page == 1 else range(100, 0, -1)
        p = {"code": "00000", "requestTime": int((t + timedelta(hours=201)).timestamp() * 1000),
             "data": [{"symbol": "BTCUSDT", "fundingRate": "0.0001",
                       "fundingTime": str(int((t + timedelta(hours=h)).timestamp() * 1000))} for h in hours]}
        return httpx.Response(200, json=p, request=req)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    history = funding_history.fetch_history("BTCUSDT", start, end, client=client)
    assert calls == [1, 2, 1] and len(funding_history.validate_history(history, "BTCUSDT", start, end)) == 150
    def overlap(req):
        page = int(req.url.params["pageNo"])
        hours = range(200, 100, -1) if page == 1 else range(101, 1, -1)
        p = {"code": "00000", "requestTime": int((t + timedelta(hours=201)).timestamp() * 1000),
             "data": [{"symbol": "BTCUSDT", "fundingRate": "0.0001",
                       "fundingTime": str(int((t + timedelta(hours=h)).timestamp() * 1000))} for h in hours]}
        return httpx.Response(200, json=p, request=req)
    client = httpx.Client(transport=httpx.MockTransport(overlap))
    with pytest.raises(ValueError, match="overlapping"):
        funding_history.fetch_history("BTCUSDT", start, end, client=client)


@pytest.mark.parametrize("side", [1, -1])
def test_actual_public_rate_is_applied_as_approximation_with_unverified_net(side, monkeypatch):
    t = capture.START
    records = [synthetic_record(t + timedelta(hours=i)) for i in range(12)]
    idx = pd.date_range(t - timedelta(hours=300), periods=312, freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {s: frame.copy() for s in capture.SYMBOLS}
    states = {name: {s: {} for s in capture.SYMBOLS} for name in ("candidate", "ema")}
    events = {}
    script = hashlib.sha256(Path(funding_observer.__file__).read_bytes()).hexdigest()
    protocol = hashlib.sha256((capture.ROOT / "FUNDING_OBSERVER_V1.md").read_bytes()).hexdigest()
    for i, record in enumerate(records):
        cutoff = t + timedelta(hours=i)
        received = cutoff + timedelta(minutes=3)
        next_update = t + timedelta(hours=8 if i < 8 else 16)
        snapshots = {}
        for symbol in capture.SYMBOLS:
            for name in states:
                states[name][symbol][cutoff] = {"signal": side if symbol == "BTCUSDT" and i == 0 else 0,
                                                "atr": 2., "exit_long": False, "exit_short": False}
            p = payload(symbol, received, next_update)
            snapshots[symbol] = funding_observer.parse_snapshot(p, symbol=symbol, cutoff=cutoff, received=received, raw=json.dumps(p).encode())
        events[record["cutoff_utc"]] = {"status": "complete", "errors": {}, "order_permissions": False,
            "created_at_utc": received.isoformat(), "observer_script_sha256": script, "protocol_sha256": protocol,
            "snapshots": snapshots}
    start, end = t + timedelta(hours=1), t + timedelta(hours=10)
    histories = {}
    for symbol in capture.SYMBOLS:
        p = history_payload(symbol, t, [(16, .0002), (8, .0001), (0, -.0001)])
        client = httpx.Client(transport=httpx.MockTransport(lambda req, p=p: httpx.Response(200, json=p, request=req)))
        histories[symbol] = funding_history.fetch_history(symbol, start, end, client=client)
    monkeypatch.setattr(realization.report, "prepare_records", lambda _: (frames, states, t, t + timedelta(hours=11)))
    out = realization.evaluate(records, events, histories, specs())
    account = out["accounts"]["candidate"]
    assert account["estimated_funding_paid_usdt"] * side > 0
    assert account["funding_net_paid"] is None and account["funding_coverage_checked"] is False
    assert out["order_permissions"] is False and "not_actual" in out["status"]
    missing = dict(events); missing.pop(records[5]["cutoff_utc"])
    with pytest.raises(ValueError, match="missing or failed"):
        realization.evaluate(records, missing, histories, specs())
    corrupted = copy.deepcopy(events)
    corrupted[records[3]["cutoff_utc"]]["snapshots"]["BTCUSDT"]["current_rate"] = .9
    with pytest.raises(ValueError, match="content mismatch"):
        realization.evaluate(records, corrupted, histories, specs())
    changed = copy.deepcopy(events)
    cutoff = t + timedelta(hours=3)
    received = cutoff + timedelta(minutes=3)
    p = payload("BTCUSDT", received, t + timedelta(hours=4), interval=4)
    changed[records[3]["cutoff_utc"]]["snapshots"]["BTCUSDT"] = funding_observer.parse_snapshot(
        p, symbol="BTCUSDT", cutoff=cutoff, received=received, raw=json.dumps(p).encode())
    with pytest.raises(ValueError, match="interval changed"):
        realization.evaluate(records, changed, histories, specs())
    incomplete = dict(histories)
    p = history_payload("BTCUSDT", t, [(16, .0002), (0, -.0001)])
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=p, request=req)))
    incomplete["BTCUSDT"] = funding_history.fetch_history("BTCUSDT", start, end, client=client)
    with pytest.raises(ValueError, match="missing or invalid"):
        realization.evaluate(records, events, incomplete, specs())
