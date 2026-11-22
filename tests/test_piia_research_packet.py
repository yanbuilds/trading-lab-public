from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("piia")

from piia.intelligence.errors import ValidationFailed
from piia.intelligence.clock import FrozenClock
from piia.intelligence.paths import DEFAULT_SQLITE
from piia.intelligence.research_packet import (
    ACTOR,
    build_request,
    import_response,
    validate_response,
    PacketResponseIn,
)


CUTOFF = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)


def _candles(n=80, start=None, last=100.0):
    start = start or (CUTOFF - timedelta(hours=n))
    rows = []
    px = last - n
    for i in range(n):
        open_at = start + timedelta(hours=i)
        close_at = open_at + timedelta(hours=1)
        o, c = px, px + 0.3
        rows.append({
            "open_at_utc": open_at.isoformat(), "close_at_utc": close_at.isoformat(),
            "open": o, "high": c + 0.2, "low": o - 0.2, "close": c, "volume": 1.0,
        })
        px = c
    return rows


def _snapshot():
    candles = _candles()
    last = candles[-1]
    ticker = {
        "last": last["close"], "mark": last["close"], "index": last["close"],
        "funding_rate": "0.0001", "observed_at_utc": CUTOFF.isoformat(),
        "provenance": "fixture", "source_url": "https://api.bitget.com/fixture",
    }
    inst = {
        "instrument_id": "btcusdt-bitget-perp",
        "ticker": ticker,
        "funding": {"funding_rate": "0.0001", "provenance": "fixture"},
        "candles_closed_1h": candles,
    }
    return {
        "ok": True,
        "retrieved_at_utc": CUTOFF.isoformat(),
        "cutoff_at_utc": CUTOFF.isoformat(),
        "missing_datasets": ["macro", "onchain"],
        "errors": [],
        "instruments": {
            "BTCUSDT": inst,
            "XAUUSDT": {**inst, "instrument_id": "xauusdt-bitget-perp"},
            "NDX100USDT": {**inst, "instrument_id": "ndx100usdt-bitget-perp"},
        },
    }


def test_request_cutoff_includes_external_completion_and_partial_completion():
    snapshot = _snapshot()
    snapshot["etf_flow_context"] = {"retrieved_at_utc": CUTOFF.isoformat(),
        "completed_at_utc": (CUTOFF + timedelta(seconds=10)).isoformat(),
        "supplemental_partial": {"retrieved_at_utc": (CUTOFF + timedelta(seconds=11)).isoformat(),
                                 "completed_at_utc": (CUTOFF + timedelta(seconds=20)).isoformat()}}
    snapshot["fear_greed_context"] = {"retrieved_at_utc": CUTOFF.isoformat(),
        "completed_at_utc": (CUTOFF + timedelta(seconds=15)).isoformat()}
    request = build_request(snapshot)
    assert request["cutoff_at_utc"] == (CUTOFF + timedelta(seconds=20)).isoformat()


def _role(role, direction, action="stand_aside", stop=None, ev_hash=""):
    return {
        "role": role, "actor": ACTOR, "generation": "manual", "direction": direction,
        "summary": f"{role} writeup for a unit test. Missing macro is acknowledged.",
        "invalidation": "A close through the last closed 1h extreme would invalidate this test stance.",
        "disagreements": ["critic notes the writeup has no macro confirmation"] if role == "critic" else [],
        "missing_data": ["macro", "onchain"],
        "suggested_action": action,
        "proposed_stop": stop,
        "claims": [{"text": "uses frozen snapshot only", "evidence_hash": ev_hash, "status": "supported"}],
    }


def _response(req, direction="flat", action="stand_aside", stop=None, **over):
    insts = []
    for sym, raw in req["instruments"].items():
        ev = raw["evidence_hash"]
        insts.append({
            "symbol": sym,
            "request_evidence_hash": ev,
            "researcher": _role("researcher", direction, action, stop, ev),
            "critic": _role("critic", "flat", "need_more_evidence", None, ev),
            "synthesizer": _role("synthesizer", direction, action, stop, ev),
        })
    payload = {
        "packet_type": "piia_chat_assisted_research_response",
        "request_hash": req["request_hash"],
        "cutoff_at_utc": req["cutoff_at_utc"],
        "independent_llm_review": "NOT_WIRED",
        "actor": ACTOR,
        "instruments": insts,
    }
    payload.update(over)
    return payload


def test_build_request_is_cutoff_consistent():
    req = build_request(_snapshot())
    assert req["packet_type"] == "piia_chat_assisted_research_request"
    assert req["independent_llm_review"] == "NOT_WIRED"
    assert "macro" in req["missing_datasets"]
    for raw in req["instruments"].values():
        assert raw["indicators_prefix_only"]["ok"] is True
        assert raw["last_closed_close_at_utc"] <= req["cutoff_at_utc"]
        assert raw["indicators_prefix_only"]["rsi14_wilder"] == pytest.approx(100.0)
        assert raw["indicators_prefix_only"]["range_24h_high"] >= raw["indicators_prefix_only"]["last_close"]


def test_request_does_not_present_stale_or_future_candles_as_current_context():
    from piia.intelligence.research_packet import _daily_context
    snap = _snapshot()
    inst = snap["instruments"]["BTCUSDT"]
    baseline = _candles(220, start=CUTOFF - timedelta(hours=220), last=1000.)
    inst["candles_closed_1h"] = baseline
    assert build_request(snap)["instruments"]["BTCUSDT"]["indicators_prefix_only"]["ok"]
    snap["retrieved_at_utc"] = (CUTOFF + timedelta(hours=3)).isoformat()
    row = build_request(snap)["instruments"]["BTCUSDT"]
    assert row["indicators_prefix_only"]["ok"] is False
    assert "last_closed_candle_missing_or_stale" in row["indicators_prefix_only"]["errors"]
    assert row["market_context"]["timeframe_alignment"] == "incomplete"
    snap["retrieved_at_utc"] = (CUTOFF - timedelta(hours=2)).isoformat()
    row = build_request(snap)["instruments"]["BTCUSDT"]
    assert row["indicators_prefix_only"]["ok"] is False
    assert "last_closed_candle_missing_or_stale" in row["indicators_prefix_only"]["errors"]
    daily = []
    for i in range(60):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        daily.append({"open_at_utc": start.isoformat(), "close_at_utc": (start + timedelta(days=1)).isoformat(),
                      "open": 100., "high": 102., "low": 99., "close": 101., "volume": 1.})
    assert _daily_context(daily, [], datetime(2026, 3, 2, tzinfo=timezone.utc))["ok"]
    stale = _daily_context(daily, [], datetime(2026, 3, 5, tzinfo=timezone.utc))
    assert not stale["ok"] and "last_closed_candle_missing_or_stale" in stale["errors"]


def test_rsi_constant_series_is_undefined():
    import numpy as np
    from piia.intelligence.research_packet import _rsi_wilder
    assert _rsi_wilder(np.full(100, 10.0)) is None
    assert _rsi_wilder(np.arange(100, 0, -1, dtype=float)) == 0.0


def test_four_hour_context_excludes_incomplete_bar_and_rejects_gaps():
    from piia.intelligence.research_packet import _completed_four_hour_trend
    candles = _candles(203, start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    baseline = _completed_four_hour_trend(candles)
    assert baseline["ok"] is True
    changed = [dict(c) for c in candles]
    for c in changed[-3:]:
        c["close"] *= 2
    assert _completed_four_hour_trend(changed) == baseline
    assert _completed_four_hour_trend(candles[:50] + candles[51:])["reason"] == "duplicate_or_missing_hourly_bars"


def test_rejects_hash_and_cutoff_mismatch():
    req = build_request(_snapshot())
    bad = _response(req)
    bad["request_hash"] = "0" * 64
    with pytest.raises(ValidationFailed, match="request_hash"):
        validate_response(req, PacketResponseIn.model_validate(bad))
    bad = _response(req)
    bad["cutoff_at_utc"] = (CUTOFF + timedelta(hours=1)).isoformat()
    with pytest.raises(ValidationFailed, match="cutoff"):
        validate_response(req, PacketResponseIn.model_validate(bad))


def test_rejects_tampered_evidence_and_earlier_candles():
    import copy
    req = build_request(_snapshot())
    response = PacketResponseIn.model_validate(_response(req))
    changed = copy.deepcopy(req)
    changed["instruments"]["BTCUSDT"]["ticker"]["mark"] *= 2
    with pytest.raises(ValidationFailed, match="evidence content hash"):
        validate_response(changed, response)
    changed = copy.deepcopy(req)
    changed["instruments"]["BTCUSDT"]["all_closed_candles"][0]["close"] *= 2
    with pytest.raises(ValidationFailed, match="candle content hash"):
        validate_response(changed, response)


def test_liquidity_context_is_bound_and_available_for_claims():
    import copy
    from piia.intelligence.research_packet import frozen_hashes_for_symbol
    snap = _snapshot()
    snap["liquidity_positioning_context"] = {"retrieved_at_utc": CUTOFF.isoformat(),
        "completed_at_utc": (CUTOFF + timedelta(seconds=5)).isoformat(), "official_state": "unknown"}
    request = build_request(snap)
    response = PacketResponseIn.model_validate(_response(request))
    validate_response(request, response)
    assert request["cutoff_at_utc"] == (CUTOFF + timedelta(seconds=5)).isoformat()
    digest = request["instruments"]["BTCUSDT"]["liquidity_positioning_context_hash"]
    assert digest in frozen_hashes_for_symbol(request, "BTCUSDT")
    altered = copy.deepcopy(request)
    altered["liquidity_positioning_context"]["official_state"] = "normal"
    with pytest.raises(ValidationFailed, match="liquidity positioning context hash mismatch"):
        validate_response(altered, response)


def test_rejects_tampered_snapshot_status_and_errors():
    import copy
    snap = _snapshot()
    snap["ok"] = False
    snap["errors"] = [{"symbol": "BTCUSDT", "part": "candles", "error": "gap"}]
    req = build_request(snap)
    response = PacketResponseIn.model_validate(_response(req))
    validate_response(req, response)
    for field, value in (("snapshot_ok", True), ("errors", []), ("exported_at_utc", CUTOFF.isoformat() + "1")):
        changed = copy.deepcopy(req)
        changed[field] = value
        with pytest.raises(ValidationFailed, match="request content hash"):
            validate_response(changed, response)


def test_legacy_packet_can_import_research_but_not_create_plan(tmp_path):
    from piia.intelligence.hashing import content_hash
    from piia.intelligence.research_packet import _canonical
    req = build_request(_snapshot())
    req.pop("integrity_version")
    digest_input = {"cutoff_at_utc": req["cutoff_at_utc"], "missing_datasets": req["missing_datasets"],
                    "instruments": {s: {k: v[k] for k in v if k not in {"all_closed_candles", "all_closed_daily_candles"}}
                                    for s, v in req["instruments"].items()}}
    req["request_hash"] = content_hash(_canonical(digest_input))
    stop = req["instruments"]["BTCUSDT"]["indicators_prefix_only"]["last_close"] * .98
    result = import_response(req, _response(req, direction="long", action="review_paper_plan", stop=stop),
                             db_path=tmp_path / "legacy.sqlite", evidence_store=tmp_path / "evidence",
                             clock=FrozenClock(CUTOFF))
    assert all(item["plan"] is None for item in result["instruments"])
    assert all("legacy_snapshot_integrity_unverified" in item["risk"]["data_quality_failures"]
               for item in result["instruments"])


def test_macro_context_is_bound_to_frozen_evidence():
    import copy
    snap = _snapshot()
    snap["macro_context"] = {"retrieved_at_utc": CUTOFF.isoformat(), "series": {"DGS10": {"value": 4.5}}}
    req = build_request(snap)
    response = PacketResponseIn.model_validate(_response(req))
    validate_response(req, response)
    changed = copy.deepcopy(req)
    changed["macro_context"]["series"]["DGS10"]["value"] = 8.0
    with pytest.raises(ValidationFailed, match="macro context hash"):
        validate_response(changed, response)


def test_open_interest_is_bound_to_instrument_evidence():
    import copy
    snap = _snapshot()
    snap["instruments"]["BTCUSDT"]["open_interest"] = {
        "period_end_utc": (CUTOFF - timedelta(hours=1)).isoformat(),
        "retrieved_at_utc": CUTOFF.isoformat(), "total_open_interest_base_units": 100.0}
    req = build_request(snap)
    response = PacketResponseIn.model_validate(_response(req))
    validate_response(req, response)
    changed = copy.deepcopy(req)
    changed["instruments"]["BTCUSDT"]["open_interest"]["total_open_interest_base_units"] = 500.0
    with pytest.raises(ValidationFailed, match="evidence content hash"):
        validate_response(changed, response)


def test_market_context_is_bound_to_snapshot_integrity():
    import copy
    req = build_request(_snapshot())
    assert req["instruments"]["BTCUSDT"]["market_context"]["timeframe_alignment"] == "incomplete"
    altered = copy.deepcopy(req)
    altered["instruments"]["BTCUSDT"]["market_context"]["timeframe_alignment"] = "aligned_up"
    with pytest.raises(ValidationFailed, match="evidence content hash"):
        validate_response(altered, PacketResponseIn.model_validate(_response(req)))


def test_paper_risk_context_is_bound_and_identified_as_virtual():
    import copy
    from piia.intelligence.research_packet import capture_paper_risk_context
    risk = capture_paper_risk_context()
    assert risk["available"]
    assert "not_actual_account" in risk["kind"]
    assert risk["example_stop_risk_budget_usdt"] == pytest.approx(risk["virtual_equity_usdt"] * risk["risk_fraction"])
    risk["captured_at_utc"] = CUTOFF.isoformat()
    snap = _snapshot()
    snap["paper_risk_context"] = risk
    req = build_request(snap)
    response = PacketResponseIn.model_validate(_response(req))
    validate_response(req, response)
    changed = copy.deepcopy(req)
    changed["paper_risk_context"]["virtual_equity_usdt"] *= 10
    with pytest.raises(ValidationFailed, match="paper risk context hash"):
        validate_response(changed, response)


def test_changed_paper_risk_keeps_research_but_blocks_plan(tmp_path):
    from piia.intelligence.research_packet import capture_paper_risk_context
    risk = capture_paper_risk_context()
    risk["captured_at_utc"] = CUTOFF.isoformat()
    risk["owner_config_hash"] = "previous_owner_config_hash"
    snap = _snapshot()
    snap["paper_risk_context"] = risk
    req = build_request(snap)
    stop = req["instruments"]["BTCUSDT"]["indicators_prefix_only"]["last_close"] * .98
    result = import_response(req, _response(req, direction="long", action="review_paper_plan", stop=stop),
                             db_path=tmp_path / "changed.sqlite", evidence_store=tmp_path / "ev",
                             clock=FrozenClock(CUTOFF))
    assert all(item["plan"] is None for item in result["instruments"])
    assert all("paper_risk_changed_or_unavailable" in item["risk"]["data_quality_failures"]
               for item in result["instruments"])


def test_rejects_claiming_wired_review_and_refuses_live_db(tmp_path):
    req = build_request(_snapshot())
    bad = _response(req)
    bad["independent_llm_review"] = "WIRED"
    with pytest.raises(Exception):
        PacketResponseIn.model_validate(bad)
    with pytest.raises(ValidationFailed, match="live research database"):
        import_response(req, _response(req), db_path=DEFAULT_SQLITE, evidence_store=tmp_path / "ev")


def test_flat_does_not_create_a_plan_long_does_but_is_not_approved(tmp_path):
    req = build_request(_snapshot())
    db = tmp_path / "packet.sqlite"
    ev = tmp_path / "ev"
    flat = import_response(req, _response(req, direction="flat", action="stand_aside"), db_path=db, evidence_store=ev)
    assert flat["approved"] is False
    assert flat["live_orders"] is False
    assert all(item["plan"] is None for item in flat["instruments"])
    assert all(item["direction"] == "flat" for item in flat["instruments"])

    # A fresh db for the long path so theses do not collide in this test
    db2 = tmp_path / "packet_long.sqlite"
    ev2 = tmp_path / "ev2"
    # change request hash by tweaking cutoff-adjacent snapshot? Use same request is ok: different db
    # but create_thesis idempotency is per db. Fine.
    stop = req["instruments"]["BTCUSDT"]["indicators_prefix_only"]["last_close"] * 0.98
    long = import_response(
        req, _response(req, direction="long", action="review_paper_plan", stop=stop),
        db_path=db2, evidence_store=ev2, clock=FrozenClock(CUTOFF),
    )
    assert long["approved"] is False
    for item in long["instruments"]:
        assert item["direction"] == "long"
        assert item["plan"] is not None
        assert item["plan"]["approved"] is False
        assert item["plan"]["side"] == "long"
        assert item["plan"]["liquidation_model_supported"] is False
        assert item["risk"]["approved"] is False


def test_stale_packet_retains_research_but_cannot_create_plan(tmp_path):
    req = build_request(_snapshot())
    stop = req["instruments"]["BTCUSDT"]["indicators_prefix_only"]["last_close"] * 0.98
    result = import_response(
        req, _response(req, direction="long", action="review_paper_plan", stop=stop),
        db_path=tmp_path / "stale.sqlite", evidence_store=tmp_path / "ev",
        clock=FrozenClock(CUTOFF + timedelta(hours=3)),
    )
    assert all(item["plan"] is None for item in result["instruments"])
    assert all("ticker_missing_or_stale" in item["risk"]["data_quality_failures"] for item in result["instruments"])


def test_import_marks_combined_evidence_available_at_packet_cutoff(tmp_path):
    from sqlalchemy import select
    from piia.intelligence.db import make_engine, session_factory
    from piia.intelligence.models import EvidenceItem

    snapshot = _snapshot()
    snapshot["macro_context"] = {"retrieved_at_utc": CUTOFF.isoformat(), "series": {}}
    request = build_request(snapshot)
    db = tmp_path / "time.sqlite"
    import_response(request, _response(request), db_path=db, evidence_store=tmp_path / "evidence", clock=FrozenClock(CUTOFF))
    session = session_factory(make_engine(f"sqlite:///{db.as_posix()}"))()
    try:
        evidence = session.scalars(select(EvidenceItem)).all()
        assert evidence
        assert all(item.available_at == CUTOFF and item.observed_at <= item.available_at for item in evidence)
    finally:
        session.close()


def test_candle_gaps_and_invalid_prices_are_flagged():
    from piia.intelligence.public_data import candle_quality
    candles = _candles()
    assert candle_quality(candles, CUTOFF) == []
    gap = candles[:20] + candles[21:]
    assert "duplicate_or_missing_candles" in candle_quality(gap, CUTOFF)
    malformed = [dict(c) for c in candles]
    malformed[0]["high"] = 0.1
    assert "invalid_ohlc" in candle_quality(malformed, CUTOFF)


@pytest.mark.parametrize("duration", [timedelta(hours=1), timedelta(days=1)])
def test_latest_closed_candle_is_required(duration):
    from piia.intelligence.public_data import candle_quality
    boundary = datetime(2026, 11, 16, tzinfo=timezone.utc)
    rows = []
    for i in range(51):
        start = boundary - duration * (51 - i)
        rows.append({"open_at_utc": start.isoformat(), "close_at_utc": (start + duration).isoformat(),
                     "open": 100., "high": 101., "low": 99., "close": 100.})
    assert candle_quality(rows, boundary, duration) == []
    assert candle_quality(rows, boundary + duration / 2, duration) == []
    assert "last_closed_candle_missing_or_stale" in candle_quality(rows[:-1], boundary + duration / 2, duration)
    assert "last_closed_candle_missing_or_stale" in candle_quality(rows, boundary + duration, duration)


def test_research_indicators_are_withheld_for_bad_hourly_bars():
    from piia.intelligence.research_packet import _indicators
    good = _candles()
    assert _indicators(good)["ok"]
    gap = good[:20] + good[21:]
    result = _indicators(gap)
    assert not result["ok"] and "duplicate_or_missing_candles" in result["errors"]
    assert "ema20" not in result
    shifted = [{**c, "open_at_utc": (_aware_test(c["open_at_utc"]) + timedelta(minutes=1)).isoformat(),
                "close_at_utc": (_aware_test(c["close_at_utc"]) + timedelta(minutes=1)).isoformat()} for c in good]
    assert "hourly_not_utc_aligned" in _indicators(shifted)["errors"]
    invalid = [dict(c) for c in good]
    invalid[0]["high"] = float("nan")
    assert "invalid_ohlc" in _indicators(invalid)["errors"]


def _aware_test(timestamp):
    return datetime.fromisoformat(timestamp)


def test_research_daily_context_checks_data_itself():
    from piia.intelligence.research_packet import _daily_context
    candles = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(60):
        opened = start + timedelta(days=i)
        candles.append({"open_at_utc": opened.isoformat(), "close_at_utc": (opened + timedelta(days=1)).isoformat(),
                        "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.})
    assert _daily_context(candles, [])["ok"]
    result = _daily_context(candles[:20] + candles[21:], [])
    assert not result["ok"] and "duplicate_or_missing_candles" in result["errors"]
    assert "ema20" not in result


def test_daily_context_is_closed_checked_and_integrity_bound():
    import copy
    from piia.intelligence.public_data import _closed_candles, candle_quality
    from piia.intelligence.research_packet import _daily_context
    now = datetime(2026, 11, 16, 9, tzinfo=timezone.utc)
    rows = []
    for i in range(61):
        opening = datetime(2026, 9, 17, tzinfo=timezone.utc) + timedelta(days=i)
        rows.append([int(opening.timestamp() * 1000), 100 + i, 102 + i, 99 + i, 101 + i, 10])
    daily = _closed_candles(rows, now, timedelta(days=1))
    assert len(daily) == 60  # Nov 16 is incomplete
    assert candle_quality(daily, now, timedelta(days=1)) == []
    assert _daily_context(daily, [])["ok"] is True
    assert not _daily_context(daily, ["duplicate_or_missing_candles"])["ok"]
    snap = _snapshot()
    for inst in snap["instruments"].values():
        inst["candles_closed_1d"] = daily
        inst["daily_quality_errors"] = []
    req = build_request(snap)
    response = PacketResponseIn.model_validate(_response(req))
    changed = copy.deepcopy(req)
    changed["instruments"]["BTCUSDT"]["all_closed_daily_candles"][0]["close"] *= 2
    with pytest.raises(ValidationFailed, match="daily candle content hash"):
        validate_response(changed, response)


def test_daily_history_pagination_overlap_and_200_day_availability(monkeypatch):
    from urllib.parse import parse_qs, urlparse
    from piia.intelligence import public_data
    from piia.intelligence.research_packet import _daily_context
    now = datetime(2026, 11, 16, 9, tzinfo=timezone.utc)
    today = now.replace(hour=0)
    rows = []
    for i in range(270):
        opening = today - timedelta(days=269 - i)
        rows.append([int(opening.timestamp() * 1000), 100 + i, 102 + i, 99 + i, 101 + i, 10])
    def fake_fetch(url, timeout):
        params = parse_qs(urlparse(url).query)
        if "startTime" not in params:
            return {"code": "00000", "data": rows[-90:]}
        start, end = int(params["startTime"][0]), int(params["endTime"][0])
        return {"code": "00000", "data": [r for r in rows if start <= r[0] < end]}
    monkeypatch.setattr(public_data, "_get_json", fake_fetch)
    daily, sources = public_data._daily_with_history("BTCUSDT", now, 1)
    assert len(sources) == 3
    assert len(daily) == 250
    assert public_data.candle_quality(daily, now, timedelta(days=1)) == []
    assert _daily_context(daily, [])["sma200"] is not None
    assert _daily_context(daily[-180:], [])["sma200"] is None
    def conflicting_fetch(url, timeout):
        result = fake_fetch(url, timeout)
        if "startTime" in parse_qs(urlparse(url).query) and result["data"]:
            changed = [list(r) for r in result["data"]]
            changed[-1][4] += 1
            return {"code": "00000", "data": changed}
        return result
    monkeypatch.setattr(public_data, "_get_json", conflicting_fetch)
    with pytest.raises(ValueError, match="conflicting daily candles"):
        public_data._daily_with_history("BTCUSDT", now, 1)


def test_claim_may_cite_frozen_subhashes_but_rejects_unknown_or_cross_symbol_hashes():
    snap = _snapshot()
    # Distinct XAU history so its candle hash is not accidentally identical to BTC.
    xau_candles = _candles(80, last=200.0)
    snap["instruments"]["XAUUSDT"]["candles_closed_1h"] = xau_candles
    req = build_request(snap)
    btc = req["instruments"]["BTCUSDT"]
    xau = req["instruments"]["XAUUSDT"]
    assert btc["candles_hash"] != xau["candles_hash"]
    ok = _response(req)
    ok["instruments"][0]["researcher"]["claims"] = [
        {"text": "uses candle hash", "evidence_hash": btc["candles_hash"], "status": "supported"},
        {"text": "uses instrument evidence", "evidence_hash": btc["evidence_hash"], "status": "supported"},
    ]
    validate_response(req, PacketResponseIn.model_validate(ok))

    fake = _response(req)
    fake["instruments"][0]["researcher"]["claims"] = [
        {"text": "forged", "evidence_hash": "a" * 64, "status": "supported"},
    ]
    with pytest.raises(ValidationFailed, match="claim evidence_hash"):
        validate_response(req, PacketResponseIn.model_validate(fake))

    crossed = _response(req)
    crossed["instruments"][0]["researcher"]["claims"] = [
        {"text": "wrong symbol hash", "evidence_hash": xau["candles_hash"], "status": "supported"},
    ]
    with pytest.raises(ValidationFailed, match="claim evidence_hash"):
        validate_response(req, PacketResponseIn.model_validate(crossed))


def test_missing_schema_fields_are_rejected():
    from pydantic import ValidationError
    req = build_request(_snapshot())
    missing_summary = _response(req)
    missing_summary["instruments"][0]["synthesizer"]["summary"] = "   "
    with pytest.raises(ValidationError):
        PacketResponseIn.model_validate(missing_summary)
    missing_role = _response(req)
    del missing_role["instruments"][0]["critic"]
    with pytest.raises(ValidationError):
        PacketResponseIn.model_validate(missing_role)
    missing_instruments = _response(req)
    missing_instruments["instruments"] = []
    with pytest.raises(ValidationFailed, match="missing instruments"):
        validate_response(req, PacketResponseIn.model_validate(missing_instruments))


def test_tampered_request_is_rejected_even_with_matching_claim_subhash():
    import copy
    req = build_request(_snapshot())
    response = PacketResponseIn.model_validate(_response(req))
    changed = copy.deepcopy(req)
    changed["instruments"]["BTCUSDT"]["all_closed_candles"][0]["close"] *= 2
    with pytest.raises(ValidationFailed, match="candle content hash"):
        validate_response(changed, response)
