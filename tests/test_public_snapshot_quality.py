from datetime import datetime, timedelta, timezone

import pytest

from piia.intelligence.public_data import _parse_current_funding, _parse_ticker


def _oi_payload(end):
    return {"code": "00000", "data": {"list": [
        {"symbol": "BTCUSDT", "timestamp": str(int((end - timedelta(hours=i)).timestamp()*1000)),
         "openInterest": str(100 - i)} for i in range(25)]}}


def test_open_interest_uses_only_contiguous_closed_hours_and_valid_symbols():
    from piia.intelligence.public_data import _parse_open_interest_history
    now = datetime(2026, 11, 16, 8, 30, tzinfo=timezone.utc)
    hour = now.replace(minute=0)
    payload = _oi_payload(hour)
    result = _parse_open_interest_history(payload, "BTCUSDT", now, "source")
    assert result["history_complete_24h"]
    assert result["change_24h_pct"] == pytest.approx((100 / 76 - 1) * 100)
    assert result["period_end_utc"] == hour.isoformat()
    gap = _oi_payload(hour)
    del gap["data"]["list"][10]
    assert _parse_open_interest_history(gap, "BTCUSDT", now, "source")["change_24h_pct"] is None
    future = _oi_payload(hour)
    future["data"]["list"].insert(0, {"symbol": "BTCUSDT", "timestamp": str(int((hour + timedelta(hours=1)).timestamp()*1000)), "openInterest": "999"})
    result = _parse_open_interest_history(future, "BTCUSDT", now, "source")
    assert result["future_rows_excluded"] == 1 and result["total_open_interest_base_units"] == 100
    for change in ({"openInterest": "nan"}, {"symbol": "XAUUSDT"}, {"timestamp": "not-a-time"}):
        bad = _oi_payload(hour)
        bad["data"]["list"][0].update(change)
        with pytest.raises(ValueError):
            _parse_open_interest_history(bad, "BTCUSDT", now, "source")
    with pytest.raises(ValueError, match="stale"):
        _parse_open_interest_history(payload, "BTCUSDT", now + timedelta(hours=3), "source")


def test_ticker_rejects_missing_stale_future_and_nonfinite_values():
    now = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    row = {"lastPr": "100", "markPrice": "101", "indexPrice": "99", "ts": str(int(now.timestamp()*1000))}
    assert _parse_ticker(row, now, "source")["mark"] == 101
    for change in ({"markPrice": "nan"}, {"lastPr": "0"}, {"indexPrice": "inf"},
                   {"ts": None}, {"ts": str(int((now - timedelta(minutes=6)).timestamp()*1000))},
                   {"ts": str(int((now + timedelta(minutes=1)).timestamp()*1000))}):
        with pytest.raises(ValueError):
            _parse_ticker({**row, **change}, now, "source")


def test_zero_funding_is_preserved_and_missing_is_not_zero():
    assert _parse_current_funding({"fundingRate": 0}, "source")["funding_rate"] == "0"
    assert _parse_current_funding({"fundingRate": "-0.0001"}, "source")["funding_rate"] == "-0.0001"
    for value in (None, "nan", float("inf"), True):
        with pytest.raises((ValueError, TypeError)):
            _parse_current_funding({"fundingRate": value}, "source")
    now = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    result = _parse_current_funding({"symbol": "BTCUSDT", "fundingRate": "0.01"}, "source", retrieved_at=now, symbol="BTCUSDT")
    assert result["retrieved_at_utc"] == now.isoformat()
    assert "not a verified settlement" in result["limitations"]
    with pytest.raises(ValueError, match="wrong funding symbol"):
        _parse_current_funding({"symbol": "XAUUSDT", "fundingRate": "0"}, "source", retrieved_at=now, symbol="BTCUSDT")
    with pytest.raises(ValueError, match="timezone"):
        _parse_current_funding({"fundingRate": "0"}, "source", retrieved_at=now.replace(tzinfo=None))


def test_funding_research_requires_fresh_timed_provider_snapshot():
    from piia.intelligence.public_data import current_funding_for_research
    now = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    funding = _parse_current_funding({"fundingRate": "0"}, "source", retrieved_at=now)
    assert current_funding_for_research(funding, now)["funding_rate"] == "0"
    assert current_funding_for_research(funding, now + timedelta(minutes=5, seconds=1)) is None
    assert current_funding_for_research(funding, now - timedelta(seconds=1)) is None
    assert current_funding_for_research({**funding, "funding_rate": "nan"}, now) is None
    assert current_funding_for_research({**funding, "retrieved_at_utc": "2026-11-16T08:00:00"}, now) is None
    assert current_funding_for_research({**funding, "provenance": "fixture"}, now) is None
    assert current_funding_for_research(None, now) is None


def test_invalid_ticker_makes_snapshot_unusable(monkeypatch):
    from piia.intelligence import public_data
    now = datetime.now(timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_rows = [[int((hour - timedelta(hours=i)).timestamp()*1000), 100, 101, 99, 100, 1]
                   for i in range(60, 0, -1)]
    daily_rows = [[int((day - timedelta(days=i)).timestamp()*1000), 100, 101, 99, 100, 1]
                  for i in range(60, 0, -1)]
    def fake_get(url, timeout):
        if "open-interest-history" in url:
            return _oi_payload(hour)
        if "/ticker?" in url:
            return {"data": [{"lastPr": "100", "markPrice": "nan", "ts": str(int(now.timestamp()*1000))}]}
        if "current-fund-rate" in url:
            return {"code": "00000", "data": {"fundingRate": "0"}}
        return {"data": hourly_rows}
    monkeypatch.setattr(public_data, "_get_json", fake_get)
    monkeypatch.setattr(public_data, "_daily_with_history", lambda *args: (
        public_data._closed_candles(daily_rows, now, timedelta(days=1)), ["source"]))
    snapshot = public_data.fetch_public_snapshot(["BTCUSDT"])
    assert snapshot["ok"] is False
    assert snapshot["instruments"]["BTCUSDT"]["ticker"] is None
    assert any(x["part"] == "ticker" for x in snapshot["errors"])
    assert all(x["part"] == "ticker" for x in snapshot["errors"])


def test_slow_snapshot_uses_completion_time_and_rejects_stale_ticker(monkeypatch):
    from piia.intelligence import public_data
    start = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    current = [start]
    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]
    rows = [[int((start - timedelta(hours=i)).timestamp() * 1000), 100, 101, 99, 100, 1]
            for i in range(60, 0, -1)]
    def fake_get(url, timeout):
        if "open-interest-history" in url:
            return _oi_payload(start)
        if "/ticker?" in url:
            return {"data": [{"lastPr": "100", "markPrice": "100", "ts": str(int(start.timestamp() * 1000))}]}
        if "current-fund-rate" in url:
            current[0] = start + timedelta(minutes=6)
            return {"code": "00000", "data": {"fundingRate": "0"}}
        return {"data": rows}
    monkeypatch.setattr(public_data, "datetime", FakeDatetime)
    monkeypatch.setattr(public_data, "_get_json", fake_get)
    monkeypatch.setattr(public_data, "_daily_with_history", lambda *args: ([], ["source"]))
    snapshot = public_data.fetch_public_snapshot(["BTCUSDT"])
    assert snapshot["request_started_at_utc"] == start.isoformat()
    assert snapshot["retrieved_at_utc"] == current[0].isoformat()
    assert not snapshot["ok"]
    assert any(e["error"] == "stale_at_completion" for e in snapshot["errors"])
    from piia.intelligence.research_packet import build_request
    assert datetime.fromisoformat(build_request(snapshot)["cutoff_at_utc"]) == current[0]


def test_funding_error_response_is_not_treated_as_valid(monkeypatch):
    from piia.intelligence import public_data
    def fake_get(url, timeout):
        if "current-fund-rate" in url:
            return {"code": "40000", "data": {"fundingRate": "0.01"}}
        raise ValueError("synthetic failure")
    monkeypatch.setattr(public_data, "_get_json", fake_get)
    monkeypatch.setattr(public_data, "_daily_with_history", lambda *args: ([], ["source"]))
    snapshot = public_data.fetch_public_snapshot(["BTCUSDT"])
    assert snapshot["instruments"]["BTCUSDT"]["funding"] is None
    assert any(e["part"] == "funding" for e in snapshot["errors"])
