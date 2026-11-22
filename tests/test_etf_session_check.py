"""Calendar comparisons use synthetic ETF records, not flow data."""
from piia.intelligence.etf_session_check import annotate_etf_session
from datetime import datetime, timezone


def item(flow_date, at):
    return {"usable_for_btc_research": True, "asof_complete_flow_date": flow_date, "completed_at_utc": at}


def test_prior_flow_does_not_clear_latest_session_block():
    out = annotate_etf_session(item("2026-11-13", "2026-11-18T01:43:00Z"))
    assert out["session_check"]["last_closed_session"] == "2026-11-17"
    assert out["session_check"]["status"] == "not_latest_closed_session"
    assert out["latest_exchange_session_verified"] is False
    matching = annotate_etf_session(item("2026-11-17", "2026-11-18T01:43:00Z"))
    assert matching["latest_exchange_session_verified"] is True
    assert matching["session_check"]["issuer_publication_verified"] is False


def test_thanksgiving_and_early_close():
    before = annotate_etf_session(item("2026-11-25", "2026-11-27T17:59:00Z"))
    assert before["session_check"]["last_closed_session"] == "2026-11-25"
    assert before["latest_exchange_session_verified"] is True
    after = annotate_etf_session(item("2026-11-27", "2026-11-27T18:01:00Z"))
    assert after["session_check"]["last_closed_session"] == "2026-11-27"
    assert after["session_check"]["last_close_utc"] == "2026-11-27T18:00:00+00:00"


def test_invalid_timestamp_fails_closed():
    result = annotate_etf_session(item("2026-11-17", "2026-11-18T01:43:00"))
    assert result["latest_exchange_session_verified"] is False
    assert result["session_check"]["status"] == "calendar_or_date_unavailable"


def test_previously_latest_session_becomes_stale_at_next_close():
    captured = item("2026-11-25", "2026-11-27T17:59:00Z")
    assert annotate_etf_session(captured)["latest_exchange_session_verified"] is True
    later = annotate_etf_session(captured, as_of=datetime(2026, 11, 27, 18, 1, tzinfo=timezone.utc))
    assert later["session_check"]["last_closed_session"] == "2026-11-27"
    assert later["latest_exchange_session_verified"] is False
