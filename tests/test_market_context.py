from piia.intelligence.market_context import describe_market
from datetime import datetime, timedelta, timezone


def test_alignment_is_context_and_does_not_hide_timeframe_disagreement():
    hour = {"ok": True, "ema20": 102., "ema50": 100., "rsi14_wilder": 55., "atr14_pct_of_close": 2.,
            "range_24h_low": 90., "range_24h_high": 110., "last_closed_bar_close_at_utc": "2026-01-01T08:00:00Z",
            "four_hour_trend": {"ok": True, "ema20": 98., "ema50": 100., "last_closed_at_utc": "2026-01-01T08:00:00Z"}}
    daily = {"ok": True, "ema20": 101., "ema50": 100., "last_closed_at_utc": "2026-01-01T00:00:00Z"}
    result = describe_market(hour, daily)
    assert result["ema_relationships"] == {"1h": "up", "4h": "down", "1d": "up"}
    assert result["timeframe_alignment"] == "mixed_or_equal"
    assert result["observed_24h_high"] == 110.
    assert result["order_permissions"] is False
    assert "not_signal" in result["kind"]


def test_missing_and_invalid_indicators_remain_unknown():
    hour = {"ok": True, "ema20": 101., "ema50": 100., "rsi14_wilder": float("nan"),
            "atr14_pct_of_close": -1, "range_24h_low": 110., "range_24h_high": 90.,
            "four_hour_trend": {"ok": False}}
    result = describe_market(hour, None)
    assert result["timeframe_alignment"] == "incomplete"
    assert result["rsi14"] is None and result["atr14_pct"] is None
    assert result["observed_24h_low"] is None and result["observed_24h_high"] is None


def test_runtime_daily_recheck_removes_stale_alignment():
    from piia.intelligence.market_context import refresh_research_context
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily = [{"open_at_utc": (start + timedelta(days=i)).isoformat(),
              "close_at_utc": (start + timedelta(days=i+1)).isoformat(),
              "open": 100. + i, "high": 102. + i, "low": 99. + i,
              "close": 101. + i, "volume": 1.} for i in range(60)]
    item = {"daily_context": {"ok": True}, "all_closed_daily_candles": daily,
            "indicators_prefix_only": {"ok": True, "ema20": 2., "ema50": 1.,
                                       "four_hour_trend": {"ok": True, "ema20": 2., "ema50": 1.}}}
    refreshed, context = refresh_research_context(item, start + timedelta(days=60))
    assert refreshed["ok"] and context["timeframe_alignment"] == "aligned_up"
    refreshed, context = refresh_research_context(item, start + timedelta(days=63))
    assert not refreshed["ok"] and context["timeframe_alignment"] == "incomplete"
    assert context["ema_relationships"]["1d"] == "unavailable"
    item["all_closed_daily_candles"] = []
    assert not refresh_research_context(item, start + timedelta(days=60))[0]["ok"]


def test_hour_against_daily_range_does_not_combine_future_or_old_days():
    hour = {"ok": True, "last_close": 111, "last_closed_bar_close_at_utc": "2026-03-02T01:00:00+00:00"}
    daily = {"ok": True, "last_20_closed_daily_high": 110, "last_20_closed_daily_low": 90,
             "last_20_closed_daily_start_utc": "2026-02-10T00:00:00+00:00", "last_closed_at_utc": "2026-03-02T00:00:00+00:00"}
    result = describe_market(hour, daily)["closed_hour_vs_20_completed_days"]
    assert result["status"] == "above" and not result["hour_included_in_range"]
    assert describe_market({**hour, "last_close": 89}, daily)["closed_hour_vs_20_completed_days"]["status"] == "below"
    assert describe_market({**hour, "last_close": 110}, daily)["closed_hour_vs_20_completed_days"]["status"] == "within_or_at_boundary"
    midnight = {**hour, "last_closed_bar_close_at_utc": daily["last_closed_at_utc"]}
    assert describe_market(midnight, daily)["closed_hour_vs_20_completed_days"]["hour_included_in_range"]
    for ts in ("2026-03-01T23:00:00+00:00", "2026-03-03T00:00:00+00:00", "2026-03-02T01:00:00"):
        assert describe_market({**hour, "last_closed_bar_close_at_utc": ts}, daily)["closed_hour_vs_20_completed_days"]["status"] == "unavailable"


def test_daily_range_uses_only_last_twenty_complete_days():
    from piia.intelligence.research_packet import _daily_context
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [{"open_at_utc": (start + timedelta(days=i)).isoformat(),
             "close_at_utc": (start + timedelta(days=i + 1)).isoformat(),
             "open": 100., "high": 101. + i, "low": 99., "close": 100., "volume": 1.}
            for i in range(60)]
    bars[0]["high"] = 1000.  # older than range; fixture remains valid OHLC
    daily = _daily_context(bars, [], start + timedelta(days=60))
    assert daily["last_20_closed_daily_high"] == 160.
    assert daily["last_20_closed_daily_low"] == 99.
    assert daily["last_20_closed_daily_start_utc"] == bars[40]["open_at_utc"]
