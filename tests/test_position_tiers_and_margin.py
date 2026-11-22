from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("piia")

from piia.intelligence.paper.isolated_margin import (
    apply_isolated_tier_constraints,
    estimated_isolated_liquidation,
    stop_before_estimated_liq,
)
from piia.intelligence.public_position_tiers import (
    conservative_tier_for_notional,
    max_notional_for_leverage,
    parse_position_tiers,
)

NOW = datetime(2026, 11, 17, 3, 30, tzinfo=timezone.utc)
URL = "https://api.bitget.com/api/v2/mix/market/query-position-lever?productType=USDT-FUTURES&symbol=BTCUSDT"


def _payload(rows, ts=None, symbol="BTCUSDT"):
    ts = ts if ts is not None else int(NOW.timestamp() * 1000)
    data = []
    for i, (start, end, lev, mmr) in enumerate(rows, start=1):
        data.append({"symbol": symbol, "level": str(i), "startUnit": start, "endUnit": end, "leverage": lev, "keepMarginRate": mmr})
    return {"code": "00000", "msg": "success", "requestTime": ts, "data": data}


def _snap(rows=None, **kw):
    rows = rows or [("0", "200000", "150", "0.0040"), ("200000", "1000000", "100", "0.0050")]
    return parse_position_tiers(_payload(rows, symbol=kw.get("symbol", "BTCUSDT")), symbol=kw.get("symbol", "BTCUSDT"), source_url=URL, retrieved=kw.get("now", NOW), raw=b"fixture-tiers")


def test_times_distinct_and_stale_future_rejected():
    retrieved = NOW + timedelta(seconds=1)
    parsed = parse_position_tiers(_payload([("0", "200000", "150", "0.0040")]), symbol="BTCUSDT", source_url=URL, retrieved=retrieved, raw=b"abc")
    assert parsed["ok"] is True
    assert parsed["endpoint_request_time_utc"] != parsed["retrieved_at_utc"]
    stale = parse_position_tiers(_payload([("0", "200000", "150", "0.0040")], ts=int((NOW - timedelta(minutes=6)).timestamp()*1000)), symbol="BTCUSDT", source_url=URL, retrieved=NOW)
    assert stale["reason"] == "stale_tiers"
    future = parse_position_tiers(_payload([("0", "200000", "150", "0.0040")], ts=int((NOW + timedelta(minutes=2)).timestamp()*1000)), symbol="BTCUSDT", source_url=URL, retrieved=NOW)
    assert future["reason"] == "future_tier_timestamp"


def test_cross_symbol_gap_overlap_and_units():
    crossed = parse_position_tiers(_payload([("0", "200000", "150", "0.0040")], symbol="XAUUSDT"), symbol="BTCUSDT", source_url=URL, retrieved=NOW)
    assert crossed["reason"] == "cross_symbol_tiers"
    gap = _snap([("0", "100", "10", "0.05"), ("200", "300", "5", "0.10")])
    assert gap["reason"] == "tier_gap"
    overlap = _snap([("0", "200", "10", "0.05"), ("100", "300", "5", "0.10")])
    assert overlap["reason"] == "tier_interior_overlap"
    ok = _snap()
    assert ok["ok"] is True
    assert ok["unit"] == "usdt_notional_value"
    # Shared boundary uses the worse (lower leverage / higher MMR) tier.
    boundary = conservative_tier_for_notional(ok, Decimal("200000"))
    assert boundary["ok"] is True
    assert boundary["tier"]["max_leverage"] == "100"
    assert boundary["tier"]["keep_margin_rate"] == "0.0050"
    assert conservative_tier_for_notional(ok, Decimal("1200000000")).get("reason") == "notional_above_published_tiers"


def test_estimated_isolated_formula_and_stop_before_liq():
    long = estimated_isolated_liquidation(side="long", qty=Decimal("0.01"), entry=Decimal("100"), leverage=Decimal("2"), mmr=Decimal("0.004"), taker=Decimal("0.0006"), price_place="1")
    assert long["ok"] is True
    assert long["exact_liquidation_verified"] is False
    assert Decimal(long["estimated_liquidation_price"]) < Decimal("100")
    # Algebra with offset 0: entry*(1/L-1)/(MMR+taker-1)
    expected = Decimal("100") * (Decimal("0.5") - 1) / (Decimal("0.0046") - 1)
    assert abs(Decimal(long["unrounded_price"]) - expected) < Decimal("1e-10")
    assert stop_before_estimated_liq("long", Decimal("90"), Decimal(long["estimated_liquidation_price"]))["ok"] is True
    assert stop_before_estimated_liq("long", Decimal("10"), Decimal(long["estimated_liquidation_price"]))["ok"] is False
    short = estimated_isolated_liquidation(side="short", qty=Decimal("0.01"), entry=Decimal("100"), leverage=Decimal("2"), mmr=Decimal("0.004"), taker=Decimal("0.0006"))
    assert Decimal(short["estimated_liquidation_price"]) > Decimal("100")
    assert estimated_isolated_liquidation(side="buy", qty=Decimal("0.01"), entry=Decimal("100"), leverage=Decimal("2"), mmr=Decimal("0.004"), taker=Decimal("0.0006"))["reason"] == "wrong_side"


def test_cross_margin_and_funding_and_size_down():
    snap = _snap()
    crossed = apply_isolated_tier_constraints(
        side="long", qty=Decimal("0.01"), entry=Decimal("100"), stop=Decimal("90"), leverage=Decimal("2"),
        taker=Decimal("0.0006"), equity=Decimal("100"), cash_floor=Decimal("0.2"), max_margin_fraction=Decimal("0.5"),
        contract_size=Decimal("1"), qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"), min_notional=Decimal("5"),
        tiers=snap, funding_rate=None, funding_rate_known=False, margin_mode="crossed",
    )
    assert crossed["reason"] == "cross_or_unsupported_margin_mode"
    # High leverage on a large notional must size downward, never raise leverage.
    tight = _snap([("0", "50", "2", "0.20"), ("50", "200", "1", "0.60")])
    result = apply_isolated_tier_constraints(
        side="long", qty=Decimal("2"), entry=Decimal("100"), stop=Decimal("95"), leverage=Decimal("2"),
        taker=Decimal("0.0006"), equity=Decimal("100"), cash_floor=Decimal("0.2"), max_margin_fraction=Decimal("0.5"),
        contract_size=Decimal("1"), qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"), min_notional=Decimal("5"),
        tiers=tight, funding_rate=Decimal("0.0001"), funding_rate_known=True, margin_mode="isolated",
    )
    assert result["ok"] is True
    assert result["sized_downward"] is True
    assert result["leverage_raised"] is False
    assert Decimal(result["notional"]) <= Decimal("50")
    assert Decimal(result["funding_reserve"]) > 0
    assert Decimal(result["qty"]) < Decimal("2")
    # Stop beyond estimated liq is rejected.
    far = apply_isolated_tier_constraints(
        side="long", qty=Decimal("0.1"), entry=Decimal("100"), stop=Decimal("1"), leverage=Decimal("2"),
        taker=Decimal("0.0006"), equity=Decimal("100"), cash_floor=Decimal("0.2"), max_margin_fraction=Decimal("0.5"),
        contract_size=Decimal("1"), qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"), min_notional=Decimal("5"),
        tiers=snap, funding_rate=Decimal("0"), funding_rate_known=True, margin_mode="isolated",
    )
    assert far["ok"] is False
    assert far["reason"] == "stop_at_or_beyond_estimated_liquidation"
    assert far["exact_liquidation_verified"] is False


def test_missing_and_unsupported_symbol_rejected():
    cap = max_notional_for_leverage({"ok": False, "reason": "symbol_unsupported"}, Decimal("2"))
    assert cap["ok"] is False
    assert apply_isolated_tier_constraints(
        side="long", qty=Decimal("0.01"), entry=Decimal("100"), stop=Decimal("90"), leverage=Decimal("2"),
        taker=Decimal("0.0006"), equity=Decimal("100"), cash_floor=Decimal("0.2"), max_margin_fraction=Decimal("0.5"),
        contract_size=Decimal("1"), qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"), min_notional=Decimal("5"),
        tiers={"ok": False, "reason": "symbol_unsupported"}, funding_rate=None, funding_rate_known=False,
    )["reason"] == "symbol_unsupported"
