from datetime import datetime, timedelta, timezone

import pytest

from piia.intelligence.paper.preflight import virtual_sizing_example
from piia.intelligence.research_packet import capture_paper_risk_context


@pytest.fixture
def packet_fixture():
    now = datetime(2026, 11, 16, 8, tzinfo=timezone.utc)
    risk = capture_paper_risk_context()
    return {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "paper_risk_context": risk,
            "instruments": {"BTCUSDT": {"ticker": {"mark": 100.0, "observed_at_utc": now.isoformat()}},
                            "XAUUSDT": {"ticker": {"mark": 4000.0, "observed_at_utc": now.isoformat()}}}}, now


@pytest.mark.parametrize("side,stop", [("long", 95.0), ("short", 105.0)])
def test_virtual_sizing_respects_stop_budget_after_assumed_costs(packet_fixture, side, stop):
    packet, now = packet_fixture
    result = virtual_sizing_example(packet, "BTCUSDT", side, stop,
                                    current_risk_hash=packet["paper_risk_context"]["owner_config_hash"], now=now)
    assert result["reason"] == "ok" and result["estimated_qty_contracts"] > 0
    assert result["estimated_stop_loss_with_assumed_fees_slippage_usdt"] <= result["risk_budget_usdt"] + 1e-12
    assert result["order_permissions"] is False
    assert result["spec_snapshot_freshness"] == "stale"
    assert result["current_exchange_specs_verified"] is False


def test_minimum_size_is_not_rounded_up(packet_fixture):
    packet, now = packet_fixture
    result = virtual_sizing_example(packet, "XAUUSDT", "long", 3850.0,
                                    current_risk_hash=packet["paper_risk_context"]["owner_config_hash"], now=now)
    assert result["estimated_qty_contracts"] == 0
    assert result["reason"] == "minimum_order_exceeds_budget"


def test_stale_snapshot_changed_config_and_wrong_stop_are_rejected(packet_fixture):
    packet, now = packet_fixture
    risk_hash = packet["paper_risk_context"]["owner_config_hash"]
    with pytest.raises(ValueError, match="过期"):
        virtual_sizing_example(packet, "BTCUSDT", "long", 95, current_risk_hash=risk_hash, now=now + timedelta(minutes=6))
    with pytest.raises(ValueError, match="已变化"):
        virtual_sizing_example(packet, "BTCUSDT", "long", 95, current_risk_hash="changed", now=now)
    with pytest.raises(ValueError, match="低于"):
        virtual_sizing_example(packet, "BTCUSDT", "long", 105, current_risk_hash=risk_hash, now=now)


@pytest.mark.parametrize("side,stop,gap", [("long", 95., 80.), ("short", 105., 120.)])
def test_gap_uses_same_quantity_and_exposes_budget_exceedance(packet_fixture, side, stop, gap):
    packet, now = packet_fixture
    risk = packet["paper_risk_context"]
    result = virtual_sizing_example(packet, "BTCUSDT", side, stop, gap_exit=gap,
                                    current_risk_hash=risk["owner_config_hash"], now=now)
    scenario = result["gap_scenario"]
    sign = 1 if side == "long" else -1
    entry = 100 * (1 + sign * risk["slippage"])
    exit_price = gap * (1 - sign * risk["slippage"])
    spec = risk["specs"]["BTCUSDT"]
    expected = result["estimated_qty_contracts"] * spec["contract_size"] * (
        sign * (entry - exit_price) + spec["taker"] * (entry + exit_price))
    assert scenario["estimated_loss_including_entry_exit_fees_usdt"] == pytest.approx(expected)
    assert scenario["exceeds_stop_risk_budget"] is True
    assert scenario["liquidation_modelled"] is False and scenario["is_maximum_loss"] is False
    with pytest.raises(ValueError, match="更不利"):
        virtual_sizing_example(packet, "BTCUSDT", side, stop, gap_exit=stop,
                               current_risk_hash=risk["owner_config_hash"], now=now)


def test_gap_is_unknown_if_minimum_order_cannot_be_met(packet_fixture):
    packet, now = packet_fixture
    result = virtual_sizing_example(packet, "XAUUSDT", "long", 3850., gap_exit=3500.,
                                    current_risk_hash=packet["paper_risk_context"]["owner_config_hash"], now=now)
    assert result["estimated_qty_contracts"] == 0
    assert result["gap_scenario"]["estimated_loss_including_entry_exit_fees_usdt"] is None


def test_future_or_missing_spec_time_is_unknown(packet_fixture):
    packet, now = packet_fixture
    risk = packet["paper_risk_context"]
    risk["spec_snapshot_at_utc"] = (now + timedelta(minutes=1)).isoformat()
    result = virtual_sizing_example(packet, "BTCUSDT", "long", 95., current_risk_hash=risk["owner_config_hash"], now=now)
    assert result["spec_snapshot_freshness"] == "unknown" and result["spec_snapshot_age_seconds"] is None
    risk["spec_snapshot_at_utc"] = now.isoformat()
    result = virtual_sizing_example(packet, "BTCUSDT", "long", 95., current_risk_hash=risk["owner_config_hash"], now=now)
    assert result["spec_snapshot_freshness"] == "recent_but_not_revalidated"
    assert result["order_eligibility_verified"] is False
