"""Synthetic portfolio inputs must not bypass deterministic risk checks."""
from datetime import datetime, timedelta, timezone

import pytest

from piia.intelligence.risk import evaluate


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def assess(*, mark=100., mark_as_of=NOW, cash=100., leverage=1., config=None, now=NOW):
    return evaluate(portfolio={"cash": cash, "positions": [{"instrument_id": "btc-usd-spot",
                    "quantity": 0.01, "mark": mark, "mark_as_of": mark_as_of, "instrument_type": "spot"}]},
                    config=config or {}, plan_request={"leverage": leverage}, now=now,
                    latest_revision_id="r1", bound_revision_id="r1")


@pytest.mark.parametrize("mark_as_of", [NOW + timedelta(seconds=1), NOW.replace(tzinfo=None), "bad timestamp"])
def test_future_naive_or_malformed_marks_fail_closed(mark_as_of):
    result = assess(mark_as_of=mark_as_of)
    assert result["passed"] is False
    assert any("stale or missing mark" in reason for reason in result["block_reasons"])


@pytest.mark.parametrize("kwargs", [{"mark": float("nan")}, {"mark": float("inf")},
                                  {"cash": float("nan")}, {"leverage": float("nan")},
                                  {"config": {"cash_floor": float("nan")}},
                                  {"now": NOW.replace(tzinfo=None)}])
def test_nonfinite_and_naive_inputs_fail_closed(kwargs):
    result = assess(**kwargs)
    assert result["passed"] is False
    assert result["actionable"] is False


def test_fresh_finite_baseline_and_perp_guard():
    assert assess()["passed"] is True
    assert assess(leverage=2.)["passed"] is False


@pytest.mark.parametrize("kind,quantity", [("perpetual", 1.), ("unknown", 1.), ("spot", -1.)])
def test_unsupported_holdings_cannot_use_spot_equity(kind, quantity):
    result = evaluate(portfolio={"cash": 200., "positions": [{"instrument_id": "fixture", "quantity": quantity,
                      "mark": 100., "mark_as_of": NOW, "instrument_type": kind}]},
                      config={}, plan_request={}, now=NOW, latest_revision_id="r", bound_revision_id="r")
    assert result["passed"] is False and result["equity"] is None
