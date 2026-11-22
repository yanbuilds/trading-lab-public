from __future__ import annotations
from datetime import datetime, timezone
import pytest
pytest.importorskip("piia")
from piia.intelligence.clock import FrozenClock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.errors import ValidationFailed
from piia.intelligence.paper.engine import REGISTRY, spec_for
from piia.intelligence.paper.specs import CONTRACTS, SAMPLE_MARKS
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.schemas import ApprovalIn, EvidenceIn, ThesisCreate
from piia.intelligence.service import IntelligenceService
from trading_lab.perps.risk import size_order

T0 = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)

@pytest.fixture
def ext(tmp_path):
    url = f"sqlite:///{(tmp_path / 'p.sqlite').as_posix()}"
    engine = make_engine(url)
    create_all(engine)
    s = session_factory(engine)()
    core = IntelligenceService(s, clock=FrozenClock(T0), evidence_store=tmp_path / "ev")
    core.seed_instruments()
    yield ExtendedService(core)
    s.close()

def _ready(ext, symbol="BTCUSDT", side="long"):
    th = ext.core.create_thesis(ThesisCreate(
        instrument_id=CONTRACTS[symbol]["instrument_id"], author="tester", direction=side,
        horizon_days=7, invalidation="stop or 8%", subjective_confidence=0.4,
        statement="LABELED_FIXTURE_EXAMPLE", label="LABELED_FIXTURE_EXAMPLE", is_example=True,
        evidence=[EvidenceIn(raw="fixture", provider="fixture", data_quality="unverified")],
    ))
    acc = ext.paper.open_account(starting_cash=30)
    px = float(SAMPLE_MARKS[symbol]["mark"])
    stop = px * 0.98 if side == "long" else px * 1.02
    return th, acc, px, stop

def test_registry_live_disabled():
    assert REGISTRY["local_simplified"]["supported"] is True
    assert REGISTRY["local_simplified"]["live_orders"] is False
    assert REGISTRY["exchange_live_verified"]["supported"] is False
    assert REGISTRY["exchange_live_verified"]["margin_model_supported"] is False

def test_xau_min_qty_fail_closed():
    spec = spec_for("XAUUSDT")
    px = float(SAMPLE_MARKS["XAUUSDT"]["mark"])
    qty, reason = size_order(30, 30, px, px * 0.02, spec, leverage=2, risk_fraction=0.01, cash_floor=0.2, max_margin_fraction=0.5, slippage=0.0005)
    assert qty == 0
    assert reason == "minimum_order_exceeds_budget"

def test_long_short_fill_pnl_fee_and_gap_stop(ext):
    th, acc, px, stop = _ready(ext, "BTCUSDT", "long")
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    assert plan["actionable"] is True
    assert plan["margin_model_supported"] is False
    assert plan["liquidation_model_supported"] is False
    assert plan["request"]["sandbox"] is True
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    assert dec["valid"] is True
    fill = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert fill["status"] == "filled"
    assert fill["sandbox"] is True
    pos = fill["position"]
    cash_after_open = fill["account"]["cash"]
    assert cash_after_open < 30
    # gap stop: open below stop
    gap = ext.paper.check_stop(pos["id"], {"open": stop - 50, "high": stop + 10, "low": stop - 80, "close": stop - 20})
    assert gap["filled"] is True and gap["reduce_only"] is True
    assert gap["exchange_stop"] is False
    assert gap["net_pnl"] < 0
    assert gap["account"]["cash"] != cash_after_open
    # idempotent execute of original decision does not open a second position
    again = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert again["idempotent"] is True

    th2, acc2, px2, stop2 = _ready(ext, "NDX100USDT", "short")
    plan2 = ext.create_sandbox_plan(thesis_id=th2["id"], account_id=acc2["id"], symbol="NDX100USDT",
                                    side="short", entry_price=px2, stop_loss=stop2, leverage=2, research_run_id=None)
    dec2 = ext.approve_paper(plan2["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    fill2 = ext.paper.execute_approved(account_id=acc2["id"], decision_id=dec2["id"])
    assert fill2["status"] == "filled" and fill2["position"]["side"] == "short"


def test_old_risk_version_cannot_open_paper_after_approval(ext):
    from piia.intelligence.models import Plan
    th, acc, px, stop = _ready(ext)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="tester", user_choice="approve"))
    row = ext.core.session.get(Plan, plan["id"])
    row.risk_result_json = {**row.risk_result_json, "engine": "deterministic_risk_v1"}
    ext.core.session.commit()
    with pytest.raises(ValidationFailed, match="risk engine version changed"):
        ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])


@pytest.mark.parametrize("side", ["long", "short"])
def test_closed_net_includes_both_fees_and_recorded_funding(ext, side):
    th, acc, px, stop = _ready(ext, "BTCUSDT", side)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side=side, entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="tester", user_choice="approve"))
    opened = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert opened["status"] == "filled"
    funding = ext.paper.apply_funding(opened["position"]["id"], 0.0001, px)
    closed = ext.paper.exit_position(opened["position"]["id"], px, "fixture_exit")
    expected = closed["gross_pnl"] - opened["fee"] - closed["fee"] - funding["amount_paid"]
    assert closed["net_pnl"] == pytest.approx(expected)
    assert closed["position"]["realized_pnl"] == pytest.approx(expected)
    assert closed["account"]["realized_pnl"] == pytest.approx(expected)
    assert closed["account"]["cash"] - acc["starting_cash"] == pytest.approx(expected)
    assert closed["net_pnl_before_entry_fee_and_funding"] == pytest.approx(closed["gross_pnl"] - closed["fee"])
    assert "funding completeness unverified" in closed["net_pnl_basis"]


def test_invalid_funding_does_not_change_paper_cash(ext):
    th, acc, px, stop = _ready(ext)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="tester", user_choice="approve"))
    opened = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    position = opened["position"]["id"]
    for rate, mark in ((float("nan"), px), (0.001, float("inf")), (0.001, 0), (True, px)):
        with pytest.raises(ValidationFailed, match="finite"):
            ext.paper.apply_funding(position, rate, mark)
    assert ext.paper.get_account(acc["id"]).cash == pytest.approx(opened["account"]["cash"])


@pytest.mark.parametrize("side", ["long", "short"])
def test_normal_stop_with_assumed_costs_stays_within_paper_budget(ext, side):
    from piia.intelligence.paper.risk_file import load_owner_risk
    th, acc, px, stop = _ready(ext, "BTCUSDT", side)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side=side, entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="tester", user_choice="approve"))
    opened = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert opened["status"] == "filled"
    cfg, _ = load_owner_risk()
    candle = {"open": px, "high": max(px, stop) + 1, "low": min(px, stop) - 1, "close": px}
    closed = ext.paper.check_stop(opened["position"]["id"], candle, slippage=cfg["slippage"])
    assert closed["filled"]
    assert -closed["net_pnl"] <= acc["starting_cash"] * cfg["risk_fraction"] + 1e-9

def test_kill_switch_blocks_new_entries_allows_exits(ext):
    th, acc, px, stop = _ready(ext)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    ext.paper.set_kill("tester", True, "manual")
    blocked = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert blocked["status"] == "rejected"
    assert blocked["reason"] == "kill_switch_enabled"
    ext.paper.set_kill("tester", False, "resume")
    opened = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert opened["status"] == "filled"
    ext.paper.set_kill("tester", True, "halt new")
    # exit still allowed
    out = ext.paper.check_stop(opened["position"]["id"], {"open": stop - 1, "high": px, "low": stop - 20, "close": stop})
    assert out["filled"] is True

def test_stale_live_leverage_and_evidence_change(ext):
    th, acc, px, stop = _ready(ext)
    with pytest.raises(ValidationFailed, match="stale"):
        ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                side="long", entry_price=px, stop_loss=stop, leverage=2,
                                research_run_id=None, stale=True)
    with pytest.raises(ValidationFailed, match="live-mode"):
        ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                side="long", entry_price=px, stop_loss=stop, leverage=2,
                                research_run_id=None, live_mode=True)
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=px, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    bad = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert bad["status"] == "rejected"
    assert bad["reason"] == "invalid_stop"
    # leverage above owner cap rejected at execute if request were tampered; plan uses cap check
    plan2 = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                    side="long", entry_price=px, stop_loss=stop, leverage=5, research_run_id=None)
    dec2 = ext.approve_paper(plan2["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    hi = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec2["id"])
    assert hi["status"] == "rejected"
    assert hi["reason"] == "leverage_exceeds_owner_limit"

def test_negative_cash_rejected(ext):
    th, acc, px, stop = _ready(ext)
    account = ext.paper.get_account(acc["id"])
    account.cash = -1
    ext.session.commit()
    plan = ext.create_sandbox_plan(thesis_id=th["id"], account_id=acc["id"], symbol="BTCUSDT",
                                   side="long", entry_price=px, stop_loss=stop, leverage=2, research_run_id=None)
    dec = ext.approve_paper(plan["id"], ApprovalIn(actor="yan-local", user_choice="approve"))
    out = ext.paper.execute_approved(account_id=acc["id"], decision_id=dec["id"])
    assert out["status"] == "rejected"
    assert out["reason"] == "negative_cash"

def test_public_marks_smoke_if_reachable():
    import os
    if os.environ.get("PIIA_SKIP_NETWORK") == "1":
        pytest.skip("live public ticker smoke test disabled for offline test runs")
    from piia.intelligence.public_data import fetch_public_marks
    data = fetch_public_marks()
    if not data["ok"]:
        pytest.skip("public Bitget ticker unreachable: " + str(data.get("errors")))
    for sym in ("BTCUSDT", "XAUUSDT", "NDX100USDT"):
        assert data["marks"][sym]["mark"] > 0
        assert data["marks"][sym]["last"] > 0
        assert data["marks"][sym]["provenance"] == "public_bitget_ticker"
    assert data["exact_liquidation_verified"] is False
