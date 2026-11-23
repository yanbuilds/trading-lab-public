from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from piia.intelligence.clock import Clock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.errors import Conflict, Immutable, NotFound, PiiaError, ValidationFailed
from piia.intelligence.paths import default_database_url
from piia.intelligence.schemas import (
    AnalysisIn, ApprovalIn, CounterIn, OutcomeIn, PlanIn, PortfolioIn,
    RiskConfigIn, ThesisCreate, ThesisRevise,
)
from piia.intelligence.service import IntelligenceService
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.settings import model_status
from piia.intelligence.paper.engine import REGISTRY
from piia.intelligence.schemas import ApprovalIn
from pydantic import BaseModel


def _engine():
    url = os.environ.get("PIIA_DATABASE_URL") or default_database_url()
    engine = make_engine(url)
    # Prefer Alembic in CLI; create_all keeps a fresh dev db usable.
    create_all(engine)
    return engine

ENGINE = None
FACTORY = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, FACTORY
    ENGINE = _engine()
    FACTORY = session_factory(ENGINE)
    with FACTORY() as s:
        IntelligenceService(s).seed_instruments()
    yield

app = FastAPI(
    title="PIIA Intelligence API",
    version="0.1.0",
    description=(
        "Phase-1 research audit service. Localhost only. No authentication is implemented; "
        "do not expose this process. There are no trade or executor routes. "
        "Approval endpoints require an explicit API call and are not invoked by a model. "
        "Independent LLM review is NOT wired. PostgreSQL is schema-compatible but not tested. "
        "User approval is never assumed."
    ),
    lifespan=lifespan,
)

def svc() -> IntelligenceService:
    session = FACTORY()
    return IntelligenceService(session, Clock())

def ext() -> ExtendedService:
    return ExtendedService(svc())

@app.exception_handler(PiiaError)
async def _piia_err(_, exc: PiiaError):
    status = 404 if isinstance(exc, NotFound) else 409 if isinstance(exc, (Conflict, Immutable)) else 400
    return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=status)

@app.get("/health")
def health():
    ms = model_status()
    return {
        "ok": True,
        "module": "piia.intelligence",
        "phase": "1b-integrated-v1",
        "trading_routes": False,
        "live_orders": False,
        "exchange_demo_orders": False,
        "paper_engine": "local_simplified",
        "execution_registry": REGISTRY,
        "auth": "absent_localhost_only",
        "llm_review": ms["independent_llm_review"],
        "model_configured": ms["configured"],
        "postgres": "schema_compatible_not_tested",
        "bind_expectation": "127.0.0.1",
    }

@app.get("/instruments")
def instruments():
    s = svc()
    try:
        return [
            {
                "id": i.id, "symbol": i.symbol, "instrument_type": i.instrument_type,
                "venue": i.venue, "contract_symbol": i.contract_symbol,
                "settlement_currency": i.settlement_currency, "contract_size": i.contract_size,
                "market_type": i.market_type, "quote_currency": i.quote_currency,
                "display_name": i.display_name, "notes": i.notes,
            }
            for i in s.list_instruments()
        ]
    finally:
        s.session.close()

@app.post("/theses")
def create_thesis(payload: ThesisCreate, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.create_thesis(payload, idempotency_key)
    finally:
        s.session.close()

@app.get("/theses/{thesis_id}")
def get_thesis(thesis_id: str):
    s = svc()
    try:
        return s.get_thesis(thesis_id)
    finally:
        s.session.close()

@app.post("/theses/{thesis_id}/revisions")
def revise(thesis_id: str, payload: ThesisRevise, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.revise_thesis(thesis_id, payload, idempotency_key)
    finally:
        s.session.close()

@app.put("/theses/{thesis_id}")
@app.patch("/theses/{thesis_id}")
@app.delete("/theses/{thesis_id}")
def deny_thesis(thesis_id: str):
    raise HTTPException(status_code=405, detail="original thesis identity has no update/delete API; append a revision")

@app.put("/evidence/{evidence_id}")
@app.patch("/evidence/{evidence_id}")
@app.delete("/evidence/{evidence_id}")
def deny_evidence(evidence_id: str):
    raise HTTPException(status_code=405, detail="evidence items are immutable")

@app.get("/evidence/{evidence_id}")
def get_evidence(evidence_id: str):
    s = svc()
    try:
        ev = s.get_evidence(evidence_id)
        meta = s._evidence_dict(ev)
        meta["raw"] = s.read_evidence_text(evidence_id)
        return meta
    finally:
        s.session.close()

@app.get("/evidence/{evidence_id}/verify")
def verify_evidence(evidence_id: str):
    s = svc()
    try:
        return s.verify_evidence(evidence_id)
    finally:
        s.session.close()

@app.post("/revisions/{revision_id}/counterarguments")
def add_counter(revision_id: str, payload: CounterIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.add_counterargument(revision_id, payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/revisions/{revision_id}/demo-counterargument")
def demo_counter(revision_id: str):
    s = svc()
    try:
        return s.deterministic_demo_counter(revision_id)
    finally:
        s.session.close()

@app.post("/revisions/{revision_id}/analyses")
def add_analysis(revision_id: str, payload: AnalysisIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.add_analysis(revision_id, payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/portfolio-snapshots")
def create_portfolio(payload: PortfolioIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.create_portfolio(payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/risk-configs")
def create_risk(payload: RiskConfigIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.create_risk_config(payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/plans")
def create_plan(payload: PlanIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.create_plan(payload, idempotency_key=idempotency_key)
    finally:
        s.session.close()

@app.post("/plans/{plan_id}/approvals")
def approve(plan_id: str, payload: ApprovalIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    """Explicit user API action. A model must not call this automatically. No login is implemented."""
    s = svc()
    try:
        return s.approve_plan(plan_id, payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/outcomes")
def evaluate(payload: OutcomeIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = svc()
    try:
        return s.evaluate_outcome(payload, idempotency_key)
    finally:
        s.session.close()

@app.get("/theses/{thesis_id}/audit")
def audit(thesis_id: str):
    s = svc()
    try:
        return s.audit_trace(thesis_id)
    finally:
        s.session.close()

@app.api_route("/trades", methods=["GET", "POST"])
@app.api_route("/execute", methods=["GET", "POST"])
@app.api_route("/orders", methods=["GET", "POST"])
@app.api_route("/live", methods=["GET", "POST"])
def no_trading():
    raise HTTPException(status_code=404, detail="live trade and exchange order routes are disabled")

class ResearchIn(BaseModel):
    thesis_id: str
    mode: str = "fixture"
    actor: str = "user"

class SandboxPlanIn(BaseModel):
    thesis_id: str
    account_id: str
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    leverage: float = 2
    research_run_id: str | None = None
    stale: bool = False
    live_mode: bool = False

class PaperExecuteIn(BaseModel):
    account_id: str
    decision_id: str

class StopCheckIn(BaseModel):
    open: float
    high: float
    low: float
    close: float

class KillIn(BaseModel):
    actor: str
    enabled: bool
    reason: str = ""

@app.get("/status")
def status():
    from piia.intelligence.public_data import fetch_public_marks
    from piia.intelligence.paper.specs import SAMPLE_MARKS, PROVENANCE
    ms = model_status()
    return {
        "model": ms,
        "live_orders": False,
        "paper_engine": "local_simplified",
        "margin_model_supported": False,
        "liquidation_exact": False,
        "sample_marks_not_live_feed": SAMPLE_MARKS,
        "specs_provenance": PROVENANCE,
        "postgres": "schema_compatible_not_tested",
    }

@app.get("/market/public-marks")
def public_marks():
    from piia.intelligence.public_data import fetch_public_marks
    return fetch_public_marks()

@app.post("/research/runs")
def research_run(payload: ResearchIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = ext()
    try:
        return s.run_research(payload.thesis_id, mode=payload.mode, actor=payload.actor, idempotency_key=idempotency_key)
    finally:
        s.session.close()

@app.post("/paper/accounts")
def paper_account():
    s = ext()
    try:
        return s.paper.open_account()
    finally:
        s.session.close()

@app.get("/paper/accounts/{account_id}")
def get_paper_account(account_id: str):
    s = ext()
    try:
        return s.paper.account_dict(s.paper.get_account(account_id))
    finally:
        s.session.close()

@app.post("/paper/plans")
def paper_plan(payload: SandboxPlanIn):
    s = ext()
    try:
        return s.create_sandbox_plan(
            thesis_id=payload.thesis_id, account_id=payload.account_id, symbol=payload.symbol,
            side=payload.side, entry_price=payload.entry_price, stop_loss=payload.stop_loss,
            leverage=payload.leverage, research_run_id=payload.research_run_id,
            stale=payload.stale, live_mode=payload.live_mode,
        )
    finally:
        s.session.close()

@app.post("/paper/plans/{plan_id}/approvals")
def paper_approve(plan_id: str, payload: ApprovalIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = ext()
    try:
        return s.approve_paper(plan_id, payload, idempotency_key)
    finally:
        s.session.close()

@app.post("/paper/execute")
def paper_execute(payload: PaperExecuteIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    s = ext()
    try:
        return s.paper.execute_approved(account_id=payload.account_id, decision_id=payload.decision_id, idempotency_key=idempotency_key)
    finally:
        s.session.close()

@app.post("/paper/positions/{position_id}/stop-check")
def paper_stop(position_id: str, payload: StopCheckIn):
    s = ext()
    try:
        return s.paper.check_stop(position_id, payload.model_dump())
    finally:
        s.session.close()

@app.post("/paper/kill-switch")
def paper_kill(payload: KillIn):
    s = ext()
    try:
        return s.paper.set_kill(payload.actor, payload.enabled, payload.reason)
    finally:
        s.session.close()


class AssistantPlanIn(BaseModel):
    symbol: str
    side: str
    equity_usdt: float = 500.0
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    target_rr: float | None = 2.5
    leverage: float = 2.0
    risk_pct: float = 1.0
    order_type: str = "limit"
    fetch_live_context: bool = False


class AssistantPositionIn(BaseModel):
    symbol: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    leverage: float = 2.0
    mark_price: float | None = None
    fetch_live: bool = False


@app.get("/assistant/scan")
def assistant_scan(
    demo: bool = False,
    equity_usdt: float = 500.0,
    leverage: float = 2.0,
    risk_pct: float = 1.0,
    order_type: str = "limit",
):
    from piia.intelligence.trade_assistant import scan_market_signals
    try:
        return scan_market_signals(
            equity_usdt=equity_usdt,
            leverage=leverage,
            risk_fraction=risk_pct / 100.0,
            order_type=order_type,
            demo_mode=demo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/assistant/plan-trade")
def assistant_plan_trade(payload: AssistantPlanIn):
    from piia.intelligence.trade_assistant import build_trade_plan
    try:
        return build_trade_plan(
            symbol=payload.symbol,
            side=payload.side,
            equity_usdt=payload.equity_usdt,
            entry_price=payload.entry_price,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            target_rr=payload.target_rr,
            leverage=payload.leverage,
            risk_pct=payload.risk_pct,
            order_type=payload.order_type,
            fetch_live_context=payload.fetch_live_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/assistant/position-advice")
def assistant_position_advice(payload: AssistantPositionIn):
    from piia.intelligence.trade_assistant import advise_open_position
    try:
        return advise_open_position(
            symbol=payload.symbol,
            side=payload.side,
            entry_price=payload.entry_price,
            qty=payload.qty,
            stop_loss=payload.stop_loss,
            leverage=payload.leverage,
            mark_price=payload.mark_price,
            fetch_live=payload.fetch_live,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class AgentCycleIn(BaseModel):
    demo_mode: bool = False
    starting_cash: float = 500.0


class AgentProposalDecisionIn(BaseModel):
    choice: str
    actor: str = "local-investor"
    note: str = ""


def _resolve_v2_ledger_path() -> Path:
    import os
    from piia.intelligence.agent_v2.continuous_paper import DEFAULT_LEDGER_PATH
    override = os.environ.get("PIIA_V2_LEDGER_PATH")
    return Path(override) if override else DEFAULT_LEDGER_PATH


@app.get("/agent-v2/verify-freeze")
def agent_v2_verify_freeze():
    from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
    return verify_piia_v2_frozen()


@app.get("/agent-v2/status")
def agent_v2_status():
    from piia.intelligence.agent_v2.continuous_paper import get_continuous_paper_status
    return get_continuous_paper_status(_resolve_v2_ledger_path())


@app.post("/agent-v2/run-cycle")
def agent_v2_run_cycle(payload: AgentCycleIn):
    from piia.intelligence.agent_v2.continuous_paper import run_prospective_cycle
    try:
        return run_prospective_cycle(
            db_path=_resolve_v2_ledger_path(),
            starting_cash=payload.starting_cash,
            demo_mode=payload.demo_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent-v2/proposals/{proposal_id}/decision")
def agent_v2_decide_proposal(proposal_id: str, payload: AgentProposalDecisionIn):
    from piia.intelligence.agent_v2.continuous_paper import approve_human_proposal
    try:
        return approve_human_proposal(
            proposal_id,
            choice=payload.choice,
            actor=payload.actor,
            note=payload.note,
            db_path=_resolve_v2_ledger_path(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
