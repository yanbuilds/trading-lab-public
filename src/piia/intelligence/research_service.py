from __future__ import annotations
from copy import deepcopy
from sqlalchemy import select
from sqlalchemy.orm import Session

from piia.intelligence.agents.workflow import run_workflow
from piia.intelligence.clock import Clock
from piia.intelligence.errors import NotFound, ValidationFailed
from piia.intelligence.hashing import content_hash
from piia.intelligence.ids import new_id
from piia.intelligence.models import Analysis, Counterargument, EvidenceLink, Plan, ResearchRun, ThesisRevision
from piia.intelligence.paper.engine import PaperEngine, REGISTRY
from piia.intelligence.paper.risk_file import load_owner_risk
from piia.intelligence.paper.specs import CONTRACTS
from piia.intelligence.schemas import AnalysisIn, ApprovalIn, CounterIn, PlanIn, RiskConfigIn
from piia.intelligence.service import IntelligenceService


def _interpret_claims(result: dict) -> dict:
    """A linked evidence ID is not an independently verified entailment."""
    result = deepcopy(result)
    for role in ("researcher", "critic", "synthesizer"):
        for claim in (result.get(role) or {}).get("claims", []):
            if claim.get("status") == "supported":
                claim["status"] = "unverified"
    result["evidence_verification_note"] = "Evidence references are not independent verification of claim truth."
    result["forecast_calibration_status"] = "not_calibrated"
    return result

def _evidence_bundle(svc: IntelligenceService, revision_id: str) -> tuple[list[dict], list[str]]:
    links = list(svc.session.scalars(select(EvidenceLink).where(EvidenceLink.revision_id == revision_id)))
    items, ids = [], []
    for link in links:
        ev = svc.get_evidence(link.evidence_id)
        raw = svc.read_evidence_text(ev.id)
        ids.append(ev.id)
        items.append({"id": ev.id, "content_hash": ev.content_hash, "raw": raw, "relation": link.relation})
    return items, ids

class ExtendedService:
    def __init__(self, core: IntelligenceService):
        self.core = core
        self.session: Session = core.session
        self.clock: Clock = core.clock
        self.paper = PaperEngine(core.session, core.clock)

    def run_research(self, thesis_id: str, mode: str = "fixture", actor: str = "user", idempotency_key: str | None = None) -> dict:
        thesis = self.core.get_thesis(thesis_id)
        rev_id = thesis["current_revision_id"]
        rev = self.session.get(ThesisRevision, rev_id)
        if not rev:
            raise NotFound("revision not found")
        evidence, ev_ids = _evidence_bundle(self.core, rev_id)
        payload = {
            "thesis_id": thesis_id, "revision_id": rev_id, "statement": rev.statement,
            "invalidation": rev.invalidation, "thesis_direction": rev.direction,
            "instrument_id": thesis["instrument_id"],
            "evidence": [{"id": e["id"], "raw": e["raw"], "content_hash": e["content_hash"]} for e in evidence],
        }
        request = {"thesis_id": thesis_id, "mode": mode, "revision_id": rev_id}
        cached = self.core._idempotent(idempotency_key, "run_research", request)
        if cached is not None:
            cached = dict(cached)
            cached["result"] = _interpret_claims(cached.get("result") or {})
            if cached.get("mode") == "model":
                cached.setdefault("recorded_independent_llm_review", cached.get("independent_llm_review"))
                cached["independent_llm_review"] = "NOT_WIRED"
                cached["result"] = {**(cached.get("result") or {}), "independent_llm_review": "NOT_WIRED"}
            return cached
        result = run_workflow(mode=mode, payload=payload, legal_evidence_ids=ev_ids)
        # Fixture and unconfigured responses are deterministic. A configured
        # model call may be billable and nondeterministic, so each invocation
        # must retain its own output and usage; explicit idempotency is checked
        # above before any call.
        existing = None
        if result.mode in {"fixture", "needs_model_configuration"}:
            existing = self.session.scalars(
                select(ResearchRun).where(ResearchRun.input_hash == result.input_hash, ResearchRun.revision_id == rev_id, ResearchRun.status == result.status)
            ).first()
        if existing:
            out = self._run_dict(existing)
            return self.core._remember(idempotency_key, "run_research", request, out)
        row = ResearchRun(
            id=new_id(), created_at=self.clock.now(), thesis_id=thesis_id, revision_id=rev_id,
            mode=result.mode, status=result.status, independent_llm_review=result.independent_llm_review,
            prompt_hash=result.prompt_hash, input_hash=result.input_hash,
            model_name=result.model_name, model_version=result.model_version, token_cost=result.token_cost,
            payload_json=payload, result_json=result.model_dump(), actor=actor,
        )
        self.session.add(row)
        self.session.flush()
        if result.status == "completed":
            if result.mode == "model":
                # Internal verified workflow path only; public manual-analysis
                # routes still reject user-supplied generation="llm".
                for role_out in (result.researcher, result.critic, result.synthesizer):
                    if not role_out:
                        continue
                    if role_out.generation != "model":
                        raise ValidationFailed("model run contains non-model role output")
                    evs = list(dict.fromkeys(i for c in role_out.claims for i in c.evidence_ids if i in ev_ids))
                    self.core._require_evidence_ids(evs)
                    analysis = Analysis(id=new_id(), revision_id=rev_id, created_at=self.clock.now(),
                        role=role_out.role, actor=role_out.actor, generation="llm", model_name=result.model_name,
                        body=role_out.summary, evidence_ids_json=evs, order_permissions=False)
                    self.session.add(analysis)
                    self.core._audit(actor, "add_model_analysis", "analysis", analysis.id,
                                     {"revision_id": rev_id, "role": role_out.role, "research_run_id": row.id}, None)
                    if role_out.role == "critic":
                        counter = Counterargument(id=new_id(), revision_id=rev_id, created_at=self.clock.now(),
                            actor=role_out.actor, generation="llm", model_name=result.model_name,
                            body=role_out.summary, evidence_ids_json=evs)
                        self.session.add(counter)
                        self.core._audit(actor, "add_model_counterargument", "counterargument", counter.id,
                                         {"revision_id": rev_id, "research_run_id": row.id}, None)
            elif result.critic:
                evs = [i for c in result.critic.claims for i in c.evidence_ids if i in ev_ids]
                self.core.add_counterargument(rev_id, CounterIn(
                    actor=result.critic.actor, body=result.critic.summary,
                    generation="deterministic_demo",
                    evidence_ids=evs,
                ))
            if result.mode != "model":
                for role_out in (result.researcher, result.critic, result.synthesizer):
                    if not role_out:
                        continue
                    evs = [i for c in role_out.claims for i in c.evidence_ids if i in ev_ids]
                    self.core.add_analysis(rev_id, AnalysisIn(
                        actor=role_out.actor, role=role_out.role, body=role_out.summary,
                        generation="deterministic_demo", evidence_ids=evs,
                    ))
        self.core._audit(actor, "run_research", "research_run", row.id, {"mode": result.mode, "status": result.status}, idempotency_key)
        self.session.commit()
        out = self._run_dict(row)
        return self.core._remember(idempotency_key, "run_research", request, out)

    def _run_dict(self, row: ResearchRun) -> dict:
        # Legacy model runs used one configured model for all roles; an older
        # version could record WIRED. Preserve the stored value for audit but do
        # not present it as independent review in the current interpretation.
        recorded_review = row.independent_llm_review
        result = _interpret_claims(row.result_json or {})
        if row.mode == "model":
            result["independent_llm_review"] = "NOT_WIRED"
        return {
            "id": row.id, "created_at": row.created_at.isoformat(), "thesis_id": row.thesis_id,
            "revision_id": row.revision_id, "mode": row.mode, "status": row.status,
            "independent_llm_review": "NOT_WIRED" if row.mode == "model" else recorded_review,
            "recorded_independent_llm_review": recorded_review, "prompt_hash": row.prompt_hash,
            "input_hash": row.input_hash, "model_name": row.model_name, "model_version": row.model_version,
            "token_cost": row.token_cost, "result": result, "actor": row.actor,
        }

    def create_sandbox_plan(
        self, *, thesis_id: str, account_id: str, symbol: str, side: str, entry_price: float,
        stop_loss: float, leverage: float, research_run_id: str | None, stale: bool = False,
        live_mode: bool = False,
    ) -> dict:
        if symbol not in CONTRACTS:
            raise ValidationFailed(f"unsupported contract {symbol}")
        if live_mode:
            raise ValidationFailed("live-mode plans are disabled; exchange orders are not available")
        if stale:
            raise ValidationFailed("stale or missing marks block an actionable plan")
        self.paper.get_account(account_id)
        cfg, cfg_hash = load_owner_risk()
        thesis = self.core.get_thesis(thesis_id)
        rev_id = thesis["current_revision_id"]
        if research_run_id:
            run = self.session.get(ResearchRun, research_run_id)
            if not run:
                raise NotFound("research run not found")
            if run.revision_id != rev_id:
                raise ValidationFailed("research run is bound to a superseded revision")
            if run.result_json and run.result_json.get("mode") == "fixture" and False:
                pass  # fixture may inform a plan but cannot auto-approve
        ev_ids = _evidence_bundle(self.core, rev_id)[1]
        ev_hash = content_hash(repr(sorted(ev_ids)).encode())
        port = self.core.create_portfolio(self.core.default_sample_portfolio(mark=entry_price, quantity=0.0))
        risk = self.core.create_risk_config(RiskConfigIn(payload={**cfg, "owner_risk_hash": cfg_hash}, notes="owner paper risk file snapshot"))
        created = self.core.create_plan(PlanIn(
            revision_id=rev_id, portfolio_id=port["id"], risk_config_id=risk["id"],
            leverage=leverage, market_type="perp", margin=True,
            margin_mode=cfg.get("margin_mode") or "isolated", side=side,
            stop_loss=stop_loss, require_fresh_evidence=False,
        ))
        row = self.session.get(Plan, created["id"])
        row.request_json = {
            **(row.request_json or {}),
            "execution_registry": "local_simplified",
            "live_orders": False,
            "exchange_demo_orders": False,
            "sandbox": True,
            "never_real_performance": True,
            "contract_symbol": symbol,
            "instrument_id": CONTRACTS[symbol]["instrument_id"],
            "instrument_type": "perpetual",
            "venue": "bitget",
            "settlement_currency": "USDT",
            "contract_size": CONTRACTS[symbol]["contract_size"],
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "side": side,
            "leverage": leverage,
            "research_run_id": research_run_id,
            "evidence_ids": ev_ids,
            "evidence_hash": ev_hash,
            "owner_risk_hash": cfg_hash,
            "paper_account_id": account_id,
            "margin_model_supported": False,
            "liquidation_model_supported": False,
            "liquidation_model": "approximate_stress_not_venue_verified",
        }
        row.side = side
        row.leverage = leverage
        row.stop_loss = stop_loss
        row.margin_mode = cfg.get("margin_mode") or "isolated"
        row.margin_model_supported = False
        row.liquidation_model_supported = False
        row.actionable = True
        row.status = "proposed"
        venue_blocks = (row.risk_result_json or {}).get("block_reasons") or []
        row.risk_result_json = {
            **(row.risk_result_json or {}),
            "paper_registry": REGISTRY["local_simplified"],
            "venue_model_still_unsupported": True,
            "sandbox": True,
            "block_reasons_venue": venue_blocks,
        }
        self.session.commit()
        return self.core._plan_dict(row)

    def approve_paper(self, plan_id: str, payload: ApprovalIn, idempotency_key: str | None = None) -> dict:
        plan = self.session.get(Plan, plan_id)
        if not plan:
            raise NotFound("plan not found")
        req = plan.request_json or {}
        if req.get("execution_registry") != "local_simplified":
            raise ValidationFailed("this approval path is only for local_simplified paper")
        thesis_rev = self.session.get(ThesisRevision, plan.revision_id)
        ev_ids = _evidence_bundle(self.core, plan.revision_id)[1]
        ev_hash = content_hash(repr(sorted(ev_ids)).encode())
        if ev_hash != req.get("evidence_hash"):
            raise ValidationFailed("evidence version changed; approval refused")
        if req.get("research_run_id"):
            run = self.session.get(ResearchRun, req["research_run_id"])
            if not run or run.revision_id != plan.revision_id:
                raise ValidationFailed("research run does not match the plan revision")
        cfg, cfg_hash = load_owner_risk()
        if cfg_hash != req.get("owner_risk_hash"):
            raise ValidationFailed("owner risk file changed; approval refused")
        return self.core.approve_plan(plan_id, payload, idempotency_key)
