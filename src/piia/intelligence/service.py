from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, func, update
from sqlalchemy.orm import Session

from piia.intelligence import risk as risk_engine
from piia.intelligence.catalog import INSTRUMENTS
from piia.intelligence.clock import Clock
from piia.intelligence.errors import Conflict, Immutable, NotFound, ValidationFailed
from piia.intelligence.hashing import content_hash, read_blob, write_blob
from piia.intelligence.ids import new_id
from piia.intelligence.models import (
    Analysis, AuditEvent, Counterargument, Decision, EvidenceItem, EvidenceLink,
    IdempotencyKey, Instrument, Outcome, Plan, PortfolioSnapshot, RiskConfig,
    Thesis, ThesisRevision,
)
from piia.intelligence.paths import EVIDENCE
from piia.intelligence.schemas import (
    AnalysisIn, ApprovalIn, CounterIn, EvidenceIn, OutcomeIn, PlanIn,
    PortfolioIn, PortfolioPositionIn, RiskConfigIn, ThesisCreate, ThesisRevise,
)

ALLOWED_GENERATIONS = {"manual", "deterministic_demo"}
# Confidence is always stored as subjective; never presented as calibrated accuracy.
MEASURED_ACCURACY_STATUS = "not_calibrated"
OUTCOME_QUOTE_WINDOW = timedelta(minutes=5)

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))

def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return dict(model)

class IntelligenceService:
    """Authoritative Phase-1 audit service. No trade/executor methods."""

    def __init__(self, session: Session, clock: Clock | None = None, evidence_store: Path | None = None):
        self.session = session
        self.clock = clock or Clock()
        self.store = Path(evidence_store or EVIDENCE)
        self.store.mkdir(parents=True, exist_ok=True)

    # ---- instruments ----
    def seed_instruments(self) -> list[dict]:
        out = []
        for row in INSTRUMENTS:
            existing = self.session.get(Instrument, row["id"])
            if existing is None:
                self.session.add(Instrument(**row))
            out.append(row)
        self.session.commit()
        return out

    def list_instruments(self) -> list[Instrument]:
        return list(self.session.scalars(select(Instrument).order_by(Instrument.id)))

    def get_instrument(self, instrument_id: str) -> Instrument:
        inst = self.session.get(Instrument, instrument_id)
        if not inst:
            raise NotFound(f"instrument not found: {instrument_id}")
        return inst

    # ---- idempotency ----
    def _idempotent(self, key: str | None, action: str, request: Any) -> dict | None:
        if not key:
            return None
        req_hash = content_hash(_canon(_dump(request) if not isinstance(request, dict) else request).encode())
        row = self.session.get(IdempotencyKey, key)
        if row:
            if row.action != action or row.request_hash != req_hash:
                raise Conflict("idempotency key reused with a different request")
            return row.response_json
        return None

    def _remember(self, key: str | None, action: str, request: Any, response: dict) -> dict:
        if key:
            req_hash = content_hash(_canon(_dump(request) if not isinstance(request, dict) else request).encode())
            self.session.merge(IdempotencyKey(
                key=key, action=action, request_hash=req_hash,
                response_json=response, created_at=self.clock.now(),
            ))
            self.session.commit()
        return response

    def _audit(self, actor: str, action: str, entity_type: str, entity_id: str, payload: dict, request_id: str | None = None) -> None:
        # Never put secrets into audit payloads; callers must already redact.
        self.session.add(AuditEvent(
            id=new_id(), created_at=self.clock.now(), actor=actor, action=action,
            entity_type=entity_type, entity_id=entity_id, request_id=request_id,
            payload_json=payload,
        ))

    # ---- evidence ----
    def _reject_late_vintage(self, ev: EvidenceIn) -> None:
        if not ev.historic_context:
            return
        observed = _aware(ev.observed_at)
        available = _aware(ev.available_at)
        vintage_dt = None
        if ev.vintage:
            try:
                vintage_dt = _aware(datetime.fromisoformat(ev.vintage))
            except ValueError:
                vintage_dt = None
        late_from = vintage_dt or available
        if observed and late_from and late_from > observed:
            raise ValidationFailed(
                "late vintage rejected as historic evidence: data became available after the observation it would have been used to justify"
            )

    def ingest_evidence(self, ev: EvidenceIn) -> EvidenceItem:
        self._reject_late_vintage(ev)
        raw = ev.raw.encode("utf-8")
        digest = write_blob(self.store, raw)
        existing = self.session.scalar(select(EvidenceItem).where(EvidenceItem.content_hash == digest))
        if existing:
            return existing
        now = self.clock.now()
        item = EvidenceItem(
            id=new_id(),
            source_url=ev.source_url,
            provider=ev.provider,
            content_hash=digest,
            quote_locator=ev.quote_locator,
            observed_at=_aware(ev.observed_at),
            published_at=_aware(ev.published_at),
            available_at=_aware(ev.available_at),
            retrieved_at=now,
            timezone=ev.timezone or "UTC",
            data_quality=ev.data_quality,
            methodology=ev.methodology,
            methodology_version=ev.methodology_version,
            vintage=ev.vintage,
            label=ev.label,
            created_at=now,
        )
        self.session.add(item)
        self.session.flush()
        return item

    def verify_evidence(self, evidence_id: str) -> dict:
        item = self.session.get(EvidenceItem, evidence_id)
        if not item:
            raise NotFound(f"evidence not found: {evidence_id}")
        try:
            data = read_blob(self.store, item.content_hash)
        except FileNotFoundError as exc:
            raise ValidationFailed(f"evidence blob missing for {evidence_id}") from exc
        except ValueError as exc:
            raise ValidationFailed(str(exc)) from exc
        return {"id": item.id, "content_hash": item.content_hash, "verified": True, "bytes": len(data)}

    def read_evidence_text(self, evidence_id: str) -> str:
        item = self.session.get(EvidenceItem, evidence_id)
        if not item:
            raise NotFound(f"evidence not found: {evidence_id}")
        return read_blob(self.store, item.content_hash).decode("utf-8")

    def get_evidence(self, evidence_id: str) -> EvidenceItem:
        item = self.session.get(EvidenceItem, evidence_id)
        if not item:
            raise NotFound(f"evidence not found: {evidence_id}")
        return item

    def _require_evidence_ids(self, ids: list[str]) -> None:
        for eid in ids:
            if not self.session.get(EvidenceItem, eid):
                raise ValidationFailed(f"missing evidence reference: {eid}")

    # ---- theses ----
    def create_thesis(self, payload: ThesisCreate, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "create_thesis", payload)
        if cached is not None:
            return cached
        inst = self.get_instrument(payload.instrument_id)
        now = self.clock.now()
        thesis = Thesis(
            id=new_id(), instrument_id=inst.id, author=payload.author,
            created_at=now, label=payload.label, current_revision_id=None,
        )
        self.session.add(thesis)
        self.session.flush()
        rev = self._new_revision(thesis, payload, now, revision_no=1, change_reason=payload.change_reason or "initial revision")
        thesis.current_revision_id = rev.id
        self._audit(payload.author, "create_thesis", "thesis", thesis.id, {"revision_id": rev.id, "label": payload.label}, idempotency_key)
        self.session.commit()
        result = self.get_thesis(thesis.id)
        return self._remember(idempotency_key, "create_thesis", payload, result)

    def _new_revision(self, thesis: Thesis, payload: ThesisCreate | ThesisRevise, now: datetime, revision_no: int, change_reason: str) -> ThesisRevision:
        expires = now + timedelta(days=int(payload.horizon_days))
        rev = ThesisRevision(
            id=new_id(), thesis_id=thesis.id, revision_no=revision_no, created_at=now,
            author=payload.author, direction=payload.direction, horizon_days=payload.horizon_days,
            expires_at=expires, invalidation=payload.invalidation,
            subjective_confidence=payload.subjective_confidence, change_reason=change_reason,
            statement=payload.statement, is_example=payload.is_example,
        )
        self.session.add(rev)
        self.session.flush()
        for ev in payload.evidence:
            item = self.ingest_evidence(ev)
            self.session.add(EvidenceLink(
                id=new_id(), revision_id=rev.id, evidence_id=item.id,
                relation=ev.relation, note=ev.note,
            ))
        return rev

    def revise_thesis(self, thesis_id: str, payload: ThesisRevise, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "revise_thesis", {"thesis_id": thesis_id, **_dump(payload)})
        if cached is not None:
            return cached
        if not payload.change_reason.strip():
            raise ValidationFailed("change_reason is required when appending a revision")
        thesis = self.session.get(Thesis, thesis_id)
        if not thesis:
            raise NotFound(f"thesis not found: {thesis_id}")
        if thesis.current_revision_id != payload.expected_revision_id:
            raise Conflict("expected_revision_id does not match current revision (optimistic concurrency)")
        current = self.session.get(ThesisRevision, thesis.current_revision_id)
        now = self.clock.now()
        rev = self._new_revision(thesis, payload, now, revision_no=current.revision_no + 1, change_reason=payload.change_reason)
        thesis.current_revision_id = rev.id
        self._supersede_plans_for_thesis(thesis_id, "superseded by a new thesis revision")
        self._audit(payload.author, "revise_thesis", "thesis", thesis.id, {"revision_id": rev.id, "change_reason": payload.change_reason}, idempotency_key)
        self.session.commit()
        result = self.get_thesis(thesis.id)
        return self._remember(idempotency_key, "revise_thesis", {"thesis_id": thesis_id, **_dump(payload)}, result)

    def get_thesis(self, thesis_id: str) -> dict:
        thesis = self.session.get(Thesis, thesis_id)
        if not thesis:
            raise NotFound(f"thesis not found: {thesis_id}")
        revs = list(self.session.scalars(
            select(ThesisRevision).where(ThesisRevision.thesis_id == thesis_id).order_by(ThesisRevision.revision_no)
        ))
        return {
            "id": thesis.id,
            "instrument_id": thesis.instrument_id,
            "author": thesis.author,
            "created_at": thesis.created_at.isoformat(),
            "label": thesis.label,
            "current_revision_id": thesis.current_revision_id,
            "measured_accuracy_status": MEASURED_ACCURACY_STATUS,
            "subjective_confidence_is_not_calibrated_accuracy": True,
            "revisions": [self._revision_dict(r) for r in revs],
        }

    def _revision_dict(self, rev: ThesisRevision) -> dict:
        links = list(self.session.scalars(select(EvidenceLink).where(EvidenceLink.revision_id == rev.id)))
        counters = list(self.session.scalars(select(Counterargument).where(Counterargument.revision_id == rev.id)))
        analyses = list(self.session.scalars(select(Analysis).where(Analysis.revision_id == rev.id)))
        outcomes = list(self.session.scalars(select(Outcome).where(Outcome.revision_id == rev.id).order_by(Outcome.created_at)))
        return {
            "id": rev.id,
            "revision_no": rev.revision_no,
            "created_at": rev.created_at.isoformat(),
            "author": rev.author,
            "direction": rev.direction,
            "horizon_days": rev.horizon_days,
            "expires_at": rev.expires_at.isoformat(),
            "invalidation": rev.invalidation,
            "subjective_confidence": rev.subjective_confidence,
            "measured_accuracy": None,
            "measured_accuracy_status": MEASURED_ACCURACY_STATUS,
            "change_reason": rev.change_reason,
            "statement": rev.statement,
            "is_example": rev.is_example,
            "evidence": [self._link_dict(l) for l in links],
            "counterarguments": [self._counter_dict(c) for c in counters],
            "analyses": [self._analysis_dict(a) for a in analyses],
            "outcomes": [self._outcome_dict(o) for o in outcomes],
        }

    def _link_dict(self, link: EvidenceLink) -> dict:
        ev = self.session.get(EvidenceItem, link.evidence_id)
        return {
            "relation": link.relation,
            "note": link.note,
            "evidence": self._evidence_dict(ev) if ev else {"id": link.evidence_id, "missing": True},
        }

    def _evidence_dict(self, ev: EvidenceItem) -> dict:
        return {
            "id": ev.id,
            "source_url": ev.source_url,
            "provider": ev.provider,
            "content_hash": ev.content_hash,
            "quote_locator": ev.quote_locator,
            "observed_at": ev.observed_at.isoformat() if ev.observed_at else None,
            "published_at": ev.published_at.isoformat() if ev.published_at else None,
            "available_at": ev.available_at.isoformat() if ev.available_at else None,
            "retrieved_at": ev.retrieved_at.isoformat(),
            "timezone": ev.timezone,
            "data_quality": ev.data_quality,
            "methodology": ev.methodology,
            "methodology_version": ev.methodology_version,
            "vintage": ev.vintage,
            "label": ev.label,
        }

    def _counter_dict(self, c: Counterargument) -> dict:
        return {
            "id": c.id, "created_at": c.created_at.isoformat(), "actor": c.actor,
            "generation": c.generation, "model_name": c.model_name, "body": c.body,
            "evidence_ids": c.evidence_ids_json or [],
            "independent_llm_review": "NOT_WIRED",
        }

    def _analysis_dict(self, a: Analysis) -> dict:
        return {
            "id": a.id, "created_at": a.created_at.isoformat(), "role": a.role, "actor": a.actor,
            "generation": a.generation, "model_name": a.model_name, "body": a.body,
            "evidence_ids": a.evidence_ids_json or [],
            "order_permissions": False,
            "independent_llm_review": "NOT_WIRED",
        }

    def _outcome_dict(self, o: Outcome) -> dict:
        return {
            "id": o.id, "created_at": o.created_at.isoformat(), "evaluated_at": o.evaluated_at.isoformat(),
            "start_price": o.start_price, "end_price": o.end_price, "quote_source": o.quote_source,
            "quote_observed_at": o.quote_observed_at.isoformat() if o.quote_observed_at else None,
            "result": o.result, "directional_move": o.directional_move, "notes": o.notes,
            "quote_verification": "user_supplied_not_independently_verified",
            "eligible_for_calibration": False,
        }

    def deny_mutation(self, kind: str, _id: str) -> None:
        raise Immutable(f"{kind} records are append-only; update and delete are not provided")

    # ---- analyses / counters ----
    def _check_generation(self, generation: str, model_name: str | None = None) -> None:
        if generation not in ALLOWED_GENERATIONS:
            raise ValidationFailed(
                "Phase 1 only accepts generation=manual or deterministic_demo; independent LLM review is NOT wired"
            )
        if model_name:
            raise ValidationFailed("model_name is reserved for a real wired model; none is configured")

    def add_counterargument(self, revision_id: str, payload: CounterIn, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "add_counterargument", {"revision_id": revision_id, **_dump(payload)})
        if cached is not None:
            return cached
        rev = self.session.get(ThesisRevision, revision_id)
        if not rev:
            raise NotFound(f"revision not found: {revision_id}")
        self._check_generation(payload.generation)
        self._require_evidence_ids(payload.evidence_ids)
        row = Counterargument(
            id=new_id(), revision_id=revision_id, created_at=self.clock.now(),
            actor=payload.actor, generation=payload.generation, model_name=None,
            body=payload.body, evidence_ids_json=payload.evidence_ids,
        )
        self.session.add(row)
        self._audit(payload.actor, "add_counterargument", "counterargument", row.id, {"revision_id": revision_id}, idempotency_key)
        self.session.commit()
        result = self._counter_dict(row)
        return self._remember(idempotency_key, "add_counterargument", {"revision_id": revision_id, **_dump(payload)}, result)

    def add_analysis(self, revision_id: str, payload: AnalysisIn, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "add_analysis", {"revision_id": revision_id, **_dump(payload)})
        if cached is not None:
            return cached
        rev = self.session.get(ThesisRevision, revision_id)
        if not rev:
            raise NotFound(f"revision not found: {revision_id}")
        self._check_generation(payload.generation)
        self._require_evidence_ids(payload.evidence_ids)
        row = Analysis(
            id=new_id(), revision_id=revision_id, created_at=self.clock.now(),
            role=payload.role, actor=payload.actor, generation=payload.generation,
            model_name=None, body=payload.body, evidence_ids_json=payload.evidence_ids,
            order_permissions=False,
        )
        self.session.add(row)
        self._audit(payload.actor, "add_analysis", "analysis", row.id, {"revision_id": revision_id, "role": payload.role}, idempotency_key)
        self.session.commit()
        result = self._analysis_dict(row)
        return self._remember(idempotency_key, "add_analysis", {"revision_id": revision_id, **_dump(payload)}, result)

    def deterministic_demo_counter(self, revision_id: str, actor: str = "piia-deterministic-demo") -> dict:
        """Labelled deterministic template. This is NOT an independent LLM review."""
        rev = self.session.get(ThesisRevision, revision_id)
        if not rev:
            raise NotFound(f"revision not found: {revision_id}")
        opposite = {"long": "continue falling or stay flat", "short": "continue rising or stay flat", "flat": "make a large directional move"}
        body = (
            "DETERMINISTIC_DEMO (not independent LLM research): "
            f"The thesis is {rev.direction} over {rev.horizon_days} days. "
            f"A contradicting path is that price could {opposite.get(rev.direction, 'move adversely')}. "
            f"Stated invalidation: {rev.invalidation}. "
            "This template does not inspect new market data and must not be treated as an independent critique."
        )
        return self.add_counterargument(revision_id, CounterIn(actor=actor, body=body, generation="deterministic_demo"))

    # ---- portfolio / risk config ----
    def create_portfolio(self, payload: PortfolioIn, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "create_portfolio", payload)
        if cached is not None:
            return cached
        now = self.clock.now()
        positions = []
        for p in payload.positions:
            positions.append({
                "instrument_id": p.instrument_id,
                "quantity": p.quantity, "cost": p.cost, "mark": p.mark,
                "mark_as_of": p.mark_as_of.isoformat() if p.mark_as_of else None,
                "mark_source": p.mark_source, "liquidity": p.liquidity or "unknown",
            })
        row = PortfolioSnapshot(
            id=new_id(), created_at=now, as_of=_aware(payload.as_of) or now,
            label=payload.label, cash=payload.cash, liabilities=payload.liabilities,
            positions_json=positions, notes=payload.notes,
        )
        self.session.add(row)
        self._invalidate_open_approvals("new portfolio snapshot")
        self._audit("system", "create_portfolio", "portfolio_snapshot", row.id, {"label": payload.label}, idempotency_key)
        self.session.commit()
        result = self._portfolio_dict(row)
        return self._remember(idempotency_key, "create_portfolio", payload, result)

    def _portfolio_dict(self, row: PortfolioSnapshot) -> dict:
        return {
            "id": row.id, "created_at": row.created_at.isoformat(), "as_of": row.as_of.isoformat(),
            "label": row.label, "cash": row.cash, "liabilities": row.liabilities,
            "positions": row.positions_json, "notes": row.notes,
        }

    def create_risk_config(self, payload: RiskConfigIn, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "create_risk_config", payload)
        if cached is not None:
            return cached
        merged = {**risk_engine.DEFAULT_CONFIG, **(payload.payload or {})}
        digest = content_hash(_canon(merged).encode())
        existing = self.session.scalar(select(RiskConfig).where(RiskConfig.config_hash == digest))
        if existing:
            return self._risk_dict(existing)
        row = RiskConfig(id=new_id(), created_at=self.clock.now(), config_hash=digest, payload_json=merged, notes=payload.notes)
        self.session.add(row)
        self._invalidate_open_approvals("new risk config")
        self._audit("system", "create_risk_config", "risk_config", row.id, {"config_hash": digest}, idempotency_key)
        self.session.commit()
        result = self._risk_dict(row)
        return self._remember(idempotency_key, "create_risk_config", payload, result)

    def _risk_dict(self, row: RiskConfig) -> dict:
        return {"id": row.id, "created_at": row.created_at.isoformat(), "config_hash": row.config_hash, "payload": row.payload_json, "notes": row.notes}

    def get_portfolio(self, portfolio_id: str) -> PortfolioSnapshot:
        row = self.session.get(PortfolioSnapshot, portfolio_id)
        if not row:
            raise NotFound(f"portfolio not found: {portfolio_id}")
        return row

    def get_risk_config(self, risk_config_id: str) -> RiskConfig:
        row = self.session.get(RiskConfig, risk_config_id)
        if not row:
            raise NotFound(f"risk config not found: {risk_config_id}")
        return row

    # ---- plans / approvals ----
    def create_plan(self, payload: PlanIn, actor: str = "user", idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "create_plan", payload)
        if cached is not None:
            return cached
        rev = self.session.get(ThesisRevision, payload.revision_id)
        if not rev:
            raise NotFound(f"revision not found: {payload.revision_id}")
        thesis = self.session.get(Thesis, rev.thesis_id)
        inst = self.get_instrument(thesis.instrument_id)
        port = self.get_portfolio(payload.portfolio_id)
        cfg = self.get_risk_config(payload.risk_config_id)
        now = self.clock.now()
        instrument_type = inst.instrument_type
        market_type = payload.market_type or inst.market_type
        request = {
            "leverage": payload.leverage,
            "market_type": market_type,
            "instrument_type": instrument_type,
            "margin": payload.margin,
            "margin_mode": payload.margin_mode,
            "side": payload.side,
            "stop_loss": payload.stop_loss,
            "take_profit": payload.take_profit,
            "exit_condition": payload.exit_condition,
            "notional": payload.notional,
            "missing_critical_evidence": payload.missing_critical_evidence,
            "require_fresh_evidence": payload.require_fresh_evidence,
            "venue": inst.venue,
            "contract_symbol": inst.contract_symbol,
        }
        # Resolve position types from the authoritative catalog, not user-supplied
        # snapshot text. Derivative/unknown holdings cannot use spot NAV accounting.
        risk_positions = []
        for position in port.positions_json:
            held = self.session.get(Instrument, position.get("instrument_id"))
            risk_positions.append({**position, "instrument_type": held.instrument_type if held else "unknown"})
        result = risk_engine.evaluate(
            portfolio={"cash": port.cash, "liabilities": port.liabilities, "positions": risk_positions},
            config=cfg.payload_json,
            plan_request=request,
            now=now,
            latest_revision_id=thesis.current_revision_id or rev.id,
            bound_revision_id=rev.id,
        )
        ttl = float((cfg.payload_json or {}).get("plan_ttl_hours") or 24)
        expires = now + timedelta(hours=ttl)
        binding = {
            "revision_id": rev.id, "portfolio_id": port.id, "risk_config_id": cfg.id,
            "risk_config_hash": cfg.config_hash,
        }
        plan_hash = content_hash(_canon({**binding, **request, "risk": result}).encode())
        passed = bool(result.get("passed"))
        row = Plan(
            id=new_id(), created_at=now, revision_id=rev.id, portfolio_id=port.id,
            risk_config_id=cfg.id, plan_hash=plan_hash,
            status="proposed" if passed else "blocked",
            actionable=passed,
            side=payload.side, leverage=payload.leverage, margin_mode=payload.margin_mode,
            stop_loss=payload.stop_loss, take_profit=payload.take_profit,
            exit_condition=payload.exit_condition,
            margin_model_supported=bool(result.get("flags", {}).get("margin_model_supported")),
            liquidation_model_supported=bool(result.get("flags", {}).get("liquidation_model_supported")),
            request_json=request, risk_result_json=result, expires_at=expires,
        )
        self.session.add(row)
        self._audit(actor, "create_plan", "plan", row.id, {"status": row.status, "actionable": row.actionable}, idempotency_key)
        self.session.commit()
        out = self._plan_dict(row)
        return self._remember(idempotency_key, "create_plan", payload, out)

    def _plan_dict(self, row: Plan) -> dict:
        return {
            "id": row.id, "created_at": row.created_at.isoformat(), "revision_id": row.revision_id,
            "portfolio_id": row.portfolio_id, "risk_config_id": row.risk_config_id,
            "plan_hash": row.plan_hash, "status": row.status, "actionable": row.actionable,
            "side": row.side, "leverage": row.leverage, "margin_mode": row.margin_mode,
            "stop_loss": row.stop_loss, "take_profit": row.take_profit, "exit_condition": row.exit_condition,
            "margin_model_supported": row.margin_model_supported,
            "liquidation_model_supported": row.liquidation_model_supported,
            "request": row.request_json, "risk": row.risk_result_json,
            "expires_at": row.expires_at.isoformat(),
            "execution": "paper_only_no_trade_routes",
        }

    def _plan_invalid_reason(self, plan: Plan) -> str | None:
        now = self.clock.now()
        if (plan.risk_result_json or {}).get("engine") != risk_engine.ENGINE_VERSION:
            return "risk engine version changed; create a new plan"
        if now > plan.expires_at:
            return "plan expired"
        rev = self.session.get(ThesisRevision, plan.revision_id)
        thesis = self.session.get(Thesis, rev.thesis_id) if rev else None
        if thesis and thesis.current_revision_id != plan.revision_id:
            return "thesis revision superseded"
        newer_port = self.session.scalar(
            select(func.count()).select_from(PortfolioSnapshot).where(
                PortfolioSnapshot.created_at > plan.created_at,
                PortfolioSnapshot.id != plan.portfolio_id,
            )
        )
        if newer_port:
            return "portfolio snapshot superseded"
        newer_cfg = self.session.scalar(
            select(func.count()).select_from(RiskConfig).where(
                RiskConfig.created_at > plan.created_at,
                RiskConfig.id != plan.risk_config_id,
            )
        )
        if newer_cfg:
            return "risk config superseded"
        return None

    def _supersede_plans_for_thesis(self, thesis_id: str, reason: str) -> None:
        revs = list(self.session.scalars(select(ThesisRevision.id).where(ThesisRevision.thesis_id == thesis_id)))
        for plan in self.session.scalars(select(Plan).where(Plan.revision_id.in_(revs), Plan.status.in_(["proposed", "blocked"]))):
            plan.status = "superseded"
        for dec in self.session.scalars(select(Decision).where(Decision.valid.is_(True))):
            plan = self.session.get(Plan, dec.plan_id)
            if plan and plan.revision_id in set(revs):
                dec.valid = False
                dec.invalid_reason = reason

    def _invalidate_open_approvals(self, reason: str) -> None:
        for dec in self.session.scalars(select(Decision).where(Decision.valid.is_(True))):
            dec.valid = False
            dec.invalid_reason = reason
        for plan in self.session.scalars(select(Plan).where(Plan.status == "proposed")):
            plan.status = "superseded"

    def approve_plan(self, plan_id: str, payload: ApprovalIn, idempotency_key: str | None = None) -> dict:
        """Explicit HTTP/API user action. Models cannot invoke this automatically.

        No current-user authentication is implemented; the app binds 127.0.0.1 only
        and trusts the actor string supplied in the request. This is documented, not implied approval.
        """
        cached = self._idempotent(idempotency_key, "approve_plan", {"plan_id": plan_id, **_dump(payload)})
        if cached is not None:
            return cached
        plan = self.session.get(Plan, plan_id)
        if not plan:
            raise NotFound(f"plan not found: {plan_id}")
        invalid = self._plan_invalid_reason(plan)
        recommendation = "approve_paper" if plan.actionable and not invalid else "do_not_execute"
        if payload.user_choice == "approve" and (invalid or not plan.actionable):
            valid = False
            invalid_reason = invalid or "plan is not actionable"
        elif payload.user_choice == "reject":
            valid = False
            invalid_reason = payload.rejection_reason or "user rejected"
        else:
            valid = True
            invalid_reason = None
        cfg = self.get_risk_config(plan.risk_config_id)
        binding = {
            "plan_id": plan.id, "plan_hash": plan.plan_hash,
            "revision_id": plan.revision_id, "portfolio_id": plan.portfolio_id,
            "risk_config_id": plan.risk_config_id, "risk_config_hash": cfg.config_hash,
            "expires_at": plan.expires_at.isoformat(),
        }
        row = Decision(
            id=new_id(), created_at=self.clock.now(), plan_id=plan.id, actor=payload.actor,
            recommendation=recommendation, user_choice=payload.user_choice,
            rejection_reason=payload.rejection_reason, approval_binding_json=binding,
            valid=valid, invalid_reason=invalid_reason,
        )
        self.session.add(row)
        self._audit(payload.actor, "approve_plan", "decision", row.id, {"valid": valid, "choice": payload.user_choice}, idempotency_key)
        self.session.commit()
        result = self._decision_dict(row)
        return self._remember(idempotency_key, "approve_plan", {"plan_id": plan_id, **_dump(payload)}, result)

    def _decision_dict(self, row: Decision) -> dict:
        return {
            "id": row.id, "created_at": row.created_at.isoformat(), "plan_id": row.plan_id,
            "actor": row.actor, "recommendation": row.recommendation, "user_choice": row.user_choice,
            "rejection_reason": row.rejection_reason, "binding": row.approval_binding_json,
            "valid": row.valid, "invalid_reason": row.invalid_reason,
        }

    # ---- outcomes ----
    def evaluate_outcome(self, payload: OutcomeIn, idempotency_key: str | None = None) -> dict:
        cached = self._idempotent(idempotency_key, "evaluate_outcome", payload)
        if cached is not None:
            return cached
        rev = self.session.get(ThesisRevision, payload.revision_id)
        if not rev:
            raise NotFound(f"revision not found: {payload.revision_id}")
        now = self.clock.now()
        if now < rev.expires_at:
            raise ValidationFailed("outcome evaluation is premature: horizon has not expired")
        for quote in (payload.start, payload.end):
            obs = _aware(quote.observed_at)
            if obs and obs > now:
                raise ValidationFailed("future-dated quotes are rejected")
        end_obs = _aware(payload.end.observed_at)
        if end_obs and end_obs < rev.expires_at:
            raise ValidationFailed("evaluation quote must be at or after the predeclared horizon")
        cfg = risk_engine.DEFAULT_CONFIG
        threshold = float(cfg.get("flat_threshold") or 0.005)
        start_px, end_px = payload.start.price, payload.end.price
        start_obs = _aware(payload.start.observed_at)
        created = _aware(rev.created_at)
        expires = _aware(rev.expires_at)
        problems = []
        for label, quote in (("start", payload.start), ("end", payload.end)):
            if quote.price is None or not math.isfinite(quote.price) or quote.price <= 0:
                problems.append(f"{label}_price_missing_or_invalid")
            if not quote.source or not quote.source.strip():
                problems.append(f"{label}_source_missing")
            if quote.observed_at is None or quote.observed_at.tzinfo is None or quote.observed_at.utcoffset() is None:
                problems.append(f"{label}_timestamp_missing_or_naive")
        if start_obs and not created - OUTCOME_QUOTE_WINDOW <= start_obs <= created:
            problems.append("start_quote_outside_creation_window")
        if end_obs and not expires <= end_obs <= expires + OUTCOME_QUOTE_WINDOW:
            problems.append("end_quote_outside_horizon_window")
        notes = ["manual quote values and timestamps are unverified; not eligible for calibration",
                 f"start_source={payload.start.source!r}; start_observed_at={start_obs.isoformat() if start_obs else 'missing'}",
                 f"end_source={payload.end.source!r}; end_observed_at={end_obs.isoformat() if end_obs else 'missing'}"]
        if problems:
            result = "unknown"
            move = None
            notes.append("; ".join(problems))
        else:
            move = (float(end_px) - float(start_px)) / float(start_px) if start_px else None
            if move is None or not math.isfinite(move):
                move = None
                result = "unknown"
                notes.append("cannot compute finite directional move")
            elif abs(move) < threshold:
                observed = "flat"
                result = "correct" if rev.direction == "flat" else "flat"
                notes.append(f"move {move:.6f} within flat threshold {threshold}")
            else:
                observed = "long" if move > 0 else "short"
                result = "correct" if observed == rev.direction else "incorrect"
                notes.append(f"move {move:.6f} vs thesis {rev.direction}; flat threshold {threshold}")
        row = Outcome(
            id=new_id(), created_at=now, revision_id=rev.id, evaluated_at=now,
            start_price=start_px if start_px is not None and math.isfinite(start_px) else None,
            end_price=end_px if end_px is not None and math.isfinite(end_px) else None,
            quote_source=payload.end.source,
            quote_observed_at=end_obs, result=result, directional_move=move,
            notes="; ".join(notes),
        )
        self.session.add(row)
        self._audit("evaluator", "evaluate_outcome", "outcome", row.id, {"revision_id": rev.id, "result": result}, idempotency_key)
        self.session.commit()
        out = self._outcome_dict(row)
        return self._remember(idempotency_key, "evaluate_outcome", payload, out)

    # ---- audit trace ----
    def audit_trace(self, thesis_id: str) -> dict:
        thesis = self.get_thesis(thesis_id)
        # Collect related plans/decisions/events below.
        rev_ids = [r["id"] for r in thesis["revisions"]]
        plans = list(self.session.scalars(select(Plan).where(Plan.revision_id.in_(rev_ids)).order_by(Plan.created_at))) if rev_ids else []
        plan_ids = [p.id for p in plans]
        decisions = list(self.session.scalars(select(Decision).where(Decision.plan_id.in_(plan_ids)).order_by(Decision.created_at))) if plan_ids else []
        event_rows = list(self.session.scalars(select(AuditEvent).order_by(AuditEvent.created_at)))
        related = []
        interesting = {thesis_id, *rev_ids, *plan_ids, *[d.id for d in decisions]}
        for ev in event_rows:
            if ev.entity_id in interesting or ev.request_id:
                if ev.entity_id in interesting or (ev.payload_json and any(i in json.dumps(ev.payload_json) for i in interesting)):
                    related.append({
                        "id": ev.id, "created_at": ev.created_at.isoformat(), "actor": ev.actor,
                        "action": ev.action, "entity_type": ev.entity_type, "entity_id": ev.entity_id,
                        "request_id": ev.request_id,
                    })
        return {
            "thesis": thesis,
            "chain": ["thesis", "evidence", "countercase", "risk/plan", "decision", "outcome"],
            "plans": [self._plan_dict(p) for p in plans],
            "decisions": [self._decision_dict(d) for d in decisions],
            "audit_events": related,
        }

    def default_sample_portfolio(self, mark: float, mark_as_of: datetime | None = None, quantity: float = 0.001) -> PortfolioIn:
        now = mark_as_of or self.clock.now()
        return PortfolioIn(
            as_of=now,
            label="LABELED_FIXTURE_SAMPLE",
            cash=60.0,
            liabilities=0.0,
            positions=[
                PortfolioPositionIn(
                    instrument_id="btc-usd-spot",
                    quantity=quantity,
                    cost=mark,
                    mark=mark,
                    mark_as_of=now,
                    mark_source="labelled_fixture",
                    liquidity="unknown",
                )
            ],
            notes="Authorized manual sample fixture only; not a live brokerage account and not the user's real holdings.",
        )
