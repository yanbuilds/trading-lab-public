from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

class UTCDateTime(TypeDecorator):
    """Store timestamps in UTC; restore tzinfo on SQLite which drops it."""
    impl = DateTime
    cache_ok = True
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass

class Instrument(Base):
    __tablename__ = "instruments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    instrument_type: Mapped[str] = mapped_column(String(32), nullable=False)  # spot | perpetual | future | equity | index
    venue: Mapped[str | None] = mapped_column(String(64), nullable=True)
    contract_symbol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settlement_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    contract_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)  # spot | perp
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    theses: Mapped[list["Thesis"]] = relationship(back_populates="instrument")

class EvidenceItem(Base):
    __tablename__ = "evidence_items"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    quote_locator: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    available_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    data_quality: Mapped[str] = mapped_column(String(32), nullable=False)
    methodology: Mapped[str] = mapped_column(Text, nullable=False, default="")
    methodology_version: Mapped[str] = mapped_column(String(64), nullable=False, default="unspecified")
    vintage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False, default="user_provided")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

class EvidenceLink(Base):
    __tablename__ = "evidence_links"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence_items.id"), nullable=False)
    relation: Mapped[str] = mapped_column(String(16), nullable=False)  # supports | refutes
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    __table_args__ = (UniqueConstraint("revision_id", "evidence_id", "relation", name="uq_rev_ev_rel"),)

class Thesis(Base):
    __tablename__ = "theses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.id"), nullable=False)
    author: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    current_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    instrument: Mapped[Instrument] = relationship(back_populates="theses")
    revisions: Mapped[list["ThesisRevision"]] = relationship(back_populates="thesis", order_by="ThesisRevision.revision_no")

class ThesisRevision(Base):
    __tablename__ = "thesis_revisions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thesis_id: Mapped[str] = mapped_column(ForeignKey("theses.id"), nullable=False)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    author: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)  # long | short | flat
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    invalidation: Mapped[str] = mapped_column(Text, nullable=False)
    subjective_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    change_reason: Mapped[str] = mapped_column(Text, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    is_example: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    thesis: Mapped[Thesis] = relationship(back_populates="revisions")
    __table_args__ = (UniqueConstraint("thesis_id", "revision_no", name="uq_thesis_revno"),)

class Counterargument(Base):
    __tablename__ = "counterarguments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    generation: Mapped[str] = mapped_column(String(32), nullable=False)
    # generation: manual | deterministic_demo | llm  — if not llm, independent LLM review is NOT wired
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

class Analysis(Base):
    """Structured researcher / critic / synthesizer record.

    Public input accepts manual or labelled deterministic_demo rows only.
    The internal model workflow can append llm rows with provenance.
    Independent LLM review is NOT wired. These roles have no order permissions.
    """
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # researcher | critic | synthesizer
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    generation: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    order_permissions: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    as_of: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    liabilities: Mapped[float] = mapped_column(Float, nullable=False)
    positions_json: Mapped[list] = mapped_column(JSON, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # positions: [{instrument_id, quantity, cost, mark, mark_as_of, mark_source, liquidity}]

class RiskConfig(Base):
    __tablename__ = "risk_configs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

class Plan(Base):
    __tablename__ = "plans"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolio_snapshots.id"), nullable=False)
    risk_config_id: Mapped[str] = mapped_column(ForeignKey("risk_configs.id"), nullable=False)
    plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # blocked | proposed | expired | superseded
    actionable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False, default="long")
    leverage: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    margin_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)  # isolated | cross
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_condition: Mapped[str | None] = mapped_column(Text, nullable=True)
    margin_model_supported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    liquidation_model_supported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    request_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    risk_result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

class Decision(Base):
    __tablename__ = "decisions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    recommendation: Mapped[str] = mapped_column(String(32), nullable=False)
    user_choice: Mapped[str] = mapped_column(String(32), nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    approval_binding_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    invalid_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

class Outcome(Base):
    __tablename__ = "outcomes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    start_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    quote_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    quote_observed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    result: Mapped[str] = mapped_column(String(32), nullable=False)  # correct | incorrect | flat | unknown
    directional_move: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    __table_args__ = (Index("ix_outcomes_revision", "revision_id"),)

class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    __table_args__ = (UniqueConstraint("key", "action", name="uq_idem_key_action"),)

class ResearchRun(Base):
    """Immutable on-demand research record."""
    __tablename__ = "research_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    thesis_id: Mapped[str] = mapped_column(ForeignKey("theses.id"), nullable=False)
    revision_id: Mapped[str] = mapped_column(ForeignKey("thesis_revisions.id"), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    independent_llm_review: Mapped[str] = mapped_column(String(16), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")


class PaperAccount(Base):
    __tablename__ = "paper_accounts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    starting_cash: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    liabilities: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    engine: Mapped[str] = mapped_column(String(32), nullable=False, default="local_simplified")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")


class PaperLedger(Base):
    __tablename__ = "paper_ledger"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    cash_after: Mapped[float] = mapped_column(Float, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ref_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provenance: Mapped[str] = mapped_column(String(64), nullable=False, default="assumed")


class PaperOrder(Base):
    __tablename__ = "paper_orders"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), nullable=False)
    plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    research_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    leverage: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sandbox: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument_id: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    leverage: Mapped[float] = mapped_column(Float, nullable=False)
    margin: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    entry_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    exit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
