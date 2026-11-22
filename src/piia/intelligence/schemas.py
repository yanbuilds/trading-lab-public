from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

Direction = Literal["long", "short", "flat"]
Relation = Literal["supports", "refutes"]
Generation = Literal["manual", "deterministic_demo"]
Quality = Literal["high", "medium", "low", "missing", "unverified"]

class EvidenceIn(BaseModel):
    raw: str
    source_url: str | None = None
    provider: str
    quote_locator: str = ""
    observed_at: datetime | None = None
    published_at: datetime | None = None
    available_at: datetime | None = None
    timezone: str = "UTC"
    data_quality: Quality = "unverified"
    methodology: str = ""
    methodology_version: str = "unspecified"
    vintage: str | None = None
    label: str = "user_provided"
    relation: Relation = "supports"
    note: str = ""
    # Reject late vintages used as historic evidence when this flag is set.
    historic_context: bool = False

    @field_validator("raw")
    @classmethod
    def raw_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw evidence snapshot is required")
        return v

class ThesisCreate(BaseModel):
    instrument_id: str = "btc-usd-spot"
    author: str
    direction: Direction
    horizon_days: int = Field(ge=1, le=3650)
    invalidation: str
    subjective_confidence: float = Field(ge=0, le=1)
    statement: str
    label: str = "LABELED_FIXTURE_EXAMPLE"
    is_example: bool = True
    evidence: list[EvidenceIn] = Field(default_factory=list)
    change_reason: str = "initial revision"

class ThesisRevise(BaseModel):
    author: str
    direction: Direction
    horizon_days: int = Field(ge=1, le=3650)
    invalidation: str
    subjective_confidence: float = Field(ge=0, le=1)
    statement: str
    change_reason: str
    expected_revision_id: str
    evidence: list[EvidenceIn] = Field(default_factory=list)
    is_example: bool = True

    @field_validator("change_reason")
    @classmethod
    def reason_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("change_reason is required when appending a revision")
        return v

class CounterIn(BaseModel):
    actor: str
    body: str
    generation: Generation = "manual"
    evidence_ids: list[str] = Field(default_factory=list)

class AnalysisIn(BaseModel):
    actor: str
    role: Literal["researcher", "critic", "synthesizer"]
    body: str
    generation: Generation = "manual"
    evidence_ids: list[str] = Field(default_factory=list)


class PortfolioPositionIn(BaseModel):
    instrument_id: str
    quantity: float | None = None
    cost: float | None = None
    mark: float | None = None
    mark_as_of: datetime | None = None
    mark_source: str | None = None
    liquidity: str = "unknown"

class PortfolioIn(BaseModel):
    as_of: datetime
    label: str = "LABELED_FIXTURE_SAMPLE"
    cash: float
    liabilities: float = 0
    positions: list[PortfolioPositionIn]
    notes: str = "Authorized manual sample fixture only; not a live brokerage account."

class RiskConfigIn(BaseModel):
    payload: dict[str, Any]
    notes: str = ""

class PlanIn(BaseModel):
    revision_id: str
    portfolio_id: str
    risk_config_id: str
    leverage: float = 1.0
    market_type: str | None = None  # filled from instrument if omitted
    margin: bool = False
    margin_mode: str | None = None  # isolated | cross
    side: Direction = "long"
    stop_loss: float | None = None
    take_profit: float | None = None
    exit_condition: str | None = None
    notional: float | None = None
    missing_critical_evidence: bool = False
    require_fresh_evidence: bool = True

class ApprovalIn(BaseModel):
    actor: str
    user_choice: Literal["approve", "reject"]
    rejection_reason: str | None = None

class OutcomeQuote(BaseModel):
    price: float | None = None
    source: str | None = None
    observed_at: datetime | None = None

class OutcomeIn(BaseModel):
    revision_id: str
    start: OutcomeQuote
    end: OutcomeQuote
