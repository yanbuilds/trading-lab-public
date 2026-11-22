from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

Role = Literal["researcher", "critic", "synthesizer"]
Direction = Literal["long", "short", "flat"]
ClaimStatus = Literal["supported", "unsupported", "unverified", "unverified_url", "invalid_ref"]

FORBIDDEN_OUTPUT_FIELDS = {
    "order_permissions", "approve", "execute", "api_key", "leverage_override",
    "margin_model_supported", "kill_switch", "live", "orders", "withdraw",
}

class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = "unsupported"

class RoleOutput(BaseModel):
    role: Role
    actor: str
    generation: Literal["deterministic_demo", "model"] = "deterministic_demo"
    direction: Direction | None = None
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    invalidation: str = ""
    data_quality_notes: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    # Suggested action is advisory only and cannot approve or execute.
    suggested_action: Literal["review_paper_plan", "stand_aside", "need_more_evidence"] = "review_paper_plan"

    @field_validator("summary", "invalidation")
    @classmethod
    def not_instruction_route(cls, v: str) -> str:
        return v

class WorkflowResult(BaseModel):
    mode: Literal["fixture", "needs_model_configuration", "model"]
    independent_llm_review: Literal["NOT_WIRED", "WIRED"]
    researcher: RoleOutput | None = None
    critic: RoleOutput | None = None
    synthesizer: RoleOutput | None = None
    disagreements: list[str] = Field(default_factory=list)
    blocked_claims: list[dict[str, Any]] = Field(default_factory=list)
    injection_flags: list[str] = Field(default_factory=list)
    prompt_hash: str
    input_hash: str
    model_name: str | None = None
    model_provider: str | None = None
    model_version: str | None = None
    token_cost: float | None = None  # unknown unless the provider returns it
    usage_records: list[dict[str, Any]] = Field(default_factory=list)
    usage_complete: bool | None = None
    forecast_calibration_status: Literal["not_calibrated"] = "not_calibrated"
    status: str
    message: str = ""

def sanitize_model_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop illegal control fields so a model cannot mutate hard limits or routes."""
    cleaned = {k: v for k, v in data.items() if k not in FORBIDDEN_OUTPUT_FIELDS}
    # Nested attempts
    for key in list(cleaned):
        if isinstance(cleaned[key], dict):
            cleaned[key] = sanitize_model_dict(cleaned[key])
    return cleaned
