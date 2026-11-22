"""Optional official-model research with an evidence-only first critic pass.

All three calls use the same configured model. Separate contexts do not prove
independent models, statistical independence, truth, or permission to trade.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from piia.intelligence.agents.io import WorkflowResult
from piia.intelligence.agents.providers import PROMPT_HASH, OfficialAPIProvider, injection_flags
from piia.intelligence.agents.workflow import input_hash, validate_claims
from piia.intelligence.settings import model_configured, model_name, model_provider


def evidence_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Both first-pass roles see the same evidence and prior questions, not prior answers."""
    clean = deepcopy(payload)
    for key in ("researcher", "critic", "synthesizer", "researcher_opinion", "critic_opinion"):
        clean.pop(key, None)
    previous = clean.get("previous_discussion")
    if isinstance(previous, list):
        clean["previous_discussion"] = [
            {"question": str(item.get("question") or "")[:300]}
            for item in previous if isinstance(item, dict)
        ][-2:]
    elif previous is not None:
        clean["previous_discussion"] = []
    return clean


def run_independent_context_research(*, payload: dict[str, Any], legal_evidence_ids: list[str]) -> WorkflowResult:
    clean = evidence_only_payload(payload)
    ih = input_hash({"mode": "same_model_separate_first_pass_contexts_v1", "payload": clean, "prompt_hash": PROMPT_HASH})
    flags = injection_flags([str(clean.get("statement") or ""), str(clean.get("invalidation") or "")]
                            + [str(e.get("raw") or "") for e in (clean.get("evidence") or []) if isinstance(e, dict)])
    if not model_configured():
        return WorkflowResult(mode="needs_model_configuration", independent_llm_review="NOT_WIRED",
            prompt_hash=PROMPT_HASH, input_hash=ih, token_cost=None, status="needs_model_configuration",
            injection_flags=flags, message="Official model settings are incomplete; no model call was made.")
    provider = OfficialAPIProvider()
    try:
        researcher = provider.complete("researcher", deepcopy(clean))
        critic = provider.complete("critic", deepcopy(clean))
        synth_payload = deepcopy(clean)
        synth_payload["researcher"] = researcher.model_dump()
        synth_payload["critic"] = critic.model_dump()
        synthesizer = provider.complete("synthesizer", synth_payload)
    except Exception:
        return WorkflowResult(mode="model", independent_llm_review="NOT_WIRED", status="model_call_failed",
            prompt_hash=PROMPT_HASH, input_hash=ih, model_name=model_name(), model_provider=model_provider(),
            token_cost=None, usage_records=getattr(provider, "usage_records", []), usage_complete=False,
            injection_flags=flags, message="Model research failed; missing usage and cost remain unknown.")
    legal = set(legal_evidence_ids)
    blocked = validate_claims(researcher, legal) + validate_claims(critic, legal) + validate_claims(synthesizer, legal)
    usage = getattr(provider, "usage_records", [])
    complete = bool(usage) and all(r.get("input_tokens") is not None and r.get("output_tokens") is not None for r in usage)
    return WorkflowResult(mode="model", independent_llm_review="NOT_WIRED", status="completed",
        researcher=researcher, critic=critic, synthesizer=synthesizer,
        disagreements=list(critic.disagreements) + list(synthesizer.disagreements),
        blocked_claims=blocked, injection_flags=flags, prompt_hash=PROMPT_HASH, input_hash=ih,
        model_name=model_name(), model_provider=model_provider(), model_version=None, token_cost=None,
        usage_records=usage, usage_complete=complete,
        message="Researcher and critic used separate evidence-only requests to the same model; statistical independence is not established. Cost is unknown.")
