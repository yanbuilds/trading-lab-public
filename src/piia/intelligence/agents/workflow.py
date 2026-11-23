from __future__ import annotations
from typing import Any, TypedDict
try:
    from langgraph.graph import END, START, StateGraph
    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover - fallback when langgraph is not installed
    END = START = StateGraph = None  # type: ignore[assignment]
    _HAS_LANGGRAPH = False
from piia.intelligence.agents.io import RoleOutput, WorkflowResult
from piia.intelligence.agents.providers import (
    PROMPT_HASH, FixtureProvider, OfficialAPIProvider, injection_flags,
)
from piia.intelligence.hashing import content_hash
from piia.intelligence.settings import model_configured, model_name, model_provider, model_status

class GraphState(TypedDict, total=False):
    payload: dict
    researcher: dict
    critic: dict
    synthesizer: dict

def _provider():
    return FixtureProvider()

def _node(role: str):
    def run(state: GraphState) -> GraphState:
        out = _provider().complete(role, state["payload"])  # type: ignore[arg-type]
        return {role: out.model_dump()}
    return run

class _SequentialGraph:
    def __init__(self, steps):
        self._steps = steps

    def invoke(self, state: GraphState) -> GraphState:
        current: dict = dict(state)
        for step in self._steps:
            current.update(step(current))
        return current  # type: ignore[return-value]


def build_graph():
    if not _HAS_LANGGRAPH:
        return _SequentialGraph((_node("researcher"), _node("critic"), _node("synthesizer")))
    g = StateGraph(GraphState)
    g.add_node("researcher", _node("researcher"))
    g.add_node("critic", _node("critic"))
    g.add_node("synthesizer", _node("synthesizer"))
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "critic")
    g.add_edge("critic", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()

_GRAPH = None

def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH

def input_hash(payload: dict) -> str:
    return content_hash(repr(sorted((k, repr(payload[k])) for k in payload)).encode())

def validate_claims(role_out: RoleOutput, legal_ids: set[str]) -> list[dict[str, Any]]:
    blocked = []
    for claim in role_out.claims:
        if not claim.evidence_ids:
            claim.status = "unsupported"
            blocked.append({"role": role_out.role, "text": claim.text, "reason": "unsupported_no_evidence"})
            continue
        bad = [i for i in claim.evidence_ids if i not in legal_ids]
        if bad:
            claim.status = "invalid_ref"
            blocked.append({"role": role_out.role, "text": claim.text, "reason": "evidence_id_not_on_revision", "ids": bad})
            claim.evidence_ids = [i for i in claim.evidence_ids if i in legal_ids]
        elif claim.status == "supported":
            # Existence of a reference does not establish that it entails the claim.
            claim.status = "unverified"
            blocked.append({"role": role_out.role, "text": claim.text, "reason": "evidence_link_not_verified"})
    return blocked

def run_workflow(*, mode: str, payload: dict, legal_evidence_ids: list[str]) -> WorkflowResult:
    """Run researcher → critic → synthesizer. Fixture mode actually uses LangGraph.

    Requesting model mode without project credentials returns needs_model_configuration
    and does not invent an independent LLM review.
    """
    texts = [payload.get("statement") or "", payload.get("invalidation") or ""]
    texts += [e.get("raw") or "" for e in (payload.get("evidence") or [])]
    flags = injection_flags(texts)
    legal = set(legal_evidence_ids)
    ih = input_hash({"mode": mode, "payload": payload, "prompt_hash": PROMPT_HASH})
    if mode == "model":
        if not model_configured():
            return WorkflowResult(
                mode="needs_model_configuration",
                independent_llm_review="NOT_WIRED",
                prompt_hash=PROMPT_HASH, input_hash=ih,
                token_cost=None, status="needs_model_configuration",
                message="Set PIIA_MODEL_PROVIDER, PIIA_MODEL_BASE_URL (allowlisted), PIIA_MODEL_NAME and PIIA_MODEL_API_KEY to enable official API research. Fixture mode remains available.",
                injection_flags=flags,
            )
        # Real model path is implemented but not invoked from tests without credentials.
        provider = OfficialAPIProvider()
        try:
            researcher = provider.complete("researcher", payload)
            critic_payload = {**payload, "researcher": researcher.model_dump()}
            critic = provider.complete("critic", critic_payload)
            synth_payload = {**payload, "researcher": researcher.model_dump(), "critic": critic.model_dump()}
            synthesizer = provider.complete("synthesizer", synth_payload)
        except Exception:
            usage = getattr(provider, "usage_records", [])
            return WorkflowResult(
                mode="model", independent_llm_review="NOT_WIRED", status="model_call_failed",
                prompt_hash=PROMPT_HASH, input_hash=ih, model_name=model_name(), model_provider=model_provider(),
                token_cost=None, usage_records=usage, usage_complete=False,
                injection_flags=flags,
                message="Model research failed; partial usage is shown where returned, and other usage and monetary cost are unknown.",
            )
        blocked = validate_claims(researcher, legal) + validate_claims(critic, legal) + validate_claims(synthesizer, legal)
        disagreements = list(critic.disagreements) + list(synthesizer.disagreements)
        usage = getattr(provider, "usage_records", [])
        complete_usage = bool(usage) and all(r.get("input_tokens") is not None and r.get("output_tokens") is not None for r in usage)
        return WorkflowResult(
            mode="model", independent_llm_review="NOT_WIRED",
            researcher=researcher, critic=critic, synthesizer=synthesizer,
            disagreements=disagreements, blocked_claims=blocked, injection_flags=flags,
            prompt_hash=PROMPT_HASH, input_hash=ih,
            model_name=model_name(), model_provider=model_provider(), model_version=None, token_cost=None,
            usage_records=usage, usage_complete=complete_usage,
            status="completed", message="Official API research completed using one model for all roles; independent review is not wired. Monetary cost is unknown.",
        )
    # Fixture: actually run LangGraph nodes.
    result = graph().invoke({"payload": payload})
    researcher = RoleOutput.model_validate(result["researcher"])
    critic = RoleOutput.model_validate(result["critic"])
    synthesizer = RoleOutput.model_validate(result["synthesizer"])
    blocked = validate_claims(researcher, legal) + validate_claims(critic, legal) + validate_claims(synthesizer, legal)
    disagreements = list(dict.fromkeys(critic.disagreements + synthesizer.disagreements))
    return WorkflowResult(
        mode="fixture", independent_llm_review="NOT_WIRED",
        researcher=researcher, critic=critic, synthesizer=synthesizer,
        disagreements=disagreements, blocked_claims=blocked, injection_flags=flags,
        prompt_hash=PROMPT_HASH, input_hash=ih,
        model_name="piia-deterministic-fixture", model_version="v1",
        token_cost=None, status="completed",
        message="Labelled deterministic fixture run via LangGraph. Not independent LLM research and not a forecast.",
    )
