"""Operational research cycle: evidence -> isolated roles -> risk -> unapproved paper plan.

This is the file-bridge orchestration path. Official in-repo model APIs are not
used. External fresh-context role runs are recorded accurately and are not
claimed to be statistically independent or autonomously hosted.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from piia.intelligence.chronos_forecast import describe_capability, run_chronos_tiny
from piia.intelligence.hashing import content_hash
from piia.intelligence.ids import new_id
from piia.intelligence.etf_flows import fetch_us_spot_btc_etf_flows
from piia.intelligence.etf_tftc import fetch_tftc_etf
from piia.intelligence.etf_session_check import annotate_etf_session
from piia.intelligence.etf_xoomar import fetch_partial_etf
from piia.intelligence.event_calendar import fetch_official_event_calendar
from piia.intelligence.gold_cftc import fetch_comex_gold_positioning
from piia.intelligence.fear_greed import fetch_fear_greed
from piia.intelligence.forward_archive import append_current
from piia.intelligence.polymarket import fetch_prediction_markets
from piia.intelligence.evidence_gates import apply_external_missing, recommendation_blockers
from piia.intelligence.macro_data import EXTENDED_SERIES, SERIES, fetch_alfred_extended_context
from piia.intelligence.onchain_btc import fetch_btc_onchain
from piia.intelligence.paths import DEFAULT_SQLITE, ROOT
from piia.intelligence.public_data import fetch_public_snapshot
from piia.intelligence.research_packet import (
    Action,
    Direction,
    InstrumentResponseIn,
    PacketResponseIn,
    RoleIn,
    build_request,
    capture_paper_risk_context,
    import_response,
    restricted_content_hash,
    write_json,
)
from piia.intelligence.settings import model_status

CYCLE_PROTOCOL = "PIIA_OPERATIONAL_RESEARCH_CYCLE_V1"
RESTRICTED_PACKET = "piia_restricted_role_evidence_v1"
ROLE_REQUEST = "piia_isolated_role_request_v1"
ROLE_RESPONSE = "piia_isolated_role_response_v1"
INDEPENDENT_RUN_MODE = "external_orchestrated_fresh_context_not_statistically_independent"
INDEPENDENT_MARKER = "ISOLATED_CONTEXT_RUNS_SAME_OR_DIFFERENT_PROVIDER_NOT_STATISTICALLY_INDEPENDENT"
FORBIDDEN_DB = DEFAULT_SQLITE.resolve()
# Global prefixes were incorrectly blocking gold and Nasdaq-100 for BTC-only
# datasets, and treating single-stock fundamentals as a universal hard gate.
# Instrument-specific hard requirements live in evidence_gates.py.
CRITICAL_MISSING_PREFIXES = ()
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
Phase = Literal[
    "preparing",
    "awaiting_independent_roles",
    "awaiting_debate_and_synthesis",
    "imported_unapproved",
    "blocked",
]


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_cutoff(ts: str | datetime) -> datetime:
    t = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def default_cycle_dir() -> Path:
    return ROOT / "artifacts" / "research_cycle"


def official_api_status() -> dict[str, Any]:
    """Inspect only whether allowlisted provider *names* are present. Never read values."""
    names = [
        "PIIA_MODEL_API_KEY",
        "PIIA_MODEL_BASE_URL",
        "PIIA_MODEL_PROVIDER",
        "PIIA_MODEL_NAME",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "FRED_API_KEY",
    ]
    present = {name: name in os.environ and bool(os.environ.get(name)) for name in names}
    any_present = any(present.values())
    return {
        "official_in_repo_model": "not_configured" if not any_present else "env_names_present_not_used",
        "paid_endpoints_called": False,
        "env_names_present": present,
        "independent_llm_review": "NOT_WIRED",
        "model_status": model_status(),
        "note": (
            "No official in-repo model call is made in this cycle. "
            "External fresh-context orchestration is the chosen research path. "
            "Environment values are never logged."
        ),
    }


def _apply_macro_missing(missing: list[str], macro: dict[str, Any] | None) -> list[str]:
    out = list(missing)
    series = (macro or {}).get("series") or {}
    def _have(name: str) -> bool:
        item = series.get(name) or {}
        return bool(item) and not item.get("stale", True)

    replacements = []
    for name, current, added in (
        ("real_rates", _have("DFII10"), "real_rates_intraday_release_time"),
        ("rrp", _have("RRPONTSYD"), "rrp_intraday_release_time"),
        ("tga", _have("WDTGAL"), "tga_intraday_release_time_and_weekly_lag"),
    ):
        if name in out and current:
            out.remove(name)
            replacements.append(added)
    if "dxy" in out and _have("DTWEXBGS"):
        out.remove("dxy")
        replacements.append("ice_dxy")
        replacements.append("broad_dollar_intraday_release_time")
    if "macro" in out and series:
        out.remove("macro")
        replacements.append("macro_other_series_and_intraday_release_times")
    for item in replacements:
        if item not in out:
            out.append(item)
    return out


def _critical_blockers(
    request: dict[str, Any],
    forecast: dict[str, Any],
    macro: dict[str, Any] | None,
    onchain: dict[str, Any] | None = None,
    etf: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = recommendation_blockers(request, forecast, macro, onchain=onchain, etf=etf)
    etf_context = etf if etf is not None else request.get("etf_flow_context")
    if (isinstance(etf_context, dict) and etf_context.get("usable_for_btc_research") is True
            and etf_context.get("latest_exchange_session_verified") is not True):
        reason = "btc_etf_latest_exchange_session_unverified"
        result["instruments"].setdefault("BTCUSDT", []).append(reason)
        result["all"].append(reason)
        result["trading_recommendation_blocked"] = True
    return result


def _restricted_instrument(raw: dict[str, Any]) -> dict[str, Any]:
    candles = list(raw.get("all_closed_candles") or [])
    daily = list(raw.get("all_closed_daily_candles") or [])
    skip = {"all_closed_candles", "all_closed_daily_candles", "evidence_hash"}
    body = {k: v for k, v in raw.items() if k not in skip}
    body["candles_closed_1h_recent"] = candles[-72:]
    body["candles_closed_1d_recent"] = daily[-60:]
    body["recent_window_note"] = (
        "Restricted role files include recent closed bars plus hashes of the full "
        "frozen arrays. The complete arrays stay in the local evidence packet."
    )
    return body


def _build_restricted(request: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    instruments = {
        symbol: _restricted_instrument(raw)
        for symbol, raw in (request.get("instruments") or {}).items()
    }
    packet = {
        "packet_type": RESTRICTED_PACKET,
        "protocol": CYCLE_PROTOCOL,
        "request_hash": request["request_hash"],
        "cutoff_at_utc": request["cutoff_at_utc"],
        "exported_at_utc": request.get("exported_at_utc"),
        "snapshot_ok": request.get("snapshot_ok"),
        "missing_datasets": request.get("missing_datasets"),
        "errors": request.get("errors"),
        "macro_context": request.get("macro_context"),
        "paper_risk_context": request.get("paper_risk_context"),
        "onchain_context": request.get("onchain_context"),
        "etf_flow_context": request.get("etf_flow_context"),
        "event_calendar_context": request.get("event_calendar_context"),
        "gold_positioning_context": request.get("gold_positioning_context"),
        "prediction_market_context": request.get("prediction_market_context"),
        "fear_greed_context": request.get("fear_greed_context"),
        "liquidity_positioning_context": request.get("liquidity_positioning_context"),
        "instruments": instruments,
        "forecast": extras.get("forecast_public"),
        "forecast_available_no_earlier_than_utc": (extras.get("archive_receipt") or {}).get("recorded_at_utc"),
        "forecast_time_note": "Forecast output is available no earlier than the archive time; the market evidence cutoff is not the forecast completion time. A missing archive means this timing is unverified.",
        "capability": extras.get("capability"),
        "trading_recommendation_blocked": bool((extras.get("gate") or {}).get("trading_recommendation_blocked")),
        "trading_recommendation_blockers": (extras.get("gate") or {}).get("all") or [],
        "packet_recommendation_blockers": (extras.get("gate") or {}).get("packet") or [],
        "instrument_recommendation_blockers": (extras.get("gate") or {}).get("instruments") or {},
        "evidence_applicability": {
            "hard_requirements": (extras.get("gate") or {}).get("hard_requirements"),
            "research_gaps": (extras.get("gate") or {}).get("research_gaps"),
            "not_applicable": (extras.get("gate") or {}).get("not_applicable"),
        },
        "independent_llm_review": "NOT_WIRED",
        "independent_run_mode": INDEPENDENT_RUN_MODE,
        "independent_marker": INDEPENDENT_MARKER,
        "instructions": (
            "Use only this frozen evidence. Do not use later prices, do not assume "
            "a hosted official model reviewed this packet, and do not approve or "
            "place orders. Cite evidence hashes. Label missing data. Direction may "
            "be flat. Missing critical evidence blocks a trading recommendation."
        ),
    }
    packet["restricted_packet_hash"] = restricted_content_hash(packet)
    return packet


def _role_request(role: str, restricted: dict[str, Any], *, includes: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body = {
        "packet_type": ROLE_REQUEST,
        "protocol": CYCLE_PROTOCOL,
        "role": role,
        "allowed_context": includes,
        "run_id": None,  # filled by caller
        "request_hash": restricted["request_hash"],
        "restricted_packet_hash": restricted["restricted_packet_hash"],
        "cutoff_at_utc": restricted["cutoff_at_utc"],
        "independent_run_mode": INDEPENDENT_RUN_MODE,
        "independent_marker": INDEPENDENT_MARKER,
        "independent_llm_review": "NOT_WIRED",
        "evidence": restricted,
        "response_schema": role_response_schema(role),
        "role_instructions": role_instructions(role),
    }
    if extra:
        body.update(extra)
    return body


def role_instructions(role: str) -> str:
    common = (
        "只使用冻结证据，不要补充截止后的行情或新闻。"
        "不要声称这是官方独立 API 审查，也不要批准或下单。"
        "结论必须引用 evidence / request hash。"
        "关键证据缺失时，不得给出可执行的交易建议，应选择 stand_aside 或 need_more_evidence。"
        "保留不确定性。方向可以为 flat。"
    )
    specific = {
        "researcher": "你是研究方。只根据证据提出对 BTCUSDT、XAUUSDT、NDX100USDT 的看法，不得看到反方或综合意见。",
        "critic_independent": "你是反方第一轮。只根据同一份冻结证据独立审查，不得看到研究方意见。先找漏洞、替代解释和缺失数据。",
        "critic_debate": "你是反方第二轮（辩论）。可以阅读已标记的研究方意见，但必须把它与证据分开标注。不要把文件哈希当作独立性证明。",
        "synthesizer": "你是综合方。必须同时看到研究方与反方独立审查；若有辩论轮次也一并考虑。保留分歧，不要投票，不要绕过程序风控。关键证据缺失时不得建议 review_paper_plan。",
    }
    return specific[role] + common


def role_response_schema(role: str) -> dict[str, Any]:
    schema = {
        "packet_type": ROLE_RESPONSE,
        "role": role,
        "run_id": "copy from request",
        "request_hash": "copy from request",
        "restricted_packet_hash": "copy from request",
        "cutoff_at_utc": "copy from request",
        "actor": "external_fresh_context",
        "generation": "external_fresh_context",
        "independent_run_mode": INDEPENDENT_RUN_MODE,
        "notes": "optional",
        "instruments": {
            "SYMBOL": {
                "direction": "long | short | flat",
                "summary": "non-empty",
                "invalidation": "non-empty",
                "claims": [{"text": "", "evidence_hash": "", "status": "supported|unsupported|unverified_url|invalid_ref"}],
                "disagreements": [],
                "missing_data": [],
                "suggested_action": "review_paper_plan | stand_aside | need_more_evidence",
                "proposed_stop": "number or null; required only if review_paper_plan",
            }
        },
    }
    if role == "synthesizer":
        schema["followup_request_hash"] = "copy from synthesizer request"
    return schema


def _public_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky sample paths from the role-facing forecast, keep provenance and quantiles."""
    public = {k: v for k, v in payload.items() if k != "forecasts"}
    forecasts = {}
    for symbol, item in (payload.get("forecasts") or {}).items():
        if not item.get("ok"):
            forecasts[symbol] = item
            continue
        slim = dict(item)
        slim.pop("samples", None)
        slim["samples_omitted_from_restricted_file"] = True
        slim["sample_count"] = item.get("num_samples")
        forecasts[symbol] = slim
    public["forecasts"] = forecasts
    public["training_leakage"] = payload.get("training_leakage", "unknown")
    public["calibration_verified"] = False
    public["alpha_claim"] = False
    return public


def _state(path: Path, **fields: Any) -> dict[str, Any]:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(fields)
    current["updated_at_utc"] = _now().isoformat()
    write_json(path, current)
    return current


def prepare_run(
    out_dir: Path,
    *,
    allow_forecast_download: bool = False,
    snapshot: dict[str, Any] | None = None,
    macro: dict[str, Any] | None = None,
    forecast_payload: dict[str, Any] | None = None,
    fetch_public_feeds: bool | None = None,
) -> dict[str, Any]:
    """Fetch or accept frozen inputs, write the isolated role bundle, and stop for dispatch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    state_path = run_dir / "STATE.json"
    _state(state_path, protocol=CYCLE_PROTOCOL, run_id=run_id, phase="preparing",
           official_api=official_api_status())
    try:
        risk = capture_paper_risk_context()
        live_snapshot = snapshot is None
        live_archive_eligible = (snapshot is None and macro is None and forecast_payload is None
                                 and fetch_public_feeds is not False)
        if snapshot is None:
            snapshot = fetch_public_snapshot()
        if macro is None:
            macro = fetch_alfred_extended_context(snapshot_dir=run_dir / "macro" / "raw")
        if fetch_public_feeds is None:
            fetch_public_feeds = live_snapshot
        snapshot = dict(snapshot)
        snapshot["paper_risk_context"] = risk
        snapshot["macro_context"] = macro
        # Current contemporaneous research only. Never rewrite a prior frozen cycle.
        if snapshot.get("onchain_context") is not None:
            write_json(run_dir / "onchain" / "context.json", snapshot["onchain_context"])
        elif fetch_public_feeds:
            snapshot["onchain_context"] = fetch_btc_onchain(snapshot_dir=run_dir / "onchain")
        if snapshot.get("etf_flow_context") is not None:
            write_json(run_dir / "etf" / "context.json", snapshot["etf_flow_context"])
        elif fetch_public_feeds:
            snapshot["etf_flow_context"] = fetch_us_spot_btc_etf_flows(snapshot_dir=run_dir / "etf")
            if not snapshot["etf_flow_context"].get("usable_for_btc_research"):
                primary = snapshot["etf_flow_context"]
                fallback = fetch_tftc_etf(snapshot_dir=run_dir / "etf" / "fallback")
                fallback["primary_source_failure"] = primary.get("errors", {})
                snapshot["etf_flow_context"] = fallback
                write_json(run_dir / "etf" / "fallback_context.json", fallback)
        if isinstance(snapshot.get("etf_flow_context"), dict):
            snapshot["etf_flow_context"] = annotate_etf_session(snapshot["etf_flow_context"])
            if fetch_public_feeds:
                snapshot["etf_flow_context"]["supplemental_partial"] = fetch_partial_etf(snapshot_dir=run_dir / "etf" / "xoomar")
        if snapshot.get("event_calendar_context") is not None:
            write_json(run_dir / "calendar" / "context.json", snapshot["event_calendar_context"])
        elif fetch_public_feeds:
            snapshot["event_calendar_context"] = fetch_official_event_calendar(snapshot_dir=run_dir / "calendar")
        if snapshot.get("gold_positioning_context") is not None:
            write_json(run_dir / "gold" / "context.json", snapshot["gold_positioning_context"])
        elif fetch_public_feeds:
            snapshot["gold_positioning_context"] = fetch_comex_gold_positioning(snapshot_dir=run_dir / "gold")
        if snapshot.get("prediction_market_context") is not None:
            write_json(run_dir / "prediction_markets" / "context.json", snapshot["prediction_market_context"])
        elif fetch_public_feeds:
            snapshot["prediction_market_context"] = fetch_prediction_markets(snapshot_dir=run_dir / "prediction_markets")
        if snapshot.get("fear_greed_context") is None and fetch_public_feeds:
            snapshot["fear_greed_context"] = fetch_fear_greed(snapshot_dir=run_dir / "sentiment")
        if snapshot.get("fear_greed_context") is not None:
            write_json(run_dir / "sentiment" / "context.json", snapshot["fear_greed_context"])
        if snapshot.get("liquidity_positioning_context") is not None:
            write_json(run_dir / "liquidity" / "context.json", snapshot["liquidity_positioning_context"])
        else:
            from piia.intelligence.liquidity_positioning import build_liquidity_context
            positioning_feeds = snapshot.get("positioning_feeds")
            macro_history = snapshot.get("macro_history") or snapshot.get("alfred_history")
            execution_books = snapshot.get("execution_books") or snapshot.get("orderbook_context")
            if fetch_public_feeds:
                try:
                    from piia.intelligence.public_positioning import fetch_positioning_feeds
                    positioning_feeds = positioning_feeds or fetch_positioning_feeds(list((snapshot.get("instruments") or {}).keys()) or ["BTCUSDT", "XAUUSDT", "NDX100USDT"])
                except Exception as exc:  # live feeds must fail open into honest missing data
                    positioning_feeds = {"errors": {"fetch": type(exc).__name__}, "instruments": {}}
                if macro_history is None:
                    try:
                        from piia.intelligence.macro_data import fetch_alfred_history
                        macro_history = fetch_alfred_history(["DFII10", "DTWEXBGS"])
                    except Exception as exc:
                        macro_history = {"errors": {"fetch": type(exc).__name__}}
                if execution_books is None:
                    try:
                        from piia.intelligence.public_orderbook import fetch_execution_books
                        execution_books = fetch_execution_books()
                    except Exception as exc:
                        execution_books = {"ok": False, "errors": {"fetch": type(exc).__name__}, "instruments": {}}
            snapshot["liquidity_positioning_context"] = build_liquidity_context(
                snapshot, positioning_feeds=positioning_feeds, macro_history=macro_history,
                execution_books=execution_books, mode="current_snapshot", now=_now() if fetch_public_feeds else None,
            )
            if fetch_public_feeds:
                snapshot["liquidity_positioning_context"]["completed_at_utc"] = _now().isoformat()
            write_json(run_dir / "liquidity" / "context.json", snapshot["liquidity_positioning_context"])
        snapshot["missing_datasets"] = apply_external_missing(
            _apply_macro_missing(list(snapshot.get("missing_datasets") or []), macro),
            snapshot.get("onchain_context"),
            snapshot.get("etf_flow_context"),
            calendar=snapshot.get("event_calendar_context"),
            gold_positioning=snapshot.get("gold_positioning_context"),
            prediction_markets=snapshot.get("prediction_market_context"),
        )
        snapshot["missing_datasets"].extend(x for x in (
            "liquidity_positioning_states_uncalibrated_and_unvalidated",
            "bitget_liquidations_limited_window_single_venue",
        ) if x not in snapshot["missing_datasets"])
        if (snapshot.get("fear_greed_context") or {}).get("usable_for_research"):
            snapshot["missing_datasets"] = [x for x in snapshot["missing_datasets"] if x != "sentiment_indices"]
            snapshot["missing_datasets"].append("sentiment_other_indices")
        request = build_request(snapshot)

        series = {}
        for symbol, inst in (request.get("instruments") or {}).items():
            candles = inst.get("all_closed_candles") or []
            series[symbol] = {
                "closes": [float(c["close"]) for c in candles],
                "last_close_at_utc": inst.get("last_closed_close_at_utc"),
                "timeframe": "1H_closed",
            }
        forecast_path = run_dir / "forecast" / "chronos_t5_tiny.json"
        if forecast_payload is None:
            forecast_payload = run_chronos_tiny(
                series, forecast_path, allow_download=allow_forecast_download,
            )
        else:
            write_json(forecast_path, forecast_payload)
        forecast_public = _public_forecast(forecast_payload)
        gate = _critical_blockers(
            request, forecast_payload, macro,
            onchain=snapshot.get("onchain_context"),
            etf=snapshot.get("etf_flow_context"),
        )
        blockers = list(gate["all"])
        capability = {
            "official_api": official_api_status(),
            "chronos": describe_capability(),
            "forecast_ran": bool(forecast_payload.get("weights_ran")),
            "macro_kind": macro.get("kind"),
            "onchain_kind": (snapshot.get("onchain_context") or {}).get("kind"),
            "etf_flow_kind": (snapshot.get("etf_flow_context") or {}).get("kind"),
            "event_calendar_kind": (snapshot.get("event_calendar_context") or {}).get("kind"),
            "gold_positioning_kind": (snapshot.get("gold_positioning_context") or {}).get("kind"),
            "prediction_market_kind": (snapshot.get("prediction_market_context") or {}).get("kind"),
            "liquidity_positioning_state": (snapshot.get("liquidity_positioning_context") or {}).get("official_state"),
            "historical_replay_supported": False,
            "autonomous_hosted_model": False,
            "paper_execution": "unapproved_plans_only_after_import",
            "trading_recommendation_blocked": bool(gate["trading_recommendation_blocked"]),
            "trading_recommendation_blockers": blockers,
            "packet_recommendation_blockers": gate["packet"],
            "instrument_recommendation_blockers": gate["instruments"],
            "alpha_claim": False,
            "forecast_calibration_verified": False,
            "exact_liquidation_verified": False,
        }
        write_json(run_dir / "evidence" / "request.json", request)
        write_json(run_dir / "macro" / "alfred_extended.json", macro)
        write_json(run_dir / "capability.json", capability)

        # Live, on-demand prepare calls archive their exact current inputs and
        # forecast before roles see them. Injected fixtures are never archived.
        archive_receipt = None
        if live_archive_eligible:
            archive_receipt = append_current(run_id, request, forecast_payload)
            write_json(run_dir / "evidence" / "archive_receipt.json", archive_receipt)

        extras = {"forecast_public": forecast_public, "capability": capability, "blockers": blockers, "gate": gate,
                  "archive_receipt": archive_receipt}
        restricted = _build_restricted(request, extras)
        write_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json", restricted)

        roles_dir = run_dir / "dispatch"
        researcher = _role_request("researcher", restricted, includes=["frozen_evidence"])
        critic = _role_request("critic_independent", restricted, includes=["frozen_evidence"])
        for packet, name in ((researcher, "RESEARCHER_REQUEST.json"), (critic, "CRITIC_INDEPENDENT_REQUEST.json")):
            packet["run_id"] = run_id
            packet["request_file"] = str((roles_dir / name).name)
            write_json(roles_dir / name, packet)
        if content_hash(_canonical(_evidence_only(researcher))) != content_hash(_canonical(_evidence_only(critic))):
            raise RuntimeError("researcher and critic independent evidence contexts diverged")

        debate_template = _role_request(
            "critic_debate", restricted,
            includes=["frozen_evidence", "researcher_opinion_labeled"],
            extra={"status": "template_not_ready",
                   "note": "Do not fill this file yet. After the researcher response is imported, run followups to materialize the debate request with the labeled opinion."},
        )
        synth_template = _role_request(
            "synthesizer", restricted,
            includes=["frozen_evidence", "researcher_opinion_labeled", "critic_independent_opinion_labeled", "critic_debate_optional"],
            extra={"status": "template_not_ready",
                   "note": "Do not fill this file yet. Run followups after independent role responses exist."},
        )
        for packet, name in ((debate_template, "CRITIC_DEBATE_REQUEST.template.json"),
                             (synth_template, "SYNTHESIZER_REQUEST.template.json")):
            packet["run_id"] = run_id
            write_json(roles_dir / name, packet)

        inbox = run_dir / "inbox"
        inbox.mkdir(exist_ok=True)
        for name in ("researcher.json", "critic_independent.json", "critic_debate.json", "synthesizer.json"):
            if not (inbox / name).exists():
                (inbox / name).write_text("", encoding="utf-8")
        schema = {
            "protocol": CYCLE_PROTOCOL,
            "roles": {
                "researcher": "isolated evidence only",
                "critic_independent": "isolated evidence only; must not include researcher opinion",
                "critic_debate": "optional second pass; may see labeled researcher opinion",
                "synthesizer": "sees researcher and critic_independent, optional debate",
            },
            "response_packet_type": ROLE_RESPONSE,
            "generation": "external_fresh_context",
            "independent_run_mode": INDEPENDENT_RUN_MODE,
            "independent_marker": INDEPENDENT_MARKER,
            "hash_note": "File or process hashes do not prove statistical independence.",
            "approval_note": "Import never approves or executes a paper plan. Historical sandbox approvals are fixture-tagged only.",
        }
        write_json(roles_dir / "ROLE_SCHEMA.json", schema)
        (roles_dir / "INSTRUCTIONS.md").write_text(_instructions_markdown(run_id, run_dir, restricted, blockers), encoding="utf-8")

        state = _state(
            state_path,
            phase="awaiting_independent_roles",
            run_id=run_id,
            exploratory_evidence_archive=archive_receipt,
            request_hash=request["request_hash"],
            restricted_packet_hash=restricted["restricted_packet_hash"],
            cutoff_at_utc=request["cutoff_at_utc"],
            snapshot_ok=request.get("snapshot_ok"),
            forecast_weights_ran=bool(forecast_payload.get("weights_ran")),
            forecast_model_id=forecast_payload.get("model_id") if forecast_payload.get("weights_ran") else None,
            forecast_revision=forecast_payload.get("revision") if forecast_payload.get("weights_ran") else None,
            trading_recommendation_blocked=bool(gate["trading_recommendation_blocked"]),
            trading_recommendation_blockers=blockers,
            packet_recommendation_blockers=gate["packet"],
            instrument_recommendation_blockers=gate["instruments"],
            official_api=official_api_status(),
            paths={
                "run_dir": str(run_dir),
                "state": str(state_path),
                "full_request": str(run_dir / "evidence" / "request.json"),
                "restricted_evidence": str(roles_dir / "RESTRICTED_EVIDENCE.json"),
                "researcher_request": str(roles_dir / "RESEARCHER_REQUEST.json"),
                "critic_independent_request": str(roles_dir / "CRITIC_INDEPENDENT_REQUEST.json"),
                "critic_debate_template": str(roles_dir / "CRITIC_DEBATE_REQUEST.template.json"),
                "synthesizer_template": str(roles_dir / "SYNTHESIZER_REQUEST.template.json"),
                "role_schema": str(roles_dir / "ROLE_SCHEMA.json"),
                "instructions": str(roles_dir / "INSTRUCTIONS.md"),
                "inbox": str(inbox),
                "forecast": str(forecast_path),
                "macro": str(run_dir / "macro" / "alfred_extended.json"),
            },
            next_step=(
                "Dispatch RESEARCHER_REQUEST.json and CRITIC_INDEPENDENT_REQUEST.json "
                "in isolated fresh contexts. Do not share answers between them. "
                "Write responses into inbox/researcher.json and inbox/critic_independent.json, "
                "then run: python -m piia research-cycle followups --run <run_dir>"
            ),
        )
        return state
    except Exception as exc:
        _state(state_path, phase="blocked", error=type(exc).__name__, detail=str(exc))
        raise


def _evidence_only(role_request: dict[str, Any]) -> dict[str, Any]:
    keys = ("request_hash", "restricted_packet_hash", "cutoff_at_utc", "evidence")
    return {k: role_request[k] for k in keys}


def _instructions_markdown(run_id: str, run_dir: Path, restricted: dict[str, Any], blockers: list[str]) -> str:
    inst_blockers = restricted.get("instrument_recommendation_blockers") or {}
    inst_lines = "\n".join(
        f"- `{symbol}`: " + (", ".join(f"`{item}`" for item in reasons) if reasons else "no instrument-specific hard evidence block")
        for symbol, reasons in inst_blockers.items()
    ) or "- (none recorded)"
    blocker_lines = "\n".join(f"- `{item}`" for item in blockers) or "- (none)"
    return f"""# 独立角色请求（运行 {run_id}）

截止时间（UTC）：`{restricted['cutoff_at_utc']}`
request_hash：`{restricted['request_hash']}`
restricted_packet_hash：`{restricted['restricted_packet_hash']}`

## 这一步要做什么

1. 用 `RESEARCHER_REQUEST.json` 在一个全新上下文中做研究方，看不到反方。
2. 用 `CRITIC_INDEPENDENT_REQUEST.json` 在另一个全新上下文中做反方第一轮，看不到研究方。
3. 把回答分别写入：
   - `{run_dir / 'inbox' / 'researcher.json'}`
   - `{run_dir / 'inbox' / 'critic_independent.json'}`
4. 然后运行 `python -m piia research-cycle followups --run {run_dir}`，生成已标注研究意见的辩论请求和综合请求。
5. 综合方必须同时看到双方意见，并将综合请求里的 `followup_request_hash` 原样写入综合回答。研究方或反方意见变动后须重新生成请求和综合回答。全部角色完成后再运行 `python -m piia research-cycle import-roles --run {run_dir}`。

## 硬限制

- 这是外部编排的全新上下文研究，**不是**仓库内已接通的自主托管模型。
- 文件哈希或进程隔离不能证明统计独立。
- 导入后最多生成 **未批准** 的纸面计划；不会自动批准或下单。
- 历史 sandbox 批准只能是带标签的测试夹具。
- 可执行建议的硬性证据门槛按品种适用：BTC 需要完整的当前链上集合和美国现货比特币 ETF 资金流；黄金和纳指不会因为缺少 BTC 链上或 ETF 数据而自动失败。单只股票基本面不是黄金或指数的通用硬门槛。
- 汇总阻断：

{blocker_lines}

- 分品种硬门槛：

{inst_lines}

- Chronos-t5-tiny 预测若已实际运行，仍未校准，训练泄漏未知，不能当作已验证优势。
- 宏观是保守日期版本的官方 ALFRED 数据，不是盘中精确发布时间，也不能回填到历史时点。
- BTC 链上来自 mempool.space 公开 REST；ETF 资金流来自 Farside 对发行人披露的汇编，不是 SEC 官方日报，也不是 AUM 或价格。
- 精确强平模型仍未验证。事件日历与更深层链上/黄金库存/纳指官方估值宽度仍可能缺失。
"""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing role file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_role_response(payload: dict[str, Any], *, role: str, state: dict[str, Any]) -> dict[str, Any]:
    if payload.get("packet_type") != ROLE_RESPONSE:
        raise ValueError(f"{role} packet_type must be {ROLE_RESPONSE}")
    if payload.get("role") != role:
        raise ValueError(f"{role} role field mismatch")
    if payload.get("run_id") != state["run_id"]:
        raise ValueError(f"{role} run_id mismatch")
    if payload.get("request_hash") != state["request_hash"]:
        raise ValueError(f"{role} request_hash mismatch")
    if payload.get("restricted_packet_hash") != state["restricted_packet_hash"]:
        raise ValueError(f"{role} restricted_packet_hash mismatch")
    if payload.get("cutoff_at_utc") != state["cutoff_at_utc"]:
        raise ValueError(f"{role} cutoff mismatch")
    if payload.get("generation") != "external_fresh_context" or payload.get("independent_run_mode") != INDEPENDENT_RUN_MODE:
        raise ValueError(f"{role} must declare the external fresh-context workflow")
    if role == "synthesizer" and (not state.get("synthesizer_request_hash")
                                  or payload.get("followup_request_hash") != state["synthesizer_request_hash"]):
        raise ValueError("synthesizer response does not match the followup request hash")
    if role == "critic_independent":
        banned = ("researcher_opinion", "researcher", "researcher_response", "other_role_opinion")
        if any(k in payload for k in banned):
            raise ValueError("critic_independent response contains researcher fields")
        for inst in (payload.get("instruments") or {}).values():
            if isinstance(inst, dict) and any(k in inst for k in banned):
                raise ValueError("critic_independent instrument contains researcher fields")
    instruments = payload.get("instruments") or {}
    if set(instruments) != set(SYMBOLS):
        raise ValueError(f"{role} must cover {SYMBOLS}")
    return payload


def _verify_cycle_binding(state: dict[str, Any], restricted: dict[str, Any]) -> None:
    """Detect drift between the status file and the evidence shown to roles.

    Local hashes do not authenticate files against an administrator. These
    checks catch accidental changes and inconsistent local workflow state.
    """
    stored = restricted.get("restricted_packet_hash")
    if not stored or restricted_content_hash(restricted) != stored:
        raise ValueError("restricted evidence file hash mismatch")
    for state_key, evidence_key in (("restricted_packet_hash", "restricted_packet_hash"),
                                    ("request_hash", "request_hash"), ("cutoff_at_utc", "cutoff_at_utc"),
                                    ("packet_recommendation_blockers", "packet_recommendation_blockers"),
                                    ("instrument_recommendation_blockers", "instrument_recommendation_blockers"),
                                    ("trading_recommendation_blocked", "trading_recommendation_blocked")):
        if state_key not in state or evidence_key not in restricted or state[state_key] != restricted[evidence_key]:
            raise ValueError(f"cycle state and restricted evidence differ: {state_key}")


def _verify_synthesis_request(run_dir: Path, state: dict[str, Any], restricted: dict[str, Any],
                              researcher: dict[str, Any], critic: dict[str, Any]) -> None:
    synth_request = _load_json(run_dir / "dispatch" / "SYNTHESIZER_REQUEST.json")
    if (synth_request.get("run_id") != state["run_id"]
            or synth_request.get("restricted_packet_hash") != state["restricted_packet_hash"]
            or synth_request.get("followup_request_hash") != state["synthesizer_request_hash"]
            or content_hash(_canonical({k: v for k, v in synth_request.items() if k != "followup_request_hash"})) != state["synthesizer_request_hash"]
            or synth_request.get("researcher_opinion_labeled", {}).get("opinion") != researcher
            or synth_request.get("critic_independent_opinion_labeled", {}).get("opinion") != critic
            or synth_request.get("evidence") != restricted):
        raise ValueError("synthesizer request no longer matches the frozen evidence and source roles")


def _to_role_in(role_name: str, item: dict[str, Any]) -> RoleIn:
    action: Action = item.get("suggested_action") or "need_more_evidence"
    direction: Direction | None = item.get("direction")
    return RoleIn(
        role="critic" if role_name.startswith("critic") else role_name,  # type: ignore[arg-type]
        actor=str(item.get("actor") or "external_fresh_context"),
        generation="external_fresh_context",
        direction=direction,
        summary=item["summary"],
        invalidation=item["invalidation"],
        claims=item.get("claims") or [],
        disagreements=item.get("disagreements") or [],
        missing_data=item.get("missing_data") or [],
        suggested_action=action,
        proposed_stop=item.get("proposed_stop"),
    )


def materialize_followups(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = _load_json(run_dir / "STATE.json")
    if state.get("phase") not in {"awaiting_independent_roles", "awaiting_debate_and_synthesis"}:
        raise ValueError("research cycle is not awaiting role followups")
    restricted = _load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json")
    _verify_cycle_binding(state, restricted)
    researcher = _validate_role_response(_load_json(run_dir / "inbox" / "researcher.json"), role="researcher", state=state)
    critic = _validate_role_response(_load_json(run_dir / "inbox" / "critic_independent.json"), role="critic_independent", state=state)
    labeled_researcher = {"label": "researcher_opinion_labeled", "not_evidence": True, "opinion": researcher}
    labeled_critic = {"label": "critic_independent_opinion_labeled", "not_evidence": True, "opinion": critic}
    debate = _role_request(
        "critic_debate", restricted,
        includes=["frozen_evidence", "researcher_opinion_labeled"],
        extra={"researcher_opinion_labeled": labeled_researcher,
               "status": "ready",
               "note": "The researcher opinion is labeled and is not itself frozen market evidence."},
    )
    synth = _role_request(
        "synthesizer", restricted,
        includes=["frozen_evidence", "researcher_opinion_labeled", "critic_independent_opinion_labeled", "critic_debate_optional"],
        extra={"researcher_opinion_labeled": labeled_researcher,
               "critic_independent_opinion_labeled": labeled_critic,
               "critic_debate_path": str(run_dir / "inbox" / "critic_debate.json"),
               "status": "ready",
               "note": "Synthesizer must retain disagreements. Debate file is optional."},
    )
    for packet, name in ((debate, "CRITIC_DEBATE_REQUEST.json"), (synth, "SYNTHESIZER_REQUEST.json")):
        packet["run_id"] = state["run_id"]
        if name == "SYNTHESIZER_REQUEST.json":
            packet["followup_request_hash"] = content_hash(_canonical(packet))
        write_json(run_dir / "dispatch" / name, packet)
    return _state(
        run_dir / "STATE.json",
        phase="awaiting_debate_and_synthesis",
        synthesizer_request_hash=synth["followup_request_hash"],
        paths={**state.get("paths", {}),
               "critic_debate_request": str(run_dir / "dispatch" / "CRITIC_DEBATE_REQUEST.json"),
               "synthesizer_request": str(run_dir / "dispatch" / "SYNTHESIZER_REQUEST.json")},
        next_step=(
            "Optional: fill inbox/critic_debate.json from CRITIC_DEBATE_REQUEST.json. "
            "Required: fill inbox/synthesizer.json from SYNTHESIZER_REQUEST.json, then "
            "run python -m piia research-cycle import-roles --run <run_dir>"
        ),
    )


def import_roles(run_dir: Path, *, db_path: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = _load_json(run_dir / "STATE.json")
    if state.get("phase") != "awaiting_debate_and_synthesis":
        raise ValueError("research cycle is not awaiting synthesis import")
    request = _load_json(run_dir / "evidence" / "request.json")
    restricted = _load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json")
    _verify_cycle_binding(state, restricted)
    researcher = _validate_role_response(_load_json(run_dir / "inbox" / "researcher.json"), role="researcher", state=state)
    critic = _validate_role_response(_load_json(run_dir / "inbox" / "critic_independent.json"), role="critic_independent", state=state)
    synth = _validate_role_response(_load_json(run_dir / "inbox" / "synthesizer.json"), role="synthesizer", state=state)
    _verify_synthesis_request(run_dir, state, restricted, researcher, critic)
    debate_path = run_dir / "inbox" / "critic_debate.json"
    debate = None
    if debate_path.is_file() and debate_path.stat().st_size > 0:
        debate = _validate_role_response(_load_json(debate_path), role="critic_debate", state=state)

    packet_blockers = list(state.get("packet_recommendation_blockers") or [])
    instrument_blockers = dict(state.get("instrument_recommendation_blockers") or {})
    instruments: list[InstrumentResponseIn] = []
    downgrades: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        syn_item = dict(synth["instruments"][symbol])
        reasons = list(packet_blockers) + list(instrument_blockers.get(symbol) or [])
        if reasons and syn_item.get("suggested_action") == "review_paper_plan":
            downgrades.append({
                "symbol": symbol,
                "from": "review_paper_plan",
                "to": "need_more_evidence",
                "reasons": reasons,
            })
            syn_item["suggested_action"] = "need_more_evidence"
            notes = list(syn_item.get("missing_data") or [])
            notes.append("cycle_blocked_trading_recommendation")
            syn_item["missing_data"] = notes
        instruments.append(InstrumentResponseIn(
            symbol=symbol,
            request_evidence_hash=(request["instruments"][symbol]["evidence_hash"]),
            researcher=_to_role_in("researcher", researcher["instruments"][symbol]),
            critic=_to_role_in("critic", critic["instruments"][symbol]),
            synthesizer=_to_role_in("synthesizer", syn_item),
        ))
    response = PacketResponseIn(
        packet_type="piia_chat_assisted_research_response",
        request_hash=request["request_hash"],
        cutoff_at_utc=request["cutoff_at_utc"],
        independent_llm_review="NOT_WIRED",
        actor="invest-desk_external_fresh_context",
        notes=(
            f"mode={INDEPENDENT_RUN_MODE}; marker={INDEPENDENT_MARKER}; "
            f"debate_present={debate is not None}; packet_blockers={packet_blockers}; instrument_blockers={instrument_blockers}"
        ),
        instruments=instruments,
    )
    db = Path(db_path) if db_path else run_dir / "db" / "research_cycle.sqlite"
    if db.resolve() == FORBIDDEN_DB:
        raise ValueError("refusing to import into the live research database")
    db.parent.mkdir(parents=True, exist_ok=True)
    if restricted.get("request_hash") != request.get("request_hash") or restricted.get("cutoff_at_utc") != request.get("cutoff_at_utc"):
        raise ValueError("restricted evidence does not match the frozen request")
    imported_at = _now()
    result = import_response(request, response, db_path=db,
                             evidence_store=run_dir / "db" / "evidence",
                             restricted_packet=restricted)
    for inst in result.get("instruments") or []:
        plan = inst.get("plan") or {}
        if plan and (plan.get("approved") or plan.get("executed") or plan.get("actionable") is True):
            raise RuntimeError("import produced an approved or actionable plan; refusing to continue")
        inst["approved"] = False
        inst["executed"] = False
    cutoff = _aware_cutoff(request["cutoff_at_utc"])
    quote_ages = {}
    for symbol, raw in (request.get("instruments") or {}).items():
        ticker_at = (raw.get("ticker") or {}).get("observed_at_utc")
        quote_ages[symbol] = {
            "ticker_observed_at_utc": ticker_at,
            "ticker_age_seconds": None if not ticker_at else (imported_at - _aware_cutoff(ticker_at)).total_seconds(),
            "last_closed_close_at_utc": raw.get("last_closed_close_at_utc"),
            "quote_treated_as_current": False,
        }
    write_json(run_dir / "import" / "result.json", {
        "import": result,
        "recommendation_downgrades": downgrades,
        "trading_recommendation_blocked": bool(packet_blockers) or any(instrument_blockers.values()),
        "trading_recommendation_blockers": list(packet_blockers) + [
            reason for symbol in SYMBOLS for reason in (instrument_blockers.get(symbol) or [])
        ],
        "packet_recommendation_blockers": packet_blockers,
        "instrument_recommendation_blockers": instrument_blockers,
        "debate_present": debate is not None,
        "approved": False,
        "executed": False,
        "evidence_refreshed": False,
        "quote_treated_as_current": False,
        "imported_at_utc": imported_at.isoformat(),
        "cutoff_at_utc": request["cutoff_at_utc"],
        "cutoff_age_seconds": (imported_at - cutoff).total_seconds(),
        "quote_ages": quote_ages,
        "plans_created": [inst["symbol"] for inst in result.get("instruments") or [] if inst.get("plan")],
    })
    return _state(
        run_dir / "STATE.json",
        phase="imported_unapproved",
        import_result=str(run_dir / "import" / "result.json"),
        import_result_sha256=content_hash((run_dir / "import" / "result.json").read_bytes()),
        imported_request_sha256=content_hash((run_dir / "evidence" / "request.json").read_bytes()),
        db=str(db),
        recommendation_downgrades=downgrades,
        evidence_refreshed=False,
        quote_treated_as_current=False,
        plans_created=[inst["symbol"] for inst in result.get("instruments") or [] if inst.get("plan")],
        next_step="No executable recommendation was produced. Do not approve or trade without an explicit later owner decision and fresh evidence.",
    )


def status(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = _load_json(run_dir / "STATE.json")
    # Status may attach execution-context summaries; it never rewrites STATE or evidence.
    from piia.intelligence.execution_revalidate import list_revalidations, snapshot_frozen_files
    attached = dict(state)
    attached["execution_revalidations"] = list_revalidations(run_dir)
    attached["frozen_file_sha256"] = snapshot_frozen_files(run_dir)
    attached["status_does_not_rewrite_frozen_files"] = True
    return attached


def revalidate_execution(run_dir: Path, **kwargs) -> dict[str, Any]:
    """Fresh public quote/book check against immutable frozen research."""
    from piia.intelligence.execution_revalidate import revalidate_run
    return revalidate_run(Path(run_dir), **kwargs)
