"""Deterministic, non-executing evidence gate for conversational research."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from piia.intelligence.public_data import candle_quality
from piia.intelligence.research_cycle import _verify_cycle_binding


def _aware(text):
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)


def assess_chat_evidence(request: dict[str, Any], symbol: str, *, now: datetime,
                         state: dict[str, Any] | None, restricted: dict[str, Any] | None,
                         archive: dict[str, Any] | None, current_risk_hash: str | None,
                         forecast_bound: bool) -> dict[str, Any]:
    """Check current evidence only. A pass is not profitability or trade approval."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    now = now.astimezone(timezone.utc)
    reasons = []
    if request.get("snapshot_ok") is not True:
        reasons.append("snapshot_incomplete")
    try:
        if not timedelta(0) <= now - _aware(request["cutoff_at_utc"]) <= timedelta(minutes=5):
            reasons.append("snapshot_missing_or_stale")
    except (KeyError, TypeError, ValueError, AttributeError):
        reasons.append("snapshot_missing_or_stale")
    instruments = request.get("instruments") or {}
    if not isinstance(instruments, dict):
        instruments = {}
    if symbol not in instruments:
        reasons.append("focus_symbol_unavailable")
    etf = request.get("etf_flow_context")
    if symbol == "BTCUSDT" and isinstance(etf, dict) and etf.get("usable_for_btc_research") is True:
        from piia.intelligence.etf_session_check import annotate_etf_session
        if (etf.get("latest_exchange_session_verified") is not True
                or annotate_etf_session(etf, as_of=now).get("latest_exchange_session_verified") is not True):
            reasons.append("btc_etf_latest_exchange_session_unverified")
    onchain = request.get("onchain_context")
    if symbol == "BTCUSDT" and isinstance(onchain, dict) and onchain.get("usable_for_btc_research") is True:
        try:
            from piia.intelligence.onchain_btc import BLOCK_STALE_SECONDS
            started = _aware(onchain["retrieved_at_utc"])
            block_at = _aware(onchain["metrics"]["last_block"]["observation_at_utc"])
            if not timedelta(0) <= now - started <= timedelta(minutes=5):
                raise ValueError("on-chain snapshot stale")
            if not timedelta(0) <= now - block_at <= timedelta(seconds=BLOCK_STALE_SECONDS):
                raise ValueError("chain tip stale or future")
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            reasons.append("btc_onchain_snapshot_or_tip_stale")
    for name, item in instruments.items():
        if not isinstance(item, dict):
            reasons.append(f"instrument_invalid:{name}")
            continue
        try:
            observed = _aware(item["ticker"]["observed_at_utc"])
            if not timedelta(0) <= now - observed <= timedelta(minutes=5):
                raise ValueError("stale quote")
        except (KeyError, TypeError, ValueError, AttributeError):
            reasons.append(f"ticker_missing_or_stale:{name}")
        try:
            if candle_quality(item.get("all_closed_candles") or [], now):
                reasons.append(f"closed_candles_missing_or_stale:{name}")
        except (KeyError, TypeError, ValueError, AttributeError):
            reasons.append(f"closed_candles_missing_or_stale:{name}")
    risk = request.get("paper_risk_context") or {}
    if not risk.get("available") or not current_risk_hash or risk.get("owner_config_hash") != current_risk_hash:
        reasons.append("paper_risk_changed_or_unavailable")
    if not isinstance(state, dict) or not isinstance(restricted, dict) or not isinstance(archive, dict):
        reasons.append("full_research_evidence_unavailable")
    else:
        try:
            _verify_cycle_binding(state, restricted)
            if (state.get("request_hash") != request.get("request_hash") or archive.get("request") != request
                    or archive.get("run_id") != state.get("run_id")
                    or archive.get("order_permissions") is not False):
                raise ValueError("snapshot binding mismatch")
            archived_at = _aware(archive["created_at_utc"])
            cutoff_at = _aware(request["cutoff_at_utc"])
            if not cutoff_at <= archived_at <= now or archived_at - cutoff_at > timedelta(minutes=5):
                raise ValueError("archive timing invalid")
            for item in instruments.values():
                if _aware(item["ticker"]["observed_at_utc"]) > archived_at:
                    raise ValueError("quote retrieved after archive")
            packet = state["packet_recommendation_blockers"]
            per_symbol = state["instrument_recommendation_blockers"]
            if not isinstance(packet, list) or not isinstance(per_symbol, dict) or not isinstance(per_symbol.get(symbol), list):
                raise ValueError("blocker fields missing")
            reasons.extend(f"research_blocker:{item}" for item in packet + per_symbol[symbol])
        except (KeyError, TypeError, ValueError, AttributeError):
            reasons.append("research_evidence_binding_invalid")
        if state.get("forecast_weights_ran") and not forecast_bound:
            reasons.append("chronos_forecast_unbound")
        if symbol == "BTCUSDT" and isinstance(onchain, dict) and onchain.get("usable_for_btc_research") is True:
            try:
                from piia.intelligence.onchain_integrity import verify_onchain_snapshot
                verify_onchain_snapshot(onchain, Path(state["paths"]["run_dir"]) / "onchain")
            except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, UnicodeError):
                reasons.append("btc_onchain_raw_snapshot_invalid")
        gold = request.get("gold_positioning_context")
        sentiment = request.get("fear_greed_context")
        if symbol == "BTCUSDT" and isinstance(sentiment, dict) and sentiment.get("usable_for_research") is True:
            try:
                from piia.intelligence.fear_greed import verify_fear_greed_snapshot
                verify_fear_greed_snapshot(sentiment, Path(state["paths"]["run_dir"]) / "sentiment", now)
            except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, UnicodeError):
                reasons.append("btc_sentiment_raw_snapshot_invalid_or_stale")
        if symbol == "XAUUSDT" and isinstance(gold, dict) and gold.get("usable_for_gold_research") is True:
            try:
                from piia.intelligence.gold_cftc import STALE_DAYS, verify_gold_snapshot
                verify_gold_snapshot(gold, Path(state["paths"]["run_dir"]) / "gold")
                asof = datetime.fromisoformat(gold["asof_date"]).date()
                if not 0 <= (now.date() - asof).days <= STALE_DAYS:
                    raise ValueError("gold report stale")
            except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, UnicodeError):
                reasons.append("gold_cot_snapshot_invalid_or_stale")
        for key in ("onchain_context", "etf_flow_context", "event_calendar_context", "gold_positioning_context",
                    "prediction_market_context", "fear_greed_context"):
            context = request.get(key)
            if isinstance(context, dict) and (context.get("usable_for_research") is True
                                              or context.get("usable_for_btc_research") is True
                                              or context.get("usable_for_gold_research") is True):
                try:
                    started = _aware(context["retrieved_at_utc"])
                    completed = _aware(context["completed_at_utc"])
                    if not started <= completed <= _aware(request["cutoff_at_utc"]) <= _aware(archive["created_at_utc"]):
                        raise ValueError("external evidence completion time invalid")
                except (KeyError, TypeError, ValueError, AttributeError):
                    reasons.append(f"external_evidence_time_invalid:{key}")
        liquidity = request.get("liquidity_positioning_context")
        if liquidity is not None:
            try:
                if not (_aware(liquidity["retrieved_at_utc"]) <= _aware(liquidity["inputs_collected_at_utc"])
                        <= _aware(liquidity["cutoff_at_utc"]) <= _aware(liquidity["completed_at_utc"])
                        <= _aware(request["cutoff_at_utc"]) <= _aware(archive["created_at_utc"])):
                    raise ValueError("liquidity evidence timing invalid")
            except (KeyError, TypeError, ValueError, AttributeError):
                reasons.append("liquidity_evidence_time_unverified")
        etf_context = request.get("etf_flow_context")
        partial = etf_context.get("supplemental_partial") if isinstance(etf_context, dict) else None
        if isinstance(partial, dict) and partial.get("available") is True:
            try:
                if not (_aware(partial["retrieved_at_utc"]) <= _aware(partial["completed_at_utc"])
                        <= _aware(request["cutoff_at_utc"]) <= _aware(archive["created_at_utc"])):
                    raise ValueError("partial evidence completion time invalid")
            except (KeyError, TypeError, ValueError, AttributeError):
                reasons.append("external_evidence_time_invalid:etf_supplemental_partial")
            try:
                from piia.intelligence.etf_xoomar import verify_partial_etf
                verify_partial_etf(partial, Path(state["paths"]["run_dir"]) / "etf" / "xoomar")
            except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError):
                reasons.append("external_evidence_raw_invalid:etf_supplemental_partial")
        macro = request.get("macro_context")
        if macro is not None:
            try:
                from piia.intelligence.macro_data import verify_vintage_snapshot
                series = macro["series"]
                if not isinstance(series, dict) or not series:
                    raise ValueError("macro series missing")
                raw_dir = Path(state["paths"]["run_dir"]) / "macro" / "raw"
                for name, item in series.items():
                    if not isinstance(item, dict) or item.get("series_id") != name:
                        raise ValueError("macro series binding mismatch")
                    if _aware(item["retrieved_at_utc"]) > _aware(request["cutoff_at_utc"]):
                        raise ValueError("macro retrieved after cutoff")
                    verify_vintage_snapshot(item, raw_dir)
                    observation_date = datetime.fromisoformat(item["observation_date"]).date()
                    if not 0 <= (now.date() - observation_date).days <= 14:
                        raise ValueError("macro observation stale at review time")
            except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, UnicodeError):
                reasons.append("macro_raw_snapshot_unavailable_or_invalid")
    return {"review_blocked": bool(reasons), "blockers": list(dict.fromkeys(reasons)),
            "checked_at_utc": now.isoformat(), "symbol": symbol,
            "approved": False, "executed": False, "order_permissions": False,
            "strategy_edge_validated": False, "exact_liquidation_verified": False,
            "note": "Evidence checks do not establish profitability, actual account position, safe leverage, trade authorization, or independently verified receipt times."}


def apply_chat_gate(analysis: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    """Block reviews when evidence or the returned model claims are unverified."""
    out = deepcopy(analysis)
    gate = deepcopy(gate)
    blockers = list(gate.get("blockers") or [])
    for field in ("blocked_claims", "injection_flags"):
        value = out.get(field)
        if not isinstance(value, list):
            blockers.append(f"model_{field}_missing_or_invalid")
        elif value:
            blockers.append(f"model_{field}_present")
    for role in ("researcher", "critic", "synthesizer"):
        role_output = out.get(role)
        claims = role_output.get("claims") if isinstance(role_output, dict) else None
        if not isinstance(claims, list) or not claims:
            blockers.append(f"model_claims_missing:{role}")
        elif any(not isinstance(claim, dict) or claim.get("status") != "supported" for claim in claims):
            blockers.append(f"model_claims_not_verified:{role}")
    gate["blockers"] = list(dict.fromkeys(blockers))
    gate["review_blocked"] = bool(gate.get("review_blocked") or blockers)
    synth = out.get("synthesizer")
    if isinstance(synth, dict) and synth.get("suggested_action") == "review_paper_plan" and gate.get("review_blocked"):
        synth["suggested_action"] = "need_more_evidence"
        notes = list(synth.get("data_quality_notes") or [])
        notes.append("Programmatic evidence gate blocked paper-plan review; see deterministic_review_gate.")
        synth["data_quality_notes"] = notes
        out["model_original_suggested_action"] = "review_paper_plan"
    out["deterministic_review_gate"] = gate
    return out
