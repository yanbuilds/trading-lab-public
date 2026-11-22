"""Chat-assisted research request/import packets for PIIA.

The official model API is not required. A human or the chat assistant fills a
structured response against a frozen evidence request. This is not autonomous
multi-agent research or independent LLM review. Configuring a model does not
change the provenance of a manually imported response.

Import never approves or executes paper or live orders. Deterministic risk
remains separate from the writeups.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from piia.intelligence.clock import Clock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.errors import ValidationFailed
from piia.intelligence.hashing import HASH_RE, content_hash
from piia.intelligence.ids import new_id
from piia.intelligence.models import ResearchRun
from piia.intelligence.paper.replay import atr, ema
from piia.intelligence.paper.specs import CONTRACTS, PROVENANCE
from piia.intelligence.paths import DEFAULT_SQLITE, ROOT
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.schemas import AnalysisIn, CounterIn, EvidenceIn, ThesisCreate
from piia.intelligence.service import IntelligenceService
from piia.intelligence.settings import model_configured, model_status

ACTOR = "invest-desk_chat_assisted"
PACKET_REQUEST = "piia_chat_assisted_research_request"
PACKET_RESPONSE = "piia_chat_assisted_research_response"
LABEL = "CHAT_ASSISTED_RESEARCH"
FORBIDDEN_DB = DEFAULT_SQLITE.resolve()
Direction = Literal["long", "short", "flat"]
Action = Literal["review_paper_plan", "stand_aside", "need_more_evidence"]


def _aware(ts: str | datetime) -> datetime:
    t = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def capture_paper_risk_context() -> dict[str, Any]:
    """Freeze owner-configured *example* risk and local contract assumptions."""
    from piia.intelligence.paper.risk_file import load_owner_risk
    try:
        cfg, cfg_hash = load_owner_risk()
        fields = ("virtual_equity_usdt", "risk_fraction", "max_leverage", "cash_floor",
                  "max_margin_fraction", "slippage")
        values = {k: float(cfg[k]) for k in fields}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("nonfinite risk configuration")
        if (values["virtual_equity_usdt"] <= 0 or not 0 < values["risk_fraction"] <= .02
                or not 1 <= values["max_leverage"] <= 3 or not 0 <= values["cash_floor"] < 1
                or not 0 < values["max_margin_fraction"] <= .5 or not 0 <= values["slippage"] < 1):
            raise ValueError("risk configuration outside paper bounds")
        return {"available": True, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "owner_configured_virtual_paper_example_not_actual_account",
                "owner_config_hash": cfg_hash, **values,
                "example_stop_risk_budget_usdt": values["virtual_equity_usdt"] * values["risk_fraction"],
                "spec_snapshot_at_utc": PROVENANCE["retrieved_at_utc"],
                "spec_source_url": PROVENANCE["contracts_url"],
                "specs": {symbol: {k: item[k] for k in ("contract_size", "min_qty", "qty_step", "min_notional", "taker")}
                          for symbol, item in CONTRACTS.items()},
                "limitations": "Virtual example only; account holdings unknown. Specs and fees may be stale; exact margin/liquidation unverified; gaps may exceed stop budget."}
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"available": False, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": type(exc).__name__, "kind": "owner_configured_virtual_paper_example_not_actual_account"}


def _rsi_wilder(close: np.ndarray, period: int = 14) -> float | None:
    if len(close) <= period:
        return None
    changes = np.diff(close)
    avg_gain = float(np.maximum(changes[:period], 0).mean())
    avg_loss = float(np.maximum(-changes[:period], 0).mean())
    for delta in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(float(delta), 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-float(delta), 0.0)) / period
    if avg_gain + avg_loss == 0:
        return None
    return 100.0 * avg_gain / (avg_gain + avg_loss)


def _completed_four_hour_trend(candles: list[dict]) -> dict[str, Any]:
    """Describe only fully observed UTC-aligned four-hour bars, not a signal."""
    observed_times = [_aware(c["open_at_utc"]).astimezone(timezone.utc) for c in candles]
    if any(b - a != timedelta(hours=1) for a, b in zip(observed_times, observed_times[1:])):
        return {"ok": False, "reason": "duplicate_or_missing_hourly_bars"}
    groups: dict[datetime, list[dict]] = {}
    for candle in candles:
        ts = _aware(candle["open_at_utc"]).astimezone(timezone.utc)
        start = ts.replace(hour=ts.hour - ts.hour % 4, minute=0, second=0, microsecond=0)
        groups.setdefault(start, []).append(candle)
    complete = []
    for start, group in sorted(groups.items()):
        times = [_aware(c["open_at_utc"]).astimezone(timezone.utc) for c in group]
        if times == [start + timedelta(hours=i) for i in range(4)]:
            complete.append((start + timedelta(hours=4), float(group[-1]["close"])))
    if len(complete) < 50:
        return {"ok": False, "reason": "insufficient_complete_4h_bars", "bars_used": len(complete)}
    values = np.array([x[1] for x in complete], dtype=float)
    fast, slow = float(ema(values, 20)[-1]), float(ema(values, 50)[-1])
    return {"ok": True, "bars_used": len(complete), "last_closed_at_utc": complete[-1][0].isoformat(),
            "ema20": fast, "ema50": slow,
            "alignment": "up" if fast > slow else "down" if fast < slow else "equal",
            "note": "已完成的4小时均线关系，仅作市场背景，不是交易信号。"}


def _indicators(candles: list[dict], as_of: datetime | None = None) -> dict[str, Any]:
    if len(candles) < 50:
        return {"ok": False, "reason": "insufficient_closed_bars", "bars_used": len(candles)}
    from piia.intelligence.public_data import candle_quality
    try:
        problems = candle_quality(candles, as_of if as_of is not None else datetime.fromisoformat(candles[-1]["close_at_utc"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        problems = ["invalid_candles"]
    if problems:
        return {"ok": False, "reason": "invalid_closed_bars", "errors": problems, "bars_used": len(candles)}
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    a14 = atr(high, low, close, 14)
    last, prev = -1, -2
    cross = "none"
    if np.isfinite([e20[prev], e50[prev], e20[last], e50[last]]).all():
        if e20[last] > e50[last] and e20[prev] <= e50[prev]:
            cross = "bullish_cross"
        elif e20[last] < e50[last] and e20[prev] >= e50[prev]:
            cross = "bearish_cross"
    last_c = candles[-1]
    return {
        "ok": True,
        "bars_used": len(candles),
        "last_closed_bar_open_at_utc": last_c["open_at_utc"],
        "last_closed_bar_close_at_utc": last_c["close_at_utc"],
        "last_close": float(close[-1]),
        "ema20": float(e20[-1]) if np.isfinite(e20[-1]) else None,
        "ema50": float(e50[-1]) if np.isfinite(e50[-1]) else None,
        "atr14": float(a14[-1]) if np.isfinite(a14[-1]) else None,
        "atr14_pct_of_close": float(100 * a14[-1] / close[-1]) if np.isfinite(a14[-1]) and close[-1] > 0 else None,
        "rsi14_wilder": _rsi_wilder(close),
        "return_24h_pct": float(100 * (close[-1] / close[-25] - 1)) if len(close) >= 25 and close[-25] > 0 else None,
        "range_24h_high": float(np.max(high[-24:])),
        "range_24h_low": float(np.min(low[-24:])),
        "four_hour_trend": _completed_four_hour_trend(candles),
        "ema_cross_on_last_closed_bar": cross,
        "close_vs_ema20": float(close[-1] - e20[-1]) if np.isfinite(e20[-1]) else None,
        "close_vs_ema50": float(close[-1] - e50[-1]) if np.isfinite(e50[-1]) else None,
        "methodology": "closed 1h bars; SMA-seeded EMA20/50, Wilder ATR14 and RSI14; last 24 bars high/low and 24h close return; RSI undefined for a constant series",
        "note": "These indicators are a deterministic context snapshot, not a trading signal and not an LLM judgment.",
    }


def _daily_context(candles: list[dict], errors: list[str], as_of: datetime | None = None) -> dict[str, Any]:
    from piia.intelligence.public_data import candle_quality
    errors = list(errors)
    if len(candles) >= 50:
        try:
            errors.extend(candle_quality(candles, as_of if as_of is not None else datetime.fromisoformat(candles[-1]["close_at_utc"]), timedelta(days=1)))
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append("invalid_candles")
    errors = list(dict.fromkeys(errors))
    if errors or len(candles) < 50:
        return {"ok": False, "reason": "daily_data_unavailable_or_invalid", "errors": errors,
                "bars_used": len(candles)}
    close = np.array([c["close"] for c in candles], dtype=float)
    fast, slow = float(ema(close, 20)[-1]), float(ema(close, 50)[-1])
    sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else None
    return {"ok": True, "bars_used": len(candles), "timeframe": "1D UTC",
            "last_closed_at_utc": candles[-1]["close_at_utc"], "last_close": float(close[-1]),
            "last_20_closed_daily_high": max(float(c["high"]) for c in candles[-20:]),
            "last_20_closed_daily_low": min(float(c["low"]) for c in candles[-20:]),
            "last_20_closed_daily_start_utc": candles[-20]["open_at_utc"],
            "ema20": fast, "ema50": slow, "sma200": sma200,
            "close_vs_sma200": ("above" if close[-1] > sma200 else "below" if close[-1] < sma200 else "equal") if sma200 is not None else "insufficient_history",
            "alignment": "up" if fast > slow else "down" if fast < slow else "equal",
            "note": "仅为已收盘日线均线背景，不是方向预测；数据不足时不计算200日均线。"}


OPTIONAL_CONTEXTS = (
    ("macro_context", "macro_context_hash", "retrieved_at_utc"),
    ("paper_risk_context", "paper_risk_context_hash", "captured_at_utc"),
    ("onchain_context", "onchain_context_hash", "retrieved_at_utc"),
    ("etf_flow_context", "etf_flow_context_hash", "retrieved_at_utc"),
    ("event_calendar_context", "event_calendar_context_hash", "retrieved_at_utc"),
    ("gold_positioning_context", "gold_positioning_context_hash", "retrieved_at_utc"),
    ("prediction_market_context", "prediction_market_context_hash", "retrieved_at_utc"),
    ("fear_greed_context", "fear_greed_context_hash", "retrieved_at_utc"),
    ("liquidity_positioning_context", "liquidity_positioning_context_hash", "retrieved_at_utc"),
)


def build_request(snapshot: dict[str, Any]) -> dict[str, Any]:
    instruments = {}
    try:
        snapshot_time = datetime.fromisoformat(snapshot["retrieved_at_utc"])
        if snapshot_time.tzinfo is None or snapshot_time.utcoffset() is None:
            raise ValueError("naive retrieval time")
    except (KeyError, TypeError, ValueError):
        snapshot_time = None
    contexts = {name: snapshot.get(name) for name, _hash_name, _time in OPTIONAL_CONTEXTS}
    for sym, inst in (snapshot.get("instruments") or {}).items():
        candles = list(inst.get("candles_closed_1h") or [])
        daily = list(inst.get("candles_closed_1d") or [])
        body = {
            "symbol": sym,
            "instrument_id": inst.get("instrument_id") or CONTRACTS.get(sym, {}).get("instrument_id"),
            "venue": "bitget",
            "ticker": inst.get("ticker"),
            "funding": inst.get("funding"),
            "candles_closed_1h_tail": candles[-8:],
            "candles_count": len(candles),
            "candles_hash": content_hash(_canonical(candles)),
            "first_closed_open_at_utc": candles[0]["open_at_utc"] if candles else None,
            "last_closed_close_at_utc": candles[-1]["close_at_utc"] if candles else None,
            "indicators_prefix_only": (_indicators(candles, snapshot_time) if snapshot_time else
                                       {"ok": False, "reason": "snapshot_retrieval_time_unavailable"}),
            "daily_context": (_daily_context(daily, inst.get("daily_quality_errors") or [], snapshot_time) if snapshot_time else
                              {"ok": False, "reason": "snapshot_retrieval_time_unavailable"}),
            "daily_candles_hash": content_hash(_canonical(daily)),
            "daily_source_url": inst.get("daily_source_url"),
            "daily_history_source_urls": inst.get("daily_history_source_urls") or [],
        }
        from piia.intelligence.market_context import describe_market
        body["market_context"] = describe_market(body["indicators_prefix_only"], body["daily_context"])
        if "open_interest" in inst:
            body["open_interest"] = inst["open_interest"]
        for name, hash_name, _time in OPTIONAL_CONTEXTS:
            if contexts.get(name) is not None:
                body[hash_name] = content_hash(_canonical(contexts[name]))
        ev_hash = content_hash(_canonical(body))
        instruments[sym] = {**body, "evidence_hash": ev_hash, "all_closed_candles": candles,
                            "all_closed_daily_candles": daily}
    times = [snapshot.get("retrieved_at_utc"), snapshot.get("cutoff_at_utc")]
    for name, _hash_name, time_key in OPTIONAL_CONTEXTS:
        payload = contexts.get(name) or {}
        if payload.get(time_key):
            times.append(payload.get(time_key))
        if payload.get("completed_at_utc"):
            times.append(payload["completed_at_utc"])
        if name == "etf_flow_context":
            partial = payload.get("supplemental_partial")
            if isinstance(partial, dict) and partial.get("completed_at_utc"):
                times.append(partial["completed_at_utc"])
        if name == "etf_flow_context" and payload.get("asof_complete_flow_date"):
            times.append(f"{payload['asof_complete_flow_date']}T00:00:00+00:00")
        if name == "onchain_context":
            last = (payload.get("metrics") or {}).get("last_block") or {}
            if last.get("observation_at_utc"):
                times.append(last["observation_at_utc"])
        if name == "prediction_market_context" and payload.get("asof_utc"):
            times.append(payload["asof_utc"])
    for inst in instruments.values():
        times.append((inst.get("ticker") or {}).get("observed_at_utc"))
        times.append((inst.get("funding") or {}).get("retrieved_at_utc"))
        times.append((inst.get("open_interest") or {}).get("retrieved_at_utc"))
        times.append((inst.get("open_interest") or {}).get("period_end_utc"))
        times.append(inst.get("last_closed_close_at_utc"))
    cutoff = max(_aware(t) for t in times if t).isoformat()
    missing_datasets = list(snapshot.get("missing_datasets") or [])
    macro_context = contexts.get("macro_context")
    if macro_context and macro_context.get("series"):
        if "macro" in missing_datasets:
            missing_datasets.remove("macro")
            missing_datasets.append("macro_other_series_and_intraday_release_times")
        real_rate = macro_context["series"].get("DFII10")
        if real_rate and not real_rate.get("stale", True) and "real_rates" in missing_datasets:
            missing_datasets.remove("real_rates")
            missing_datasets.append("real_rates_intraday_release_time")
    request = {
        "packet_type": PACKET_REQUEST,
        "integrity_version": 2,
        "exported_at_utc": snapshot.get("retrieved_at_utc"),
        "cutoff_at_utc": cutoff,
        "independent_llm_review": "NOT_WIRED",
        "mode": "chat_assisted_manual",
        "model_status": model_status(),
        "missing_datasets": missing_datasets,
        "errors": snapshot.get("errors") or [],
        "snapshot_ok": bool(snapshot.get("ok")),
        "exact_liquidation_verified": False,
        "instructions": (
            "Fill a matching response using only evidence in this request. "
            "Do not add later prices, do not claim independent LLM review, "
            "and do not approve or place orders. Preserve disagreements. "
            "Label missing or delayed macro/on-chain data explicitly. Direction may be flat."
        ),
        "instruments": instruments,
    }
    for name, _hash_name, _time in OPTIONAL_CONTEXTS:
        if contexts.get(name) is not None:
            request[name] = contexts[name]
    digest_input = {
        "cutoff_at_utc": request["cutoff_at_utc"],
        "missing_datasets": request["missing_datasets"],
        "instruments": {s: {k: v[k] for k in v if k not in {"all_closed_candles", "all_closed_daily_candles"}} for s, v in instruments.items()},
        "integrity_version": request["integrity_version"],
        "packet_type": request["packet_type"],
        "exported_at_utc": request["exported_at_utc"],
        "snapshot_ok": request["snapshot_ok"],
        "errors": request["errors"],
    }
    for name, _hash_name, _time in OPTIONAL_CONTEXTS:
        if name in request:
            digest_input[name] = request[name]
    request["request_hash"] = content_hash(_canonical(digest_input))
    return request


class ClaimIn(BaseModel):
    text: str
    evidence_hash: str
    status: Literal["supported", "unsupported", "unverified_url", "invalid_ref"] = "supported"


class RoleIn(BaseModel):
    role: Literal["researcher", "critic", "synthesizer"]
    actor: str = ACTOR
    generation: Literal["manual", "external_fresh_context"] = "manual"
    direction: Direction | None = None
    summary: str
    claims: list[ClaimIn] = Field(default_factory=list)
    invalidation: str = ""
    disagreements: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    suggested_action: Action = "need_more_evidence"
    proposed_stop: float | None = None

    @field_validator("summary", "invalidation")
    @classmethod
    def nonempty_summary(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("summary/invalidation must not be empty")
        return v


class InstrumentResponseIn(BaseModel):
    symbol: str
    request_evidence_hash: str
    researcher: RoleIn
    critic: RoleIn
    synthesizer: RoleIn


class PacketResponseIn(BaseModel):
    packet_type: Literal["piia_chat_assisted_research_response"]
    request_hash: str
    cutoff_at_utc: str
    independent_llm_review: Literal["NOT_WIRED"] = "NOT_WIRED"
    actor: str = ACTOR
    notes: str = ""
    instruments: list[InstrumentResponseIn]


def _refuse_live_db(db_path: Path) -> None:
    if db_path.resolve() == FORBIDDEN_DB:
        raise ValidationFailed("refusing to import a research packet into the live research database")


def _role_body(role: RoleIn) -> str:
    return json.dumps({
        "direction": role.direction,
        "summary": role.summary,
        "invalidation": role.invalidation,
        "disagreements": role.disagreements,
        "missing_data": role.missing_data,
        "suggested_action": role.suggested_action,
        "proposed_stop": role.proposed_stop,
        "claims": [c.model_dump() for c in role.claims],
        "claim_status_note": "Claim statuses are the manual author's labels; references and source truth are not independently verified.",
        "generation": role.generation,
        "mode": "chat_assisted_manual" if role.generation == "manual" else "external_fresh_context",
        "independent_llm_review": "NOT_WIRED",
    }, ensure_ascii=False, indent=2)



FROZEN_INSTRUMENT_HASH_FIELDS = (
    "evidence_hash",
    "candles_hash",
    "daily_candles_hash",
    "macro_context_hash",
    "paper_risk_context_hash",
    "onchain_context_hash",
    "etf_flow_context_hash",
    "event_calendar_context_hash",
    "gold_positioning_context_hash",
    "prediction_market_context_hash",
    "fear_greed_context_hash",
    "liquidity_positioning_context_hash",
)


def _frozen_digest(value: Any) -> str | None:
    if isinstance(value, str) and HASH_RE.match(value):
        return value
    return None


def restricted_content_hash(packet: dict[str, Any]) -> str:
    """Hash a restricted evidence packet, excluding its own hash field."""
    body = {k: v for k, v in packet.items() if k != "restricted_packet_hash"}
    return content_hash(_canonical(body))


def frozen_hashes_for_symbol(request: dict[str, Any], symbol: str, *, restricted_packet: dict[str, Any] | None = None) -> set[str]:
    """Allow only sha256 digests that actually belong to this frozen request (and optional verified restricted packet)."""
    allowed: set[str] = set()
    digest = _frozen_digest(request.get("request_hash"))
    if digest:
        allowed.add(digest)
    raw = (request.get("instruments") or {}).get(symbol) or {}
    for field in FROZEN_INSTRUMENT_HASH_FIELDS:
        digest = _frozen_digest(raw.get(field))
        if digest:
            allowed.add(digest)
    if restricted_packet is not None:
        stored = _frozen_digest(restricted_packet.get("restricted_packet_hash"))
        actual = restricted_content_hash(restricted_packet)
        if stored != actual:
            raise ValidationFailed("restricted packet hash mismatch")
        if restricted_packet.get("request_hash") != request.get("request_hash"):
            raise ValidationFailed("restricted packet request_hash does not match the frozen request")
        if restricted_packet.get("cutoff_at_utc") != request.get("cutoff_at_utc"):
            raise ValidationFailed("restricted packet cutoff does not match the frozen request")
        allowed.add(actual)
    return allowed


def validate_response(request: dict[str, Any], response_obj: PacketResponseIn, *, restricted_packet: dict[str, Any] | None = None) -> None:
    instruments = request.get("instruments") or {}
    for symbol, raw in instruments.items():
        body = {k: v for k, v in raw.items() if k not in {"evidence_hash", "all_closed_candles", "all_closed_daily_candles"}}
        if content_hash(_canonical(body)) != raw.get("evidence_hash"):
            raise ValidationFailed(f"evidence content hash mismatch for {symbol}")
        if raw.get("candles_hash") and content_hash(_canonical(raw.get("all_closed_candles") or [])) != raw["candles_hash"]:
            raise ValidationFailed(f"candle content hash mismatch for {symbol}")
        if raw.get("daily_candles_hash") and content_hash(_canonical(raw.get("all_closed_daily_candles") or [])) != raw["daily_candles_hash"]:
            raise ValidationFailed(f"daily candle content hash mismatch for {symbol}")
        labels = {
            "macro_context": "macro context",
            "paper_risk_context": "paper risk context",
            "onchain_context": "onchain context",
            "etf_flow_context": "etf flow context",
            "event_calendar_context": "event calendar context",
            "gold_positioning_context": "gold positioning context",
            "prediction_market_context": "prediction market context",
            "fear_greed_context": "fear greed context",
            "liquidity_positioning_context": "liquidity positioning context",
        }
        for name, hash_name, _time in OPTIONAL_CONTEXTS:
            if raw.get(hash_name) and content_hash(_canonical(request.get(name))) != raw[hash_name]:
                raise ValidationFailed(f"{labels[name]} hash mismatch for {symbol}")
    digest_input = {
        "cutoff_at_utc": request.get("cutoff_at_utc"),
        "missing_datasets": request.get("missing_datasets"),
        "instruments": {s: {k: v[k] for k in v if k not in {"all_closed_candles", "all_closed_daily_candles"}} for s, v in instruments.items()},
    }
    if request.get("integrity_version") == 2:
        digest_input.update({k: request.get(k) for k in
                             ("integrity_version", "packet_type", "exported_at_utc", "snapshot_ok", "errors")})
        if request.get("packet_type") != PACKET_REQUEST:
            raise ValidationFailed("invalid request packet type")
    elif "integrity_version" in request:
        raise ValidationFailed("unsupported packet integrity version")
    for name, _hash_name, _time in OPTIONAL_CONTEXTS:
        if name in request:
            digest_input[name] = request[name]
    expected_request_hash = content_hash(_canonical(digest_input))
    if expected_request_hash != request.get("request_hash"):
        raise ValidationFailed("request content hash mismatch")
    if response_obj.request_hash != request.get("request_hash"):
        raise ValidationFailed("response request_hash does not match the frozen request")
    if _aware(response_obj.cutoff_at_utc) != _aware(request["cutoff_at_utc"]):
        raise ValidationFailed("response cutoff_at_utc must equal the request cutoff")
    if response_obj.independent_llm_review != "NOT_WIRED" and not model_configured():
        raise ValidationFailed("independent LLM review is not wired; cannot claim WIRED")
    req_inst = request.get("instruments") or {}
    cutoff = _aware(request["cutoff_at_utc"])
    for name, _hash_name, time_key in OPTIONAL_CONTEXTS:
        stamp = (request.get(name) or {}).get(time_key)
        if stamp and _aware(stamp) > cutoff:
            raise ValidationFailed(f"{name} timestamp is after cutoff")
    seen = set()
    for inst in response_obj.instruments:
        if inst.symbol in seen:
            raise ValidationFailed(f"duplicate symbol {inst.symbol}")
        seen.add(inst.symbol)
        raw = req_inst.get(inst.symbol)
        if not raw:
            raise ValidationFailed(f"response symbol {inst.symbol} is not in the request")
        if inst.request_evidence_hash != raw.get("evidence_hash"):
            raise ValidationFailed(f"evidence hash mismatch for {inst.symbol}")
        if inst.researcher.role != "researcher" or inst.critic.role != "critic" or inst.synthesizer.role != "synthesizer":
            raise ValidationFailed("roles must be researcher, critic, synthesizer")
        for role in (inst.researcher, inst.critic, inst.synthesizer):
            if role.generation not in {"manual", "external_fresh_context"}:
                raise ValidationFailed("import only accepts generation=manual or external_fresh_context")
            allowed_hashes = frozen_hashes_for_symbol(request, inst.symbol, restricted_packet=restricted_packet)
            for claim in role.claims:
                if claim.evidence_hash not in allowed_hashes:
                    raise ValidationFailed(f"claim evidence_hash is not in the frozen request for {inst.symbol}")
        ticker = (raw.get("ticker") or {}).get("observed_at_utc")
        last_close = raw.get("last_closed_close_at_utc")
        oi = raw.get("open_interest") or {}
        funding = raw.get("funding") or {}
        for ts in (ticker, last_close, funding.get("retrieved_at_utc"), oi.get("period_end_utc"), oi.get("retrieved_at_utc")):
            if ts and _aware(ts) > cutoff:
                raise ValidationFailed("request itself has evidence after cutoff")
    missing = set(req_inst) - seen
    if missing:
        raise ValidationFailed("response is missing instruments: " + ", ".join(sorted(missing)))


def import_response(
    request: dict[str, Any],
    response: dict[str, Any] | PacketResponseIn,
    *,
    db_path: Path,
    evidence_store: Path,
    clock: Clock | None = None,
    restricted_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _refuse_live_db(db_path)
    resp = response if isinstance(response, PacketResponseIn) else PacketResponseIn.model_validate(response)
    validate_response(request, resp, restricted_packet=restricted_packet)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_store.mkdir(parents=True, exist_ok=True)
    engine = make_engine(f"sqlite:///{db_path.as_posix()}")
    create_all(engine)
    session = session_factory(engine)()
    clock = clock or Clock()
    try:
        core = IntelligenceService(session, clock=clock, evidence_store=evidence_store)
        core.seed_instruments()
        ext = ExtendedService(core)
        account = ext.paper.open_account(label="CHAT_ASSISTED_RESEARCH_PACKET_SANDBOX")
        imported: list[dict[str, Any]] = []
        now = clock.now()
        for inst in resp.instruments:
            raw = request["instruments"][inst.symbol]
            ev_payload = json.dumps({
                "label": LABEL,
                "cutoff_at_utc": request["cutoff_at_utc"],
                "request_hash": request["request_hash"],
                "evidence_hash": raw["evidence_hash"],
                "ticker": raw.get("ticker"),
                "funding": raw.get("funding"),
                "open_interest": raw.get("open_interest"),
                "indicators_prefix_only": raw.get("indicators_prefix_only"),
                "daily_context": raw.get("daily_context"),
                "market_context": raw.get("market_context"),
                "candles_closed_1d": raw.get("all_closed_daily_candles"),
                "macro_context": request.get("macro_context"),
                "paper_risk_context": request.get("paper_risk_context"),
                "onchain_context": request.get("onchain_context"),
                "etf_flow_context": request.get("etf_flow_context"),
                "event_calendar_context": request.get("event_calendar_context"),
                "gold_positioning_context": request.get("gold_positioning_context"),
                "prediction_market_context": request.get("prediction_market_context"),
                "fear_greed_context": request.get("fear_greed_context"),
                "last_closed_close_at_utc": raw.get("last_closed_close_at_utc"),
                "candles_closed_1h": raw.get("all_closed_candles") or raw.get("candles_closed_1h_tail"),
                "missing_datasets": request.get("missing_datasets"),
            }, sort_keys=True, default=str)
            observed = _aware(raw.get("last_closed_close_at_utc") or request["cutoff_at_utc"])
            syn = inst.synthesizer
            direction: Direction = syn.direction or "flat"
            ev = EvidenceIn(
                raw=ev_payload,
                provider="bitget_public_snapshot",
                source_url=(raw.get("ticker") or {}).get("source_url"),
                quote_locator=f"{inst.symbol} closed 1h through {raw.get('last_closed_close_at_utc')}",
                observed_at=observed,
                published_at=None,
                available_at=_aware(request["cutoff_at_utc"]),
                timezone="UTC",
                data_quality="unverified",
                methodology="allowlisted public Bitget ticker, funding, and closed 1h candles; prefix-only indicators",
                methodology_version="chat_assisted_packet_v1",
                vintage=str(observed),
                label=LABEL,
                relation="supports",
                note="As-of snapshot at request cutoff. Macro, if present, is date-vintage only; intraday release times are unverified. On-chain data is missing.",
                historic_context=False,
            )
            statement = (
                f"{LABEL} chat-assisted writeup for {inst.symbol}. "
                f"Not an autonomous model run. independent_llm_review=NOT_WIRED. "
                f"direction={direction}. {syn.summary}"
            )
            th = core.create_thesis(ThesisCreate(
                instrument_id=raw["instrument_id"],
                author=resp.actor,
                direction=direction,
                horizon_days=1,
                invalidation=syn.invalidation,
                subjective_confidence=0.0,
                statement=statement,
                label=LABEL,
                is_example=False,
                evidence=[ev],
                change_reason="chat-assisted research packet import",
            ), idempotency_key=content_hash(f"{LABEL}:{request['request_hash']}:{inst.symbol}".encode())[:32])
            rev_id = th["current_revision_id"]
            # evidence id for claims
            rev = next(r for r in th.get("revisions") or [] if r["id"] == rev_id)
            ev_ids = [item["evidence"]["id"] for item in rev.get("evidence") or [] if item.get("evidence", {}).get("id")]
            if not ev_ids:
                raise ValidationFailed(f"imported thesis {th['id']} has no evidence ids")
            core.add_analysis(rev_id, AnalysisIn(
                actor=inst.researcher.actor, role="researcher", body=_role_body(inst.researcher),
                generation="manual", evidence_ids=ev_ids,
            ))
            core.add_analysis(rev_id, AnalysisIn(
                actor=inst.critic.actor, role="critic", body=_role_body(inst.critic),
                generation="manual", evidence_ids=ev_ids,
            ))
            core.add_analysis(rev_id, AnalysisIn(
                actor=inst.synthesizer.actor, role="synthesizer", body=_role_body(inst.synthesizer),
                generation="manual", evidence_ids=ev_ids,
            ))
            core.add_counterargument(rev_id, CounterIn(
                actor=inst.critic.actor, body=_role_body(inst.critic),
                generation="manual", evidence_ids=ev_ids,
            ))
            run = ResearchRun(
                id=new_id(), created_at=now, thesis_id=th["id"], revision_id=rev_id,
                mode="chat_assisted", status="completed",
                independent_llm_review="NOT_WIRED",
                prompt_hash=content_hash(b"piia_chat_assisted_packet_v1"),
                input_hash=request["request_hash"],
                model_name=None, model_version=None, token_cost=None,
                payload_json={"request_hash": request["request_hash"], "symbol": inst.symbol, "cutoff_at_utc": request["cutoff_at_utc"]},
                result_json={
                    "mode": "chat_assisted_manual",
                    "independent_llm_review": "NOT_WIRED",
                    "researcher": inst.researcher.model_dump(),
                    "critic": inst.critic.model_dump(),
                    "synthesizer": inst.synthesizer.model_dump(),
                    "disagreements": list(dict.fromkeys(inst.researcher.disagreements + inst.critic.disagreements + inst.synthesizer.disagreements)),
                },
                actor=resp.actor,
            )
            session.add(run)
            session.commit()
            plan_out = None
            risk_note = {
                "margin_model_supported": False,
                "liquidation_model_supported": False,
                "missing_datasets": request.get("missing_datasets"),
                "approved": False,
                "executed": False,
            }
            want_plan = (
                direction in {"long", "short"}
                and syn.suggested_action == "review_paper_plan"
                and syn.proposed_stop is not None
                and raw.get("ticker")
            )
            ticker_time = (raw.get("ticker") or {}).get("observed_at_utc")
            close_time = raw.get("last_closed_close_at_utc")
            candles = raw.get("all_closed_candles") or []
            freshness_reasons = []
            frozen_risk = request.get("paper_risk_context")
            if frozen_risk is not None:
                from piia.intelligence.paper.risk_file import load_owner_risk
                try:
                    _, current_risk_hash = load_owner_risk()
                    if not frozen_risk.get("available") or current_risk_hash != frozen_risk.get("owner_config_hash"):
                        freshness_reasons.append("paper_risk_changed_or_unavailable")
                except (OSError, TypeError, ValueError):
                    freshness_reasons.append("paper_risk_changed_or_unavailable")
            if request.get("integrity_version") != 2:
                freshness_reasons.append("legacy_snapshot_integrity_unverified")
            if not request.get("snapshot_ok"):
                freshness_reasons.append("snapshot_incomplete")
            if not raw.get("candles_hash"):
                freshness_reasons.append("candle_integrity_unverified")
            for label, timestamp, limit in (("ticker", ticker_time, timedelta(minutes=5)),
                                             ("closed_candle", close_time, timedelta(hours=2))):
                if not timestamp or not timedelta(0) <= now - _aware(timestamp) <= limit:
                    freshness_reasons.append(f"{label}_missing_or_stale")
            from piia.intelligence.public_data import candle_quality
            try:
                freshness_reasons.extend(candle_quality(candles, now))
            except (KeyError, ValueError, TypeError):
                freshness_reasons.append("invalid_candles")
            if not raw.get("indicators_prefix_only", {}).get("ok"):
                freshness_reasons.append("unavailable_indicators")
            if want_plan and not freshness_reasons:
                mark = float(raw["ticker"]["mark"] if raw["ticker"].get("mark") is not None else raw["ticker"]["last"])
                stop = float(syn.proposed_stop)
                plan = ext.create_sandbox_plan(
                    thesis_id=th["id"], account_id=account["id"], symbol=inst.symbol,
                    side=direction, entry_price=mark, stop_loss=stop, leverage=2.0,
                    research_run_id=run.id, stale=False, live_mode=False,
                )
                plan_out = {
                    "plan_id": plan["id"],
                    "status": plan.get("status"),
                    "actionable": plan.get("actionable"),
                    "side": direction,
                    "entry_price": mark,
                    "stop_loss": stop,
                    "approved": False,
                    "margin_model_supported": plan.get("margin_model_supported"),
                    "liquidation_model_supported": plan.get("liquidation_model_supported"),
                }
            elif want_plan:
                risk_note["no_plan_reason"] = "snapshot failed freshness or data quality checks"
                risk_note["data_quality_failures"] = freshness_reasons
            else:
                risk_note["no_plan_reason"] = (
                    "synthesizer is flat or asked to stand aside / get more evidence, or stop is missing"
                )
            imported.append({
                "symbol": inst.symbol,
                "instrument_id": raw["instrument_id"],
                "thesis_id": th["id"],
                "revision_id": rev_id,
                "research_run_id": run.id,
                "direction": direction,
                "suggested_action": syn.suggested_action,
                "invalidation": syn.invalidation,
                "disagreements": run.result_json["disagreements"],
                "plan": plan_out,
                "risk": risk_note,
            })
        session.commit()
        return {
            "imported": True,
            "label": LABEL,
            "mode": "chat_assisted_manual",
            "independent_llm_review": "NOT_WIRED",
            "approved": False,
            "live_orders": False,
            "request_hash": request["request_hash"],
            "cutoff_at_utc": request["cutoff_at_utc"],
            "database": str(db_path),
            "account_id": account["id"],
            "instruments": imported,
        }
    finally:
        session.close()


def default_packet_dir() -> Path:
    return ROOT / "artifacts" / "research_packets"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
