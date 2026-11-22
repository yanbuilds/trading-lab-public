"""Phase-2 execution of frozen chat-assisted historical research decisions.

Decisions are read-only. Approvals are sandbox-harness only, not user approvals.
This is a retrospective exploratory run on already-seen data, not a holdout and
not an independent multi-agent forecast.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from piia.intelligence.clock import FrozenClock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.hashing import content_hash
from piia.intelligence.historical_packets import (
    ASSET_MAP,
    DECISION_TIMES,
    HOLD_HOURS,
    PAIR,
    PROTOCOL,
    STOP_ATR_MULT,
)
from piia.intelligence.ids import new_id
from piia.intelligence.models import PaperLedger, PaperPosition, Plan, ResearchRun
from piia.intelligence.paper.engine import PaperEngine, spec_for
from piia.intelligence.paper.replay import (
    ACTOR,
    DAILY_LOSS_HALT,
    SLIPPAGE,
    TOTAL_LOSS_HALT,
    WIN_EPS,
    _equity,
)
from piia.intelligence.paper.specs import CONTRACTS
from piia.intelligence.paths import ROOT
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.schemas import (
    AnalysisIn,
    ApprovalIn,
    CounterIn,
    EvidenceIn,
    ThesisCreate,
)
from piia.intelligence.service import IntelligenceService
from trading_lab.perps.compare import funding_complete_between, load_cache

EXPECTED_DECISIONS_SHA256 = "d881a688aba853b7c8cd05f2028ba0bfe4a1ffcedfa0bb8abc42fe61624888ae"
LABEL = "HISTORICAL_RESEARCH_PACKETS_V1_PHASE2"
AUTHOR = "invest-desk_chat_assisted"
SANDBOX_APPROVAL_ACTOR = ACTOR  # sandbox_test_harness; not a user approval
DECISIONS_NAME = "CHAT_DECISIONS_FROZEN.json"
PACKET_DIR = ROOT / "artifacts" / "research_packets" / "historical_v1"
DATA_DIR = ROOT / "data" / "perps-extended"
DB_PATH = ROOT / "data" / "piia" / "piia.historical_research_v1.sqlite"
EVIDENCE_STORE = ROOT / "data" / "piia" / "historical_research_v1_evidence"
FORBIDDEN_DB = ROOT / "data" / "piia" / "piia.dev.sqlite"


class FrozenDecisionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aware(ts: str | datetime | pd.Timestamp) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def _taipei(ts: str | datetime | pd.Timestamp) -> str:
    t = pd.Timestamp(_aware(ts)).tz_convert("Asia/Taipei")
    return t.strftime("%Y-%m-%d %H:%M") + " (台北时间)"


def validate_decisions(payload: dict[str, Any], expected_ids: list[str]) -> None:
    if payload.get("protocol") != PROTOCOL:
        raise FrozenDecisionError("protocol mismatch")
    if payload.get("actor") != AUTHOR:
        raise FrozenDecisionError("unexpected actor")
    if payload.get("generation") != "manual":
        raise FrozenDecisionError("generation must be manual")
    if payload.get("independent_llm_review") != "NOT_WIRED":
        raise FrozenDecisionError("cannot claim independent LLM review")
    rows = payload.get("decisions") or []
    ids = [r.get("packet_id") for r in rows]
    if ids != expected_ids:
        raise FrozenDecisionError(f"packet_id order mismatch: {ids}")
    allowed = {"long", "short", "flat"}
    assets = {"Asset-A", "Asset-B", "Asset-C", None}
    n_dir = 0
    for row in rows:
        for role in ("researcher", "critic", "synthesizer"):
            body = row.get(role) or {}
            if body.get("direction") not in allowed:
                raise FrozenDecisionError(f"{row.get('packet_id')} {role} bad direction")
            if not str(body.get("summary") or "").strip() or not str(body.get("invalidation") or "").strip():
                raise FrozenDecisionError(f"{row.get('packet_id')} {role} missing rationale")
            if body.get("asset") not in assets:
                raise FrozenDecisionError(f"{row.get('packet_id')} {role} bad asset")
        syn = row["synthesizer"]
        if syn["direction"] in {"long", "short"}:
            n_dir += 1
            if syn.get("asset") is None:
                raise FrozenDecisionError(f"{row['packet_id']} directional without asset")
            if syn.get("suggested_action") != "review_paper_plan":
                raise FrozenDecisionError(f"{row['packet_id']} directional action must be review_paper_plan")
            if syn.get("accept_default_2atr_stop_and_72h_horizon") is not True:
                raise FrozenDecisionError(f"{row['packet_id']} must accept frozen stop/horizon")
        if syn["direction"] == "flat" and syn.get("suggested_action") != "stand_aside":
            raise FrozenDecisionError(f"{row['packet_id']} flat must stand_aside")
        for claim in (row["researcher"].get("claims") or []):
            if claim.get("status") not in {"supported", "unsupported", "unverified_url", "invalid_ref"}:
                raise FrozenDecisionError("invalid claim status")
    if n_dir != 6:
        raise FrozenDecisionError(f"expected 6 directional synthesizer decisions, got {n_dir}")


def load_frozen(packet_dir: Path = PACKET_DIR) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    dec_path = packet_dir / DECISIONS_NAME
    digest = sha256_file(dec_path)
    if digest != EXPECTED_DECISIONS_SHA256:
        raise FrozenDecisionError(f"decisions hash mismatch: {digest}")
    decisions = json.loads(dec_path.read_text())
    parent = json.loads((packet_dir / "PACKETS_FOR_REVIEW.json").read_text())
    sealed = json.loads((packet_dir / "BACKEND_SEALED.json").read_text())
    validate_decisions(decisions, [p["packet_id"] for p in parent["packets"]])
    if sealed.get("protocol") != PROTOCOL or sealed.get("contains_outcomes"):
        raise FrozenDecisionError("sealed mapping is not the phase-1 freeze")
    return decisions, parent, sealed, digest


def _role_body(role: str, body: dict[str, Any]) -> str:
    return json.dumps({
        "role": role,
        "direction": body.get("direction"),
        "asset": body.get("asset"),
        "summary": body.get("summary"),
        "invalidation": body.get("invalidation"),
        "disagreements": body.get("disagreements") or [],
        "missing_data": body.get("missing_data") or [],
        "suggested_action": body.get("suggested_action"),
        "claims": body.get("claims") or [],
        "claim_status_meaning": "supported means the claim is backed by then-known packet observations, not that a future return is true",
        "generation": "manual",
        "mode": "chat_assisted_manual",
        "independent_llm_review": "NOT_WIRED",
        "accept_default_2atr_stop_and_72h_horizon": body.get("accept_default_2atr_stop_and_72h_horizon"),
    }, ensure_ascii=False, indent=2)


@dataclass
class TradeRec:
    packet_id: str
    alias: str
    symbol: str
    side: str
    signal_time: str
    entry_time: str | None = None
    exit_time: str | None = None
    entry: float | None = None
    exit: float | None = None
    stop: float | None = None
    qty: float | None = None
    entry_fee: float = 0.0
    exit_fee: float | None = None
    funding_paid: float = 0.0
    net_usd: float | None = None
    exit_reason: str | None = None
    thesis_id: str | None = None
    plan_id: str | None = None
    decision_id: str | None = None
    position_id: str | None = None
    rejected: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _then_known_evidence(packet: dict[str, Any], event: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Only decision-time fields. No subsequent prices or returns."""
    return {
        "label": LABEL,
        "not_a_contemporaneous_publication": True,
        "protocol": PROTOCOL,
        "packet_id": packet["packet_id"],
        "decision_at_utc": event["decision_at_utc"],
        "last_closed_bar_open_utc": {a: event["assets"][a]["last_bar_open"] for a in ASSET_MAP},
        "then_known_raw": {a: {
            "last_close": event["assets"][a]["last_close"],
            "atr14": event["assets"][a]["atr14"],
            "ema20": event["assets"][a]["ema20"],
            "ema50": event["assets"][a]["ema50"],
            "bars_used": event["assets"][a]["bars_used"],
        } for a in ASSET_MAP},
        "parent_features": packet["assets"],
        "missing_datasets": parent.get("missing_datasets_all_packets"),
        "indicators_use_prefix_bars_only": True,
        "future_prices_excluded": True,
    }


def _record_exit(trade: TradeRec, result: dict[str, Any], when: datetime, reason: str) -> None:
    if not result.get("filled"):
        return
    trade.exit = float(result["exit_price"])
    trade.exit_time = when.isoformat()
    trade.exit_reason = reason
    trade.exit_fee = float(result["fee"])
    gross = float(result["gross_pnl"])
    trade.net_usd = gross - trade.entry_fee - float(result["fee"]) - trade.funding_paid


def run_turtle_same_window(data_dir: Path, start, end, capital: float, *, leverage: float, slippage: float) -> dict[str, Any]:
    """Frozen turtle comparator. Uses freqtrade-env when TA-Lib is not in this interpreter."""
    try:
        import talib  # noqa: F401
        from trading_lab.perps.compare import attach_turtle_s1, load_cache as _load, simulate_book
        frames, specs, funding, provenance = _load(data_dir)
        intervals = {k: v["funding_interval_hours"] for k, v in provenance["symbols"].items()}
        return simulate_book(
            frames, specs, funding, capital=capital, leverage=leverage,
            start=start, end=end, slippage=slippage, preparer=attach_turtle_s1, halt_mode="equity",
            funding_intervals=intervals,
        )
    except ModuleNotFoundError:
        py = ROOT.parent / "freqtrade-env" / "bin" / "python"
        if not py.exists():
            raise
        payload = {
            "data_dir": str(data_dir),
            "start": str(start),
            "end": str(end),
            "capital": capital,
            "leverage": leverage,
            "slippage": slippage,
        }
        code = (
            "import json,sys,pandas as pd\n"
            "from pathlib import Path\n"
            "from trading_lab.perps.compare import load_cache, simulate_book, attach_turtle_s1\n"
            "cfg=json.loads(sys.stdin.read())\n"
            "frames,specs,funding,prov=load_cache(Path(cfg['data_dir']))\n"
            "intervals={k:v['funding_interval_hours'] for k,v in prov['symbols'].items()}\n"
            "out=simulate_book(frames,specs,funding,capital=float(cfg['capital']),leverage=float(cfg['leverage']),"
            "start=pd.Timestamp(cfg['start']),end=pd.Timestamp(cfg['end']),slippage=float(cfg['slippage']),"
            "preparer=attach_turtle_s1,halt_mode='equity',funding_intervals=intervals)\n"
            "out.pop('nav',None)\n"
            "print(json.dumps(out,default=str))"
        )
        proc = subprocess.run(
            [str(py), "-c", code],
            input=json.dumps(payload),
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"turtle comparator failed: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])



def run_phase2(
    *,
    packet_dir: Path = PACKET_DIR,
    data_dir: Path = DATA_DIR,
    db_path: Path = DB_PATH,
    evidence_store: Path = EVIDENCE_STORE,
    starting_cash: float = 30.0,
    decisions_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any], str] | None = None,
    label: str = LABEL,
    author: str = AUTHOR,
    review_marker: str = "NOT_WIRED",
    mode: str = "chat_assisted_manual",
    label_quality: str = "retrospective_exploratory_chat_assisted",
    entry_prepare=None,
    report_extra: dict[str, Any] | None = None,
    integrity_ok: bool | None = None,
    protected_dbs: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    protected = {FORBIDDEN_DB.resolve(), *(p.resolve() for p in (protected_dbs or ()))}
    if db_path.resolve() in protected:
        raise FrozenDecisionError("refusing to write a protected research database")
    if db_path.exists():
        raise FileExistsError(f"replay database already exists; choose a new path: {db_path}")
    if decisions_bundle is None:
        decisions, parent, sealed, digest = load_frozen(packet_dir)
    else:
        decisions, parent, sealed, digest = decisions_bundle
    pre_hash = digest
    frames, specs, funding, _prov = load_cache(data_dir)
    by_id = {e["packet_id"]: e for e in sealed["events"]}
    dec_by_id = {d["packet_id"]: d for d in decisions["decisions"]}
    parent_by_id = {p["packet_id"]: p for p in parent["packets"]}
    # chronological
    ordered = sorted(sealed["events"], key=lambda e: e["sequence"])
    first = pd.Timestamp(ordered[0]["decision_at_utc"])
    last_exit = pd.Timestamp(ordered[-1]["horizon_exit_at_utc"])
    intervals = {k: v["funding_interval_hours"] for k, v in _prov["symbols"].items()}
    if set(funding) != set(frames) or set(intervals) != set(frames):
        raise FrozenDecisionError("missing funding or interval metadata for a research symbol")
    coverage = funding_complete_between(funding, intervals, first, last_exit)
    if not coverage["complete"]:
        raise FrozenDecisionError("funding missing or invalid under assumed intervals; replay refused")
    fmap = {
        k: {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["rate"]) for r in rows}
        for k, rows in funding.items()
    }
    pair0 = PAIR["BTCUSDT"]
    hours = frames[pair0].loc[first:last_exit].index
    for alias, sym in ASSET_MAP.items():
        other = frames[PAIR[sym]].loc[first:last_exit].index
        if not hours.equals(other):
            raise FrozenDecisionError(f"hourly timeline mismatch for {sym}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_store.mkdir(parents=True, exist_ok=True)
    clock = FrozenClock(first.to_pydatetime())
    engine = make_engine(f"sqlite:///{db_path.as_posix()}")
    create_all(engine)
    session = session_factory(engine)()
    try:
        core = IntelligenceService(session, clock=clock, evidence_store=evidence_store)
        core.seed_instruments()
        ext = ExtendedService(core)
        paper: PaperEngine = ext.paper
        acc = paper.open_account(starting_cash=starting_cash, label=LABEL + "_SANDBOX_30U")
        account_id = acc["id"]

        audits: list[dict[str, Any]] = []
        trades: list[TradeRec] = []
        rejections: Counter[str] = Counter()
        open_trade: TradeRec | None = None
        pos_id: str | None = None
        funding_net_paid = 0.0
        fees = 0.0
        nav: list[tuple[datetime, float]] = []
        kill = False
        day = None
        day_start = starting_cash
        stress_breaches = 0
        last_ts = None
        event_at = {pd.Timestamp(e["decision_at_utc"]): e for e in ordered}

        def current_spec():
            if not pos_id:
                return spec_for("BTCUSDT")
            pos = session.get(PaperPosition, pos_id)
            return spec_for(pos.symbol)

        for ts in hours:
            ts = pd.Timestamp(ts)
            if last_ts is not None and ts - last_ts != pd.Timedelta(hours=1):
                raise FrozenDecisionError("gapped evaluation timeline")
            last_ts = ts
            clock.set(ts.to_pydatetime())
            mark_for_day = None
            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                mark_for_day = float(frames[PAIR[pos.symbol]].at[ts, "open"])
            if ts.date() != day:
                day = ts.date()
                day_start = _equity(paper, session, account_id, pos_id, mark_for_day or starting_cash, current_spec())

            # funding then stops for carry-in positions
            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                if pos and pos.status == "open":
                    pair = PAIR[pos.symbol]
                    rate = fmap.get(pair, {}).get(ts, 0.0)
                    o = float(frames[pair].at[ts, "open"])
                    if rate:
                        paid = paper.apply_funding(pos_id, rate, o)
                        amount = float(paid["amount_paid"])
                        funding_net_paid += amount
                        if open_trade:
                            open_trade.funding_paid += amount
                        if paper.get_account(account_id).cash < 0:
                            raise FrozenDecisionError("Funding insolvency; exact liquidation model required; returns not published")
                    pos = session.get(PaperPosition, pos_id)
                    horizon = pd.Timestamp(by_id[open_trade.packet_id]["horizon_exit_at_utc"]) if open_trade else None
                    o, h, l, c = (float(frames[pair].at[ts, k]) for k in ("open", "high", "low", "close"))
                    side_i = 1 if pos.side == "long" else -1
                    gap = (side_i == 1 and o <= pos.stop_loss) or (side_i == -1 and o >= pos.stop_loss)
                    if horizon is not None and ts == horizon and not gap:
                        result = paper.exit_position(pos_id, o, "time_horizon", slippage=SLIPPAGE)
                        _record_exit(open_trade, result, ts.to_pydatetime(), "time_horizon")
                        fees += float(result["fee"])
                        pos_id = None
                        open_trade = None
                    else:
                        reason = "gap_stop" if gap else "atr_stop"
                        # At the scheduled horizon, only a gap at the open may precede the time exit
                        # (already handled). Intra-bar stops still apply on non-horizon bars,
                        # and on the horizon bar only if gapped at the open.
                        if gap or ts != horizon:
                            result = paper.check_stop(
                                pos_id, {"open": o, "high": h, "low": l, "close": c},
                                slippage=SLIPPAGE, reason=reason,
                            )
                            if result.get("filled"):
                                when = ts.to_pydatetime() if gap else (ts + pd.Timedelta(hours=1)).to_pydatetime()
                                _record_exit(open_trade, result, when, "gap_stop" if gap else "atr_stop")
                                fees += float(result["fee"])
                                if pos and pos.margin is not None and (-float(result["gross_pnl"]) + float(result["fee"])) > pos.margin + 1e-12:
                                    stress_breaches += 1
                                pos_id = None
                                open_trade = None

            eq_open_mark = starting_cash
            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                eq_open_mark = float(frames[PAIR[pos.symbol]].at[ts, "open"])
            eq_open = _equity(paper, session, account_id, pos_id, eq_open_mark, current_spec())
            if eq_open <= day_start * (1 - DAILY_LOSS_HALT) or eq_open <= starting_cash * (1 - TOTAL_LOSS_HALT):
                kill = True

            if ts in event_at:
                event = event_at[ts]
                packet_id = event["packet_id"]
                decision = dec_by_id[packet_id]
                packet = parent_by_id[packet_id]
                syn = decision["synthesizer"]
                direction = syn["direction"]
                alias = syn.get("asset")
                symbol = ASSET_MAP[alias] if alias else None
                instrument_id = (
                    CONTRACTS[symbol]["instrument_id"] if symbol else CONTRACTS["BTCUSDT"]["instrument_id"]
                )
                ev_payload = _then_known_evidence(packet, event, parent)
                observed = _aware(event["assets"][alias or "Asset-A"]["last_bar_open"]) + timedelta(hours=1)
                if observed > _aware(event["decision_at_utc"]):
                    raise FrozenDecisionError("evidence close time after decision clock")
                ev = EvidenceIn(
                    raw=json.dumps(ev_payload, sort_keys=True, default=str),
                    provider="bitget_public_ohlcv_cache",
                    source_url=None,
                    quote_locator=f"{packet_id} then-known closed 1h through {event['decision_at_utc']}",
                    observed_at=observed,
                    published_at=None,
                    available_at=observed,
                    timezone="UTC",
                    data_quality="unverified",
                    methodology="prefix-only indicators from the frozen historical packet; no future bars",
                    methodology_version="historical_packets_v1_phase2",
                    vintage=str(event["decision_at_utc"]),
                    label=label,
                    relation="supports",
                    note="As-of snapshot of then-known observations. Claims marked supported are input-supported, not future-return proofs.",
                    historic_context=True,
                )
                statement = (
                    f"{label} writeup for {packet_id}. "
                    f"Not an autonomous model run. independent_llm_review={review_marker}. "
                    f"direction={direction} asset={alias}. {syn['summary']}"
                )
                th = core.create_thesis(ThesisCreate(
                    instrument_id=instrument_id,
                    author=author,
                    direction=direction,
                    horizon_days=3,
                    invalidation=syn["invalidation"],
                    subjective_confidence=0.0,
                    statement=statement,
                    label=label,
                    is_example=True,
                    evidence=[ev],
                    change_reason="frozen research packet import; decisions are read-only",
                ), idempotency_key=content_hash(f"{label}:{digest}:{packet_id}".encode())[:32])
                rev_id = th["current_revision_id"]
                rev = next(r for r in th.get("revisions") or [] if r["id"] == rev_id)
                ev_ids = [item["evidence"]["id"] for item in rev.get("evidence") or [] if item.get("evidence", {}).get("id")]
                for role_name in ("researcher", "critic", "synthesizer"):
                    core.add_analysis(rev_id, AnalysisIn(
                        actor=author, role=role_name, body=_role_body(role_name, decision[role_name]),
                        generation="manual", evidence_ids=ev_ids,
                    ))
                core.add_counterargument(rev_id, CounterIn(
                    actor=author, body=_role_body("critic", decision["critic"]),
                    generation="manual", evidence_ids=ev_ids,
                ))
                run = ResearchRun(
                    id=new_id(), created_at=clock.now(), thesis_id=th["id"], revision_id=rev_id,
                    mode="chat_assisted", status="completed",
                    independent_llm_review=review_marker,
                    prompt_hash=content_hash(PROTOCOL.encode()),
                    input_hash=digest,
                    model_name=None, model_version=None, token_cost=None,
                    payload_json={"packet_id": packet_id, "decision_at_utc": event["decision_at_utc"]},
                    result_json={
                        "mode": mode,
                        "independent_llm_review": review_marker,
                        "researcher": decision["researcher"],
                        "critic": decision["critic"],
                        "synthesizer": decision["synthesizer"],
                    },
                    actor=author,
                )
                session.add(run)
                session.commit()
                audit = {
                    "packet_id": packet_id,
                    "decision_at_utc": event["decision_at_utc"],
                    "decision_at_taipei": _taipei(event["decision_at_utc"]),
                    "horizon_exit_at_taipei": _taipei(event["horizon_exit_at_utc"]),
                    "synthesizer_direction": direction,
                    "synthesizer_asset": alias,
                    "symbol": symbol,
                    "thesis_id": th["id"],
                    "research_run_id": run.id,
                    "action": syn.get("suggested_action") or "stand_aside",
                    "approval": "none",
                }
                if direction in {"long", "short"} and alias and syn.get("suggested_action") == "review_paper_plan":
                    if pos_id is not None:
                        rejections["max_open_positions"] += 1
                        audit["action"] = "skipped_open_position"
                        audit["rejected"] = "max_open_positions"
                    elif kill:
                        rejections["loss_halt"] += 1
                        audit["action"] = "skipped_loss_halt"
                        audit["rejected"] = "loss_halt"
                    else:
                        raw_open = float(frames[PAIR[symbol]].at[ts, "open"])
                        atr14 = float(event["assets"][alias]["atr14"])
                        side_i = 1 if direction == "long" else -1
                        stop = raw_open - side_i * (STOP_ATR_MULT * atr14)
                        gate = {"proceed": True, "reason": "ok", "max_qty": None, "notes": {}}
                        if entry_prepare is not None:
                            gate = entry_prepare(
                                raw_open=raw_open, stop_distance=abs(raw_open - stop),
                                side=direction, symbol=symbol, packet_id=packet_id,
                                decision=decision, event=event,
                            ) or gate
                        if not gate.get("proceed"):
                            reason = str(gate.get("reason") or "entry_gate_failed")
                            rejections[reason] += 1
                            audit["action"] = "stand_aside_gate"
                            audit["rejected"] = reason
                            audit["entry_gate"] = gate.get("notes") or {}
                            trades.append(TradeRec(
                                packet_id=packet_id, alias=alias, symbol=symbol, side=direction,
                                signal_time=event["decision_at_utc"], thesis_id=th["id"],
                                rejected=reason,
                            ))
                        else:
                            plan = ext.create_sandbox_plan(
                                thesis_id=th["id"], account_id=account_id, symbol=symbol,
                                side=direction, entry_price=raw_open, stop_loss=stop, leverage=2.0,
                                research_run_id=run.id, stale=False, live_mode=False,
                            )
                            if gate.get("max_qty") is not None:
                                prow = session.get(Plan, plan["id"])
                                req = dict(prow.request_json or {})
                                req["max_qty"] = float(gate["max_qty"])
                                req["entry_gate_notes"] = gate.get("notes") or {}
                                prow.request_json = req
                                session.commit()
                            dec = ext.approve_paper(
                                plan["id"],
                                ApprovalIn(actor=SANDBOX_APPROVAL_ACTOR, user_choice="approve", rejection_reason=None),
                                idempotency_key=content_hash(f"{label}:sandbox_approve:{packet_id}".encode())[:32],
                            )
                            audit["plan_id"] = plan["id"]
                            audit["approval"] = "sandbox_test_harness_only_not_user"
                            audit["entry_gate"] = gate.get("notes") or {}
                            if not dec.get("valid"):
                                rejections[f"approval_invalid:{dec.get('invalid_reason')}"] += 1
                                audit["action"] = "approval_invalid"
                                audit["rejected"] = dec.get("invalid_reason")
                            else:
                                fill = paper.execute_approved(account_id=account_id, decision_id=dec["id"])
                                if fill.get("status") != "filled":
                                    reason = str(fill.get("reason") or "rejected")
                                    rejections[reason] += 1
                                    audit["action"] = "order_rejected"
                                    audit["rejected"] = reason
                                    trades.append(TradeRec(
                                        packet_id=packet_id, alias=alias, symbol=symbol, side=direction,
                                        signal_time=event["decision_at_utc"], thesis_id=th["id"],
                                        plan_id=plan["id"], decision_id=dec["id"], rejected=reason,
                                    ))
                                else:
                                    pos = fill["position"]
                                    pos_id = pos["id"]
                                    fees += float(fill["fee"])
                                    open_trade = TradeRec(
                                        packet_id=packet_id, alias=alias, symbol=symbol, side=direction,
                                        signal_time=event["decision_at_utc"],
                                        entry_time=ts.isoformat(),
                                        entry=float(fill["fill_price"]),
                                        stop=float(pos["stop_loss"]),
                                        qty=float(fill["qty"]),
                                        entry_fee=float(fill["fee"]),
                                        thesis_id=th["id"], plan_id=plan["id"], decision_id=dec["id"],
                                        position_id=pos_id,
                                    )
                                    trades.append(open_trade)
                                    audit["action"] = "filled"
                                    audit["entry"] = open_trade.entry
                                    audit["qty"] = open_trade.qty
                                    audit["stop"] = open_trade.stop
                                    # intra-bar stop after entry, same bar
                                    pair = PAIR[symbol]
                                    o, h, l, c = (float(frames[pair].at[ts, k]) for k in ("open", "high", "low", "close"))
                                    result = paper.check_stop(
                                        pos_id, {"open": o, "high": h, "low": l, "close": c},
                                        slippage=SLIPPAGE, reason="atr_stop",
                                    )
                                    if result.get("filled"):
                                        _record_exit(open_trade, result, (ts + pd.Timedelta(hours=1)).to_pydatetime(), "atr_stop")
                                        fees += float(result["fee"])
                                        pos_id = None
                                        open_trade = None
                audits.append(audit)
                # attach exit later

            mark = starting_cash
            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                mark = float(frames[PAIR[pos.symbol]].at[ts, "close"])
            eq = _equity(paper, session, account_id, pos_id, mark, current_spec())
            if eq < 0:
                raise FrozenDecisionError("Negative equity stress path; exact liquidation required")
            nav.append(((ts + pd.Timedelta(hours=1)).to_pydatetime(), eq))
            if eq <= day_start * (1 - DAILY_LOSS_HALT) or eq <= starting_cash * (1 - TOTAL_LOSS_HALT):
                kill = True
            if kill and pos_id:
                pos = session.get(PaperPosition, pos_id)
                if pos and pos.status == "open":
                    px = float(frames[PAIR[pos.symbol]].at[ts, "close"])
                    clock.set((ts + pd.Timedelta(hours=1)).to_pydatetime())
                    result = paper.exit_position(pos_id, px, "loss_halt", slippage=SLIPPAGE)
                    _record_exit(open_trade, result, (ts + pd.Timedelta(hours=1)).to_pydatetime(), "loss_halt")
                    fees += float(result["fee"])
                    pos_id = None
                    open_trade = None
                    nav[-1] = (nav[-1][0], paper.get_account(account_id).cash)

        # attach exits to audits
        trade_by_packet = {t.packet_id: t for t in trades}
        for audit in audits:
            t = trade_by_packet.get(audit["packet_id"])
            if t:
                audit["exit_time_taipei"] = _taipei(t.exit_time) if t.exit_time else None
                audit["exit_reason"] = t.exit_reason
                audit["net_usd"] = t.net_usd
                audit["rejected"] = audit.get("rejected") or t.rejected

        account = paper.get_account(account_id)
        closed = [t for t in trades if t.exit_reason is not None]
        still_open = [t for t in trades if t.entry and t.exit_reason is None]
        unfilled = [t for t in trades if t.rejected]
        pnls = [float(t.net_usd) for t in closed]
        wins = [p for p in pnls if p > WIN_EPS]
        losses = [p for p in pnls if p < -WIN_EPS]
        breakeven = [p for p in pnls if abs(p) <= WIN_EPS]
        eq_series = pd.Series([starting_cash] + [v for _, v in nav], dtype=float)
        final_eq = float(eq_series.iloc[-1]) if len(eq_series) else starting_cash
        open_unrealized = 0.0
        if still_open and pos_id:
            pos = session.get(PaperPosition, pos_id)
            last_close = float(frames[PAIR[pos.symbol]]["close"].loc[:last_exit].iloc[-1])
            open_unrealized = _equity(paper, session, account_id, pos_id, last_close, spec_for(pos.symbol)) - account.cash - (pos.margin or 0)
        closed_net = sum(pnls)
        if still_open and pos_id:
            expected_final = starting_cash + closed_net + open_unrealized - (still_open[0].entry_fee + still_open[0].funding_paid)
            recon = final_eq - expected_final
        else:
            recon = closed_net - (final_eq - starting_cash)
        from sqlalchemy import select as _select
        ledger = list(session.scalars(_select(PaperLedger).where(PaperLedger.account_id == account_id)))
        ledger_sum = sum(float(x.amount) for x in ledger)
        # cash should equal sum of ledger (open_account + subsequent)
        cash_vs_ledger = float(account.cash) - ledger_sum

        turtle = run_turtle_same_window(
            data_dir, first, last_exit, starting_cash, leverage=2.0, slippage=SLIPPAGE,
        )
        # drop bulky nav from turtle copy for the saved report but keep summary
        turtle_summary = {k: turtle[k] for k in turtle if k not in {"nav", "trades"}}
        turtle_summary["trades"] = [
            {k: t[k] for k in t if k in {"symbol", "side", "entry_time", "exit_time", "net_pnl", "exit_reason"}}
            for t in turtle.get("trades") or []
        ]
        turtle_summary["comparison_note"] = (
            "Same starting 30U, leverage, fees, funding proxy, slippage, and overall window "
            f"({first} to {last_exit} UTC). Turtle uses its own Donchian entries, 10-day exits "
            "and 2N stops, not the 10 discrete research clocks or the 72-hour hold. "
            "This is not an apples-to-apples trade-for-trade match."
        )

        report = {
            "label": label,
            "protocol": PROTOCOL,
            "phase": 2,
            "mode": mode,
            "independent_llm_review": review_marker,
            "authorship": {
                "writeup_actor": author,
                "approval_actor": SANDBOX_APPROVAL_ACTOR,
                "approval_kind": "harness_sandbox_only_not_user",
                "user_approvals": False,
                "live_orders": False,
            },
            "decisions_sha256": pre_hash,
            "decisions_file_unmodified": (
                sha256_file(packet_dir / DECISIONS_NAME) == pre_hash
                if integrity_ok is None and (packet_dir / DECISIONS_NAME).exists()
                else (True if integrity_ok is None else bool(integrity_ok))
            ),
            "packet_file_hash": parent.get("packet_file_hash"),
            "held_out": False,
            "already_seen_dataset": True,
            "label_quality": label_quality,
            "starting_cash": starting_cash,
            "final_equity": final_eq,
            "return_pct": (final_eq / starting_cash - 1.0) * 100,
            "max_drawdown_pct": float((1 - eq_series / eq_series.cummax()).max() * 100),
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "winrate_pct": (100 * len(wins) / len(pnls)) if pnls else None,
            "winrate_definition": f"closed fills only; win if net_usd > {WIN_EPS}; not a calibrated predictive accuracy",
            "fees": fees,
            "funding_net_paid": funding_net_paid,
            "net_costs": fees + funding_net_paid,
            "slippage_each_side": SLIPPAGE,
            "rejections": dict(rejections),
            "unfilled": [t.as_dict() for t in unfilled],
            "open_positions": len(still_open),
            "open_unrealized": open_unrealized,
            "loss_halt": bool(kill),
            "daily_loss_halt": DAILY_LOSS_HALT,
            "total_loss_halt": TOTAL_LOSS_HALT,
            "stress_margin_breaches": stress_breaches,
            "liquidation_model": "approximate_stress_not_venue_verified",
            "reconciliation_error": recon,
            "reconciliation_ok": abs(recon) <= 1e-6,
            "cash_vs_ledger": cash_vs_ledger,
            "ledger_ok": abs(cash_vs_ledger) <= 1e-6,
            "window_start_utc": str(first),
            "window_end_utc": str(last_exit),
            "window_start_taipei": _taipei(first),
            "window_end_taipei": _taipei(last_exit),
            "account_id": account_id,
            "db_path": str(db_path),
            "asset_map": ASSET_MAP,
            "directional_candidates": 6,
            "flat_decisions": 4,
            "audits": audits,
            "closed_trade_summaries": [t.as_dict() for t in closed],
            "open_trade_summaries": [t.as_dict() for t in still_open],
            "flat_cash_benchmark": {
                "starting_cash": starting_cash,
                "final_equity": starting_cash,
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "note": "Stay in cash, no trades, same window.",
            },
            "turtle_s1_same_window": turtle_summary,
            "limitations": [
                "Ten pre-frozen windows on already-seen data; exploratory and retrospective, not an independent holdout.",
                "Decisions were chat-assisted role-separated writeups by one assistant, not an independent LLM or official model API.",
                "Only then-known technical snapshots were shown at decision time; macro, news, and on-chain data were missing.",
                "This is not a complete PIIA all-data / multi-agent predictive proof.",
                "Masking reduced but did not eliminate pretraining or prior-session contamination.",
                "Claims marked supported mean the observation was in the packet, not that a future profit claim is true; future-profit claims were marked unsupported.",
                "Exact venue liquidation is not verified.",
                "Turtle comparison shares budget, costs, and the calendar window, but not entry clocks or holding rules.",
                "Sandbox approvals are a test harness, not user approvals, and no live or demo orders were sent.",
            ],
        }
        if report_extra:
            report.update(report_extra)
        return report
    finally:
        session.close()


def write_report(report: dict[str, Any], out_dir: Path = PACKET_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "PHASE2_RESULTS.json"
    # Do not write into the frozen decisions file.
    slim = {k: v for k, v in report.items()}
    js.write_text(json.dumps(slim, indent=2, ensure_ascii=False, default=str))
    wr = report["winrate_pct"]
    wr_s = "n/a" if wr is None else f"{wr:.1f}% ({report['wins']}/{report['closed_trades']})"
    turtle = report["turtle_s1_same_window"]
    twr = turtle.get("winrate_pct")
    twr_s = "n/a" if twr is None else f"{twr:.1f}% ({turtle['wins']}/{turtle['closed_trades']})"
    lines = [
        "PIIA 历史研究决策测试 — 第2阶段结果（探索性回溯，不是样本外）",
        f"protocol={PROTOCOL}",
        f"decisions_sha256={report['decisions_sha256']} unmodified={report['decisions_file_unmodified']}",
        f"窗口：{report['window_start_taipei']} 至 {report['window_end_taipei']}",
        (
            f"研究决策连续账户：{report['starting_cash']:.2f}U -> {report['final_equity']:.2f}U  "
            f"收益率 {report['return_pct']:.2f}%  最大回撤 {report['max_drawdown_pct']:.2f}%  "
            f"平仓胜率 {wr_s}  手续费 {report['fees']:.4f}U  资金费净支付 {report['funding_net_paid']:.4f}U"
        ),
        f"拒单 {report['rejections'] or '{}'}  未成交 {len(report['unfilled'])}  未平仓 {report['open_positions']}  停机={report['loss_halt']}",
        f"对账 {'通过' if report['reconciliation_ok'] and report['ledger_ok'] else '失败'}  recon={report['reconciliation_error']:.6g}  cash_vs_ledger={report['cash_vs_ledger']:.6g}",
        (
            f"同窗口冻结海龟（自身入场/离场，非72小时对照）："
            f"{turtle['initial']:.2f}U -> {turtle['final']:.2f}U  收益率 {turtle['return_pct']:.2f}%  "
            f"最大回撤 {turtle['max_drawdown_pct']:.2f}%  平仓胜率 {twr_s}"
        ),
        "现金基准：30U 不动，收益率 0%。",
        "本结果不能证明策略未来盈利，也不是完整多Agent预测验证。",
    ]
    for a in report["audits"]:
        extra = ""
        if a.get("action") == "filled" or a.get("net_usd") is not None:
            extra = f" 净盈亏={a.get('net_usd')} 离场={a.get('exit_reason')} {a.get('exit_time_taipei')}"
        elif a.get("rejected"):
            extra = f" 拒绝={a.get('rejected')}"
        lines.append(
            f"- {a['packet_id']} {a['decision_at_taipei']} {a['synthesizer_direction']}"
            f"/{a['synthesizer_asset'] or '-'} {a.get('symbol') or '-'} {a['action']}{extra}"
        )
    txt = out_dir / "PHASE2_RESULTS.txt"
    txt.write_text("\n".join(lines) + "\n")
    return {"json": str(js), "txt": str(txt)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Replay frozen research into a new local database")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="new replay database path; existing files are never replaced")
    parser.add_argument("--out-dir", type=Path, default=PACKET_DIR, help="directory for generated reports")
    args = parser.parse_args()
    report = run_phase2(db_path=args.db)
    paths = write_report(report, out_dir=args.out_dir)
    # Re-check frozen file after the run.
    if sha256_file(PACKET_DIR / DECISIONS_NAME) != EXPECTED_DECISIONS_SHA256:
        raise SystemExit("frozen decisions file changed during the run")
    print("wrote", paths["txt"])
    print("wrote", paths["json"])
    print("return_pct", round(report["return_pct"], 4), "dd", round(report["max_drawdown_pct"], 4))
    print("wins/closed", report["wins"], report["closed_trades"], "rejections", report["rejections"])
    print("recon_ok", report["reconciliation_ok"], "ledger_ok", report["ledger_ok"])
    print("turtle_return", round(report["turtle_s1_same_window"]["return_pct"], 4))
    print("decisions_unmodified", report["decisions_file_unmodified"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
