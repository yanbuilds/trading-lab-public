"""Frozen-research vs fresh-entry risk revalidation.

The original research cycle evidence, role files, and STATE.json are never
rewritten. A new timestamped execution context is written beside them. A fresh
ticker or book does not cure stale research, does not turn a flat stance into
a signal, and never auto-approves a plan.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from piia.intelligence.hashing import content_hash
from piia.intelligence.ids import new_id
from piia.intelligence.paper.risk_file import load_owner_risk
from piia.intelligence.public_orderbook import (
    QUOTE_MAX_AGE,
    fetch_execution_books,
    floor_qty,
    walk_book,
)
from piia.intelligence.research_packet import write_json

PROTOCOL = "PIIA_EXECUTION_REVALIDATION_V1"
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
CANDLE_MAX_AGE = timedelta(hours=2)
PROPOSAL_TTL = timedelta(minutes=5)
FIXTURE_LABEL = "FIXTURE_NOT_REAL_MODEL_TRADE"
FROZEN_RELATIVE = (
    "STATE.json",
    "evidence/request.json",
    "dispatch/RESTRICTED_EVIDENCE.json",
    "import/result.json",
    "inbox/researcher.json",
    "inbox/critic_independent.json",
    "inbox/synthesizer.json",
)


def _aware(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _dec(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_frozen_files(run_dir: Path) -> dict[str, str | None]:
    return {rel: file_sha256(Path(run_dir) / rel) for rel in FROZEN_RELATIVE}


def _explicit_expiry(*payloads: dict[str, Any] | None) -> datetime | None:
    keys = ("thesis_expires_at_utc", "horizon_end_utc", "expires_at_utc", "invalidation_at_utc")
    found = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            raw = payload.get(key)
            if not raw:
                continue
            try:
                stamp = _aware(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
            except (TypeError, ValueError):
                continue
            found.append(stamp)
    return min(found) if found else None


def _portfolio_state(run_dir: Path, risk: dict[str, Any]) -> dict[str, Any]:
    """Virtual sandbox portfolio only. Never infer the user's real holdings."""
    positions: list[dict[str, Any]] = []
    db = Path(run_dir) / "db" / "research_cycle.sqlite"
    if db.is_file():
        try:
            import sqlite3
            with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        "select symbol, side, qty, status from paper_positions where status = 'open'"
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                for row in rows:
                    positions.append({"symbol": row["symbol"], "side": row["side"], "qty": str(row["qty"]), "status": row["status"]})
        except sqlite3.Error:
            positions = []
    body = {
        "kind": "virtual_paper_sandbox_not_user_account",
        "virtual_equity_usdt": str(risk.get("virtual_equity_usdt")),
        "open_positions": positions,
        "assumes_user_holdings": False,
        "source": "cycle_isolated_sqlite_or_empty",
    }
    body["portfolio_state_hash"] = content_hash(json.dumps(body, sort_keys=True, default=str).encode("utf-8"))
    return body


def research_blockers(
    request: dict[str, Any],
    synth_item: dict[str, Any],
    symbol: str,
    *,
    now: datetime,
    frozen_hashes: dict[str, str | None],
    expected_hashes: dict[str, str | None] | None,
    instrument_blockers: list[str],
    packet_blockers: list[str],
) -> list[str]:
    """Stale or incomplete research. A new quote cannot clear these reasons."""
    reasons: list[str] = []
    if expected_hashes:
        for rel, digest in expected_hashes.items():
            if digest and frozen_hashes.get(rel) != digest:
                reasons.append(f"tampered_or_unknown_frozen_hash:{rel}")
    raw = (request.get("instruments") or {}).get(symbol)
    if not raw:
        reasons.append("missing_frozen_instrument")
        return list(dict.fromkeys(reasons))
    if request.get("integrity_version") != 2:
        reasons.append("legacy_snapshot_integrity_unverified")
    if not request.get("snapshot_ok"):
        reasons.append("snapshot_incomplete")
    close_time = raw.get("last_closed_close_at_utc")
    try:
        close_at = _aware(datetime.fromisoformat(close_time)) if close_time else None
    except (TypeError, ValueError):
        close_at = None
    if close_at is None or not timedelta(0) <= now - close_at <= CANDLE_MAX_AGE:
        reasons.append("research_candle_stale_or_missing")
    expiry = _explicit_expiry(synth_item, raw, request)
    if expiry is not None and now >= expiry:
        reasons.append("thesis_expired")
    direction = synth_item.get("direction") or "flat"
    action = synth_item.get("suggested_action")
    if direction not in {"long", "short"} or action != "review_paper_plan" or synth_item.get("proposed_stop") is None:
        if action == "stand_aside":
            reasons.append("synthesizer_stand_aside")
        elif action == "need_more_evidence":
            reasons.append("synthesizer_need_more_evidence")
        else:
            reasons.append("synthesizer_flat_or_not_actionable")
    reasons.extend(f"research_blocker:{item}" for item in packet_blockers)
    reasons.extend(f"research_blocker:{item}" for item in instrument_blockers)
    return list(dict.fromkeys(reasons))


def _levels(book: dict[str, Any], key: str) -> list[dict[str, str]]:
    return list(book.get(key) or [])


def _entry_exit_sides(side: str) -> tuple[str, str]:
    if side == "long":
        return "asks", "bids"
    if side == "short":
        return "bids", "asks"
    raise ValueError("side must be long or short")


def cost_entry_and_exit(
    book: dict[str, Any],
    *,
    side: str,
    qty: Decimal,
    stop: Decimal,
    taker: Decimal,
    contract_size: Decimal,
) -> dict[str, Any]:
    if side not in {"long", "short"}:
        return {"ok": False, "reason": "wrong_side"}
    if not book.get("ok"):
        return {"ok": False, "reason": book.get("reason") or "missing_or_stale_book"}
    entry_key, exit_key = _entry_exit_sides(side)
    entry = walk_book(_levels(book, entry_key), qty)
    exit_fill = walk_book(_levels(book, exit_key), qty)
    if not entry.get("ok"):
        return {"ok": False, "reason": f"entry_{entry.get('reason')}", "entry": entry, "exit": exit_fill}
    if not exit_fill.get("ok"):
        return {"ok": False, "reason": f"exit_{exit_fill.get('reason')}", "entry": entry, "exit": exit_fill}
    entry_vwap = Decimal(entry["vwap"])
    exit_vwap = Decimal(exit_fill["vwap"])
    mid = _dec(book.get("mid"))
    mark_note = "Mark/index prices are not executable fills."
    if (side == "long" and stop >= entry_vwap) or (side == "short" and stop <= entry_vwap):
        return {"ok": False, "reason": "stop_does_not_invalidate_correctly", "entry": entry, "exit": exit_fill}
    # Conservative realized exit: worse of the stop and the visible exit book.
    if side == "long":
        conservative_exit = min(stop, exit_vwap)
        gross = (conservative_exit - entry_vwap) * qty * contract_size
    else:
        conservative_exit = max(stop, exit_vwap)
        gross = (entry_vwap - conservative_exit) * qty * contract_size
    entry_fee = entry_vwap * qty * contract_size * taker
    exit_fee = conservative_exit * qty * contract_size * taker
    net = gross - entry_fee - exit_fee
    slip_vs_mid = None
    if mid and mid > 0:
        slip_vs_mid = (entry_vwap - mid) / mid if side == "long" else (mid - entry_vwap) / mid
    return {
        "ok": True,
        "reason": None,
        "side": side,
        "qty": str(qty),
        "entry_side": entry_key,
        "exit_side": exit_key,
        "entry": entry,
        "exit": exit_fill,
        "entry_vwap": str(entry_vwap),
        "visible_exit_vwap": str(exit_vwap),
        "stop": str(stop),
        "conservative_exit": str(conservative_exit),
        "taker_fee_rate": str(taker),
        "entry_taker_fee": str(entry_fee),
        "exit_taker_fee": str(exit_fee),
        "estimated_net_if_stopped": str(net),
        "spread": book.get("spread"),
        "mid": book.get("mid"),
        "slippage_vs_mid": None if slip_vs_mid is None else str(slip_vs_mid),
        "mark_is_not_executable": True,
        "mark_note": mark_note,
        "future_fill_not_guaranteed": True,
    }


def _virtual_qty(risk: dict[str, Any], *, price: Decimal, stop: Decimal, contract: dict[str, Any], open_count: int) -> tuple[Decimal, str]:
    from trading_lab.perps.risk import Spec, size_order
    if open_count >= int(risk.get("max_open_positions") or 1):
        return Decimal("0"), "existing_virtual_position_blocks_new_entry"
    spec = Spec(
        symbol=str(contract["symbol"]),
        min_qty=float(contract["min_qty"]),
        qty_step=float(contract["qty_step"]),
        min_notional=float(contract["min_notional"]),
        taker=float(contract["taker"]),
        contract_size=float(contract.get("contract_size") or 1),
    )
    equity = float(risk["virtual_equity_usdt"])
    qty, reason = size_order(
        equity, equity, float(price), float(abs(price - stop)), spec,
        leverage=float(risk["max_leverage"]),
        risk_fraction=float(risk["risk_fraction"]),
        cash_floor=float(risk["cash_floor"]),
        max_margin_fraction=float(risk["max_margin_fraction"]),
        slippage=float(risk.get("slippage") or 0),
    )
    stepped = floor_qty(Decimal(str(qty)), Decimal(str(contract["qty_step"])))
    if stepped != Decimal(str(qty)) and qty:
        return Decimal("0"), "qty_precision"
    return stepped, reason


def evaluate_symbol(
    symbol: str,
    *,
    request: dict[str, Any],
    synth_item: dict[str, Any],
    books: dict[str, Any],
    risk: dict[str, Any],
    risk_hash: str,
    frozen_risk_hash: str | None,
    portfolio: dict[str, Any],
    now: datetime,
    research_reasons: list[str],
    bindings: dict[str, Any],
    tiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inst_books = (books.get("instruments") or {}).get(symbol) or {}
    book = inst_books.get("orderbook") or {"ok": False, "reason": "missing_new_context"}
    ticker = inst_books.get("ticker") or {"ok": False, "reason": "missing_new_context"}
    contract = inst_books.get("contract")
    execution_reasons: list[str] = []
    if frozen_risk_hash and frozen_risk_hash != risk_hash:
        execution_reasons.append("risk_config_changed")
    if not inst_books or not contract or not book or not ticker:
        execution_reasons.append("missing_new_execution_context")
    if ticker and not ticker.get("ok"):
        execution_reasons.append(f"ticker_{ticker.get('reason') or 'unavailable'}")
    if book and not book.get("ok"):
        execution_reasons.append(f"book_{book.get('reason') or 'unavailable'}")
    if contract and contract.get("status") not in (None, "normal"):
        execution_reasons.append(f"contract_status_{contract.get('status')}")
    if book.get("symbol") and book.get("symbol") != symbol:
        execution_reasons.append("cross_symbol_book")
    if ticker.get("symbol") and ticker.get("symbol") != symbol:
        execution_reasons.append("cross_symbol_ticker")

    constraints = {
        "symbol": symbol,
        "min_qty": None if not contract else contract.get("min_qty"),
        "qty_step": None if not contract else contract.get("qty_step"),
        "min_notional": None if not contract else contract.get("min_notional"),
        "taker": None if not contract else contract.get("taker"),
        "max_market_order_qty": None if not contract else contract.get("max_market_order_qty"),
        "visible_bid_size_sum": book.get("visible_bid_size_sum"),
        "visible_ask_size_sum": book.get("visible_ask_size_sum"),
        "spread": book.get("spread"),
        "best_bid": book.get("best_bid"),
        "best_ask": book.get("best_ask"),
        "mark": ticker.get("mark"),
        "mark_is_not_executable": True,
        "exact_liquidation_verified": False,
        "margin_tiers_supported": bool((tiers or {}).get("ok")),
        "margin_mode": risk.get("margin_mode") or "isolated",
        "future_fill_not_guaranteed": True,
    }
    result = {
        "symbol": symbol,
        "research_blockers": research_reasons,
        "execution_blockers": execution_reasons,
        "fresh_quote_cures_stale_research": False,
        "constraints": constraints,
        "proposal": None,
        "decision": "structured_rejection",
        "approved": False,
        "executed": False,
    }
    if research_reasons:
        result["decision"] = "structured_rejection"
        result["no_plan_reason"] = "stale_or_non_actionable_research_not_cured_by_fresh_quote"
        return result
    if execution_reasons:
        result["no_plan_reason"] = "execution_context_rejected"
        return result

    side = synth_item["direction"]
    stop = _dec(synth_item.get("proposed_stop"))
    if stop is None or stop <= 0:
        result["execution_blockers"].append("missing_or_invalid_stop")
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    # Use best executable price for initial sizing, not the mark.
    entry_key, _exit_key = _entry_exit_sides(side)
    best = _dec(book.get("best_ask") if entry_key == "asks" else book.get("best_bid"))
    if best is None:
        result["execution_blockers"].append("missing_best_executable_price")
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    qty, size_reason = _virtual_qty(risk, price=best, stop=stop, contract=contract, open_count=len(portfolio.get("open_positions") or []))
    if qty <= 0:
        result["execution_blockers"].append(f"size_{size_reason}")
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    cap = _dec(contract.get("max_market_order_qty"))
    if cap is not None and qty > cap:
        result["execution_blockers"].append("qty_exceeds_max_market_order")
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    step = Decimal(str(contract["qty_step"]))
    if qty != floor_qty(qty, step):
        result["execution_blockers"].append("qty_precision")
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    costs = cost_entry_and_exit(
        book, side=side, qty=qty, stop=stop,
        taker=Decimal(str(contract["taker"])),
        contract_size=Decimal(str(contract.get("contract_size") or 1)),
    )
    if not costs.get("ok"):
        result["execution_blockers"].append(costs.get("reason") or "book_cost_failed")
        result["costs"] = costs
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    equity = Decimal(str(risk["virtual_equity_usdt"]))
    notional = Decimal(costs["entry"]["notional"])
    leverage = Decimal(str(risk["max_leverage"]))
    max_weight = Decimal(str(risk.get("max_single_weight") or risk["max_margin_fraction"]))
    margin = notional / leverage if leverage > 0 else notional
    if equity > 0 and margin / equity > max_weight:
        result["execution_blockers"].append("concentration_budget")
        result["costs"] = costs
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    budget = equity * Decimal(str(risk["risk_fraction"]))
    estimated_loss = -Decimal(costs["estimated_net_if_stopped"]) if Decimal(costs["estimated_net_if_stopped"]) < 0 else Decimal("0")
    if estimated_loss > budget:
        result["execution_blockers"].append("max_loss_budget")
        result["costs"] = costs
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    from piia.intelligence.paper.isolated_margin import apply_isolated_tier_constraints
    funding_raw = ticker.get("funding_rate")
    funding_rate = _dec(funding_raw)
    funding_known = funding_rate is not None
    margin_check = apply_isolated_tier_constraints(
        side=side, qty=qty, entry=Decimal(costs["entry"]["vwap"]), stop=stop,
        leverage=leverage, taker=Decimal(str(contract["taker"])),
        equity=equity, cash_floor=Decimal(str(risk["cash_floor"])),
        max_margin_fraction=Decimal(str(risk["max_margin_fraction"])),
        contract_size=Decimal(str(contract.get("contract_size") or 1)),
        qty_step=step, min_qty=Decimal(str(contract["min_qty"])),
        min_notional=Decimal(str(contract["min_notional"])),
        tiers=tiers or {"ok": False, "reason": "missing_new_tier_context"},
        funding_rate=funding_rate, funding_rate_known=funding_known,
        price_place=contract.get("price_place"),
        margin_mode=str(risk.get("margin_mode") or "isolated"),
        mark=_dec(ticker.get("mark")),
    )
    result["margin"] = margin_check
    constraints["margin_tiers_supported"] = bool(margin_check.get("margin_tiers_supported"))
    constraints["exact_liquidation_verified"] = False
    constraints["estimated_liquidation_price"] = (margin_check.get("liquidation") or {}).get("estimated_liquidation_price")
    constraints["tier"] = margin_check.get("tier")
    if not margin_check.get("ok"):
        result["execution_blockers"].append(margin_check.get("reason") or "isolated_margin_rejected")
        result["costs"] = costs
        result["no_plan_reason"] = "execution_context_rejected"
        return result
    sized_qty = _dec(margin_check.get("qty")) or qty
    if sized_qty < qty:
        qty = sized_qty
        costs = cost_entry_and_exit(
            book, side=side, qty=qty, stop=stop,
            taker=Decimal(str(contract["taker"])),
            contract_size=Decimal(str(contract.get("contract_size") or 1)),
        )
        if not costs.get("ok"):
            result["execution_blockers"].append(costs.get("reason") or "book_cost_failed_after_tier_resize")
            result["costs"] = costs
            result["no_plan_reason"] = "execution_context_rejected"
            return result
    # Current book already through the stop.
    best_exit = _dec(book.get("best_bid") if side == "long" else book.get("best_ask"))
    if best_exit is not None and ((side == "long" and best_exit <= stop) or (side == "short" and best_exit >= stop)):
        result["research_blockers"].append("thesis_invalidated_by_current_book")
        result["costs"] = costs
        result["no_plan_reason"] = "stale_or_non_actionable_research_not_cured_by_fresh_quote"
        result["decision"] = "structured_rejection"
        return result

    expires = now + PROPOSAL_TTL
    proposal_body = {
        "protocol": PROTOCOL,
        "kind": "unapproved_paper_proposal",
        "approved": False,
        "executed": False,
        "actionable": False,
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "stop": str(stop),
        "leverage": str(risk["max_leverage"]),
        "margin_mode": str(risk.get("margin_mode") or "isolated"),
        "margin": margin_check,
        "max_loss_budget_usdt": str(budget),
        "concentration_budget": str(max_weight),
        "fees_slippage_funding_assumptions": {
            "taker": contract.get("taker"),
            "visible_book_vwap_used": True,
            "config_slippage_assumption": str(risk.get("slippage")),
            "funding_rate": ticker.get("funding_rate"),
            "funding_is_not_a_forecast": True,
            "mark_is_not_executable": True,
        },
        "costs": costs,
        "bindings": {
            **bindings,
            "symbol": symbol,
            "risk_config_hash": risk_hash,
            "portfolio_state_hash": portfolio.get("portfolio_state_hash"),
        },
        "proposal_expires_at_utc": _iso(expires),
        "invalidates_on": [
            "price_or_book_hash_change",
            "risk_config_change",
            "portfolio_state_change",
            "frozen_research_hash_change",
            "proposal_expiry",
            "margin_tier_hash_change",
        ],
        "exact_liquidation_verified": False,
        "margin_tiers_supported": bool(margin_check.get("margin_tiers_supported")),
        "exchange_production_liquidation_validated": False,
        "future_fill_not_guaranteed": True,
    }
    proposal_body["proposal_hash"] = content_hash(json.dumps({k: v for k, v in proposal_body.items() if k != "proposal_hash"}, sort_keys=True, default=str).encode("utf-8"))
    result.update({
        "decision": "unapproved_paper_proposal",
        "proposal": proposal_body,
        "costs": costs,
        "no_plan_reason": None,
    })
    return result


def proposal_still_valid(previous: dict[str, Any], *, now: datetime, current_bindings: dict[str, Any], current_book_hash: str | None) -> dict[str, Any]:
    reasons = []
    expected_hash = content_hash(json.dumps({k: v for k, v in previous.items() if k != "proposal_hash"}, sort_keys=True, default=str).encode("utf-8"))
    if not previous.get("proposal_hash") or previous["proposal_hash"] != expected_hash:
        reasons.append("proposal_hash_missing_or_mismatch")
    if (previous.get("kind") != "unapproved_paper_proposal" or previous.get("approved") is not False
            or previous.get("executed") is not False or previous.get("actionable") is not False):
        reasons.append("proposal_status_invalid")
    try:
        expires = _aware(datetime.fromisoformat(previous["proposal_expires_at_utc"]))
        if now >= expires:
            reasons.append("proposal_expired")
    except (KeyError, TypeError, ValueError):
        reasons.append("proposal_expiry_missing")
    prev_bind = previous.get("bindings") or {}
    for key in ("request_hash", "restricted_packet_hash", "risk_config_hash", "portfolio_state_hash", "position_tiers_raw_sha256"):
        if not prev_bind.get(key) or not current_bindings.get(key):
            reasons.append(f"{key}_missing")
        elif current_bindings[key] != prev_bind[key]:
            reasons.append(f"{key}_changed")
    if not current_book_hash or not prev_bind.get("orderbook_raw_sha256"):
        reasons.append("book_hash_missing")
    elif prev_bind["orderbook_raw_sha256"] != current_book_hash:
        reasons.append("book_hash_changed")
    return {"valid": not reasons, "reasons": reasons, "approved": False}


def _synth_from_run(run_dir: Path, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]], list[str], dict[str, Any]]:
    import_path = run_dir / "import" / "result.json"
    state = _load_json(run_dir / "STATE.json") if (run_dir / "STATE.json").is_file() else {}
    packet_blockers = list(state.get("packet_recommendation_blockers") or [])
    instrument_blockers = {s: list((state.get("instrument_recommendation_blockers") or {}).get(s) or []) for s in SYMBOLS}
    synth: dict[str, Any] = {}
    if import_path.is_file() and import_path.stat().st_size:
        imported = _load_json(import_path)
        for inst in (imported.get("import") or {}).get("instruments") or []:
            symbol = inst.get("symbol")
            if not symbol:
                continue
            synth[symbol] = {
                "direction": inst.get("direction") or "flat",
                "suggested_action": inst.get("suggested_action"),
                "proposed_stop": inst.get("proposed_stop") or ((inst.get("synthesizer") or {}).get("proposed_stop") if isinstance(inst.get("synthesizer"), dict) else None),
                "thesis_id": inst.get("thesis_id"),
                "revision_id": inst.get("revision_id"),
                "research_run_id": inst.get("research_run_id"),
            }
            # import result may nest suggested fields only
    if not synth and (run_dir / "inbox" / "synthesizer.json").is_file():
        payload = _load_json(run_dir / "inbox" / "synthesizer.json")
        for symbol, item in (payload.get("instruments") or {}).items():
            synth[symbol] = {
                "direction": item.get("direction") or "flat",
                "suggested_action": item.get("suggested_action"),
                "proposed_stop": item.get("proposed_stop"),
                "thesis_expires_at_utc": item.get("thesis_expires_at_utc") or item.get("horizon_end_utc") or item.get("expires_at_utc"),
            }
    for symbol in SYMBOLS:
        synth.setdefault(symbol, {"direction": "flat", "suggested_action": "need_more_evidence", "proposed_stop": None})
    return synth, instrument_blockers, packet_blockers, state


def _role_chain_verified(run_dir: Path, state: dict[str, Any], request: dict[str, Any]) -> bool:
    """Check local workflow bindings; not proof of model independence or truth."""
    from piia.intelligence.research_cycle import _validate_role_response, _verify_cycle_binding, _verify_synthesis_request
    try:
        if state.get("phase") != "imported_unapproved":
            return False
        if state.get("request_hash") != request.get("request_hash") or state.get("cutoff_at_utc") != request.get("cutoff_at_utc"):
            return False
        if not state.get("imported_request_sha256") or content_hash((run_dir / "evidence" / "request.json").read_bytes()) != state["imported_request_sha256"]:
            return False
        result_path = run_dir / "import" / "result.json"
        if not state.get("import_result_sha256") or content_hash(result_path.read_bytes()) != state["import_result_sha256"]:
            return False
        restricted = _load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json")
        _verify_cycle_binding(state, restricted)
        researcher = _validate_role_response(_load_json(run_dir / "inbox" / "researcher.json"), role="researcher", state=state)
        critic = _validate_role_response(_load_json(run_dir / "inbox" / "critic_independent.json"), role="critic_independent", state=state)
        _validate_role_response(_load_json(run_dir / "inbox" / "synthesizer.json"), role="synthesizer", state=state)
        _verify_synthesis_request(run_dir, state, restricted, researcher, critic)
        return True
    except (ValueError, KeyError, TypeError, OSError):
        return False


def revalidate_run(
    run_dir: Path,
    *,
    now: datetime | None = None,
    books: dict[str, Any] | None = None,
    expected_frozen_hashes: dict[str, str | None] | None = None,
    fixture_label: str | None = None,
    fetch_public: bool = True,
    current_risk: dict[str, Any] | None = None,
    current_risk_hash: str | None = None,
    tiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a new execution context for a frozen run. Never rewrites frozen files."""
    run_dir = Path(run_dir)
    retrieved = _aware(now)
    before = snapshot_frozen_files(run_dir)
    request_path = run_dir / "evidence" / "request.json"
    if not request_path.is_file():
        raise FileNotFoundError("frozen request is missing")
    request = _load_json(request_path)
    state_path = run_dir / "STATE.json"
    state = _load_json(state_path) if state_path.is_file() else {}
    synth, instrument_blockers, packet_blockers, state = _synth_from_run(run_dir, request)
    packet_blockers = list(state.get("packet_recommendation_blockers") or packet_blockers)
    fixture_file = run_dir / "FIXTURE_LABEL.txt"
    labelled_fixture = (fixture_label == FIXTURE_LABEL and fixture_file.is_file()
                        and fixture_file.read_text(encoding="utf-8").strip() == FIXTURE_LABEL)
    if not labelled_fixture and not _role_chain_verified(run_dir, state, request):
        packet_blockers.append("research_role_chain_unverified")
    instrument_blockers = {s: list((state.get("instrument_recommendation_blockers") or {}).get(s) or instrument_blockers.get(s) or []) for s in SYMBOLS}

    if current_risk is None or current_risk_hash is None:
        current_risk, current_risk_hash = load_owner_risk()
    frozen_risk = request.get("paper_risk_context") or {}
    frozen_risk_hash = frozen_risk.get("owner_config_hash")
    if not frozen_risk.get("available", True if frozen_risk_hash else False):
        frozen_risk_hash = frozen_risk.get("owner_config_hash")
    portfolio = _portfolio_state(run_dir, current_risk)

    reval_id = new_id()
    out_dir = run_dir / "execution_context" / reval_id
    if books is None:
        if not fetch_public:
            books = {"ok": False, "instruments": {}, "errors": {"fetch": "missing_new_execution_context"}}
        else:
            books = fetch_execution_books(SYMBOLS, snapshot_dir=out_dir / "raw")
    if tiers is None:
        if not fetch_public:
            tiers = {"ok": False, "instruments": {}, "errors": {"fetch": "missing_new_tier_context"}}
        else:
            from piia.intelligence.public_position_tiers import fetch_position_tiers
            tiers = fetch_position_tiers(SYMBOLS, snapshot_dir=out_dir / "raw")

    bindings = {
        "research_run_id": state.get("run_id") or run_dir.name,
        "request_hash": request.get("request_hash"),
        "restricted_packet_hash": state.get("restricted_packet_hash") or (
            (_load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json") or {}).get("restricted_packet_hash")
            if (run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json").is_file() else None
        ),
        "role_file_sha256": {
            "researcher": before.get("inbox/researcher.json"),
            "critic_independent": before.get("inbox/critic_independent.json"),
            "synthesizer": before.get("inbox/synthesizer.json"),
        },
        "request_file_sha256": before.get("evidence/request.json"),
        "state_file_sha256": before.get("STATE.json"),
        "import_result_sha256": before.get("import/result.json"),
        "cutoff_at_utc": request.get("cutoff_at_utc"),
        "risk_config_hash": current_risk_hash,
        "frozen_risk_config_hash": frozen_risk_hash,
        "portfolio_state_hash": portfolio.get("portfolio_state_hash"),
    }

    instruments: dict[str, Any] = {}
    for symbol in SYMBOLS:
        reasons = research_blockers(
            request, synth.get(symbol) or {}, symbol,
            now=retrieved, frozen_hashes=before, expected_hashes=expected_frozen_hashes,
            instrument_blockers=instrument_blockers.get(symbol) or [],
            packet_blockers=packet_blockers,
        )
        inst_bind = dict(bindings)
        inst_bind["orderbook_raw_sha256"] = ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("raw_sha256")
        inst_bind["ticker_raw_sha256"] = ((books.get("instruments") or {}).get(symbol) or {}).get("ticker", {}).get("raw_sha256")
        inst_bind["position_tiers_raw_sha256"] = ((tiers.get("instruments") or {}).get(symbol) or {}).get("raw_sha256")
        if synth.get(symbol, {}).get("thesis_id"):
            inst_bind["thesis_id"] = synth[symbol]["thesis_id"]
            inst_bind["revision_id"] = synth[symbol].get("revision_id")
            inst_bind["research_run_id_imported"] = synth[symbol].get("research_run_id")
        instruments[symbol] = evaluate_symbol(
            symbol, request=request, synth_item=synth.get(symbol) or {}, books=books,
            risk=current_risk, risk_hash=current_risk_hash, frozen_risk_hash=frozen_risk_hash,
            portfolio=portfolio, now=retrieved, research_reasons=reasons, bindings=inst_bind,
            tiers=(tiers.get("instruments") or {}).get(symbol),
        )

    after = snapshot_frozen_files(run_dir)
    unchanged = before == after
    result = {
        "protocol": PROTOCOL,
        "revalidation_id": reval_id,
        "retrieved_at_utc": _iso(retrieved),
        "run_dir": str(run_dir),
        "research_run_id": bindings["research_run_id"],
        "frozen_files_unchanged": unchanged,
        "frozen_file_sha256_before": before,
        "frozen_file_sha256_after": after,
        "fresh_quote_cures_stale_research": False,
        "approved": False,
        "executed": False,
        "quote_treated_as_current_for_frozen_evidence": False,
        "existing_freshness_gate_preserved": True,
        "assumes_user_holdings": False,
        "exact_liquidation_verified": False,
        "margin_tiers_supported": bool(tiers.get("ok")),
        "exchange_production_liquidation_validated": False,
        "margin_mode_supported": "isolated_only",
        "tiers_ok": bool(tiers.get("ok")),
        "tier_errors": tiers.get("errors") or {},
        "fixture_label": fixture_label,
        "bindings": bindings,
        "portfolio_state": portfolio,
        "books_ok": bool(books.get("ok")),
        "book_errors": books.get("errors") or {},
        "instruments": instruments,
        "plans_created": [s for s, item in instruments.items() if item.get("decision") == "unapproved_paper_proposal"],
        "rejections": {s: {"research_blockers": item.get("research_blockers"), "execution_blockers": item.get("execution_blockers"), "reason": item.get("no_plan_reason")} for s, item in instruments.items() if item.get("decision") != "unapproved_paper_proposal"},
    }
    if any((item.get("proposal") or {}).get("approved") for item in instruments.values()):
        raise RuntimeError("revalidation attempted to approve a plan")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "RESULT.json", result)
    # Include a compact book as-of report without copying frozen evidence.
    write_json(out_dir / "books_summary.json", {
        "retrieved_at_utc": books.get("retrieved_at_utc") or _iso(retrieved),
        "ok": books.get("ok"),
        "instruments": {
            symbol: {
                "ticker_ok": ((books.get("instruments") or {}).get(symbol) or {}).get("ticker", {}).get("ok"),
                "ticker_observed_at_utc": ((books.get("instruments") or {}).get(symbol) or {}).get("ticker", {}).get("ticker_observed_at_utc"),
                "endpoint_request_time_utc": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("endpoint_request_time_utc"),
                "book_generated_at_utc": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("book_generated_at_utc"),
                "retrieved_at_utc": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("retrieved_at_utc"),
                "best_bid": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("best_bid"),
                "best_ask": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("best_ask"),
                "spread": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("spread"),
                "reason": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook", {}).get("reason"),
            }
            for symbol in SYMBOLS
        },
    })
    write_json(out_dir / "tiers_summary.json", {
        "retrieved_at_utc": tiers.get("retrieved_at_utc"),
        "ok": tiers.get("ok"),
        "historical_replay_supported": False,
        "exact_liquidation_verified": False,
        "instruments": {
            symbol: {
                "ok": ((tiers.get("instruments") or {}).get(symbol) or {}).get("ok"),
                "reason": ((tiers.get("instruments") or {}).get(symbol) or {}).get("reason"),
                "endpoint_request_time_utc": ((tiers.get("instruments") or {}).get(symbol) or {}).get("endpoint_request_time_utc"),
                "retrieved_at_utc": ((tiers.get("instruments") or {}).get(symbol) or {}).get("retrieved_at_utc"),
                "raw_sha256": ((tiers.get("instruments") or {}).get(symbol) or {}).get("raw_sha256"),
                "unit": ((tiers.get("instruments") or {}).get(symbol) or {}).get("unit"),
                "tier_count": ((tiers.get("instruments") or {}).get(symbol) or {}).get("tier_count"),
                "first": (((tiers.get("instruments") or {}).get(symbol) or {}).get("tiers") or [None])[0],
                "last": (((tiers.get("instruments") or {}).get(symbol) or {}).get("tiers") or [None])[-1],
            }
            for symbol in SYMBOLS
        },
    })
    if snapshot_frozen_files(run_dir) != before:
        raise RuntimeError("revalidation mutated frozen research files")
    result["output_dir"] = str(out_dir)
    return result


def list_revalidations(run_dir: Path) -> list[dict[str, Any]]:
    root = Path(run_dir) / "execution_context"
    items = []
    if not root.is_dir():
        return items
    for path in sorted(root.glob("*/RESULT.json")):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        items.append({
            "revalidation_id": payload.get("revalidation_id") or path.parent.name,
            "retrieved_at_utc": payload.get("retrieved_at_utc"),
            "plans_created": payload.get("plans_created") or [],
            "rejections": payload.get("rejections") or {},
            "frozen_files_unchanged": payload.get("frozen_files_unchanged"),
            "approved": False,
            "path": str(path),
        })
    return items
