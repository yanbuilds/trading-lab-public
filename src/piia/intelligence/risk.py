from __future__ import annotations
from datetime import datetime, timedelta
import math
from typing import Any

ENGINE_VERSION = "deterministic_risk_v2"

# Static illustrative shocks. These are example assumptions, not measured probabilities.
DEFAULT_CONFIG = {
    "max_single_weight": 0.40,
    "max_related_weight": 0.60,
    "cash_floor": 0.10,
    "stale_price_hours": 36,
    "related_groups": [["btc-usd-spot", "mstr-us-equity"]],
    "scenario_shocks": {
        "illustrative_joint_drawdown": {
            "btc-usd-spot": -0.30,
            "mstr-us-equity": -0.45,
            "xau-usd-spot": -0.15,
        }
    },
    "scenario_disclaimer": "Static example assumptions only; not measured probabilities.",
    "flat_threshold": 0.005,
    "plan_ttl_hours": 24,
}

def position_equity(positions: list[dict], cash: float, liabilities: float) -> tuple[float | None, list[str]]:
    issues: list[str] = []
    if not math.isfinite(cash) or not math.isfinite(liabilities) or cash < 0 or liabilities < 0:
        return None, ["invalid cash or liabilities"]
    nav = cash - liabilities
    for p in positions:
        qty = p.get("quantity")
        mark = p.get("mark")
        if qty is None or mark is None:
            issues.append(f"unknown mark or quantity for {p.get('instrument_id')}")
            continue
        try:
            qty, mark = float(qty), float(mark)
        except (TypeError, ValueError, OverflowError):
            issues.append(f"invalid mark or quantity for {p.get('instrument_id')}")
            continue
        if not math.isfinite(qty) or not math.isfinite(mark) or mark <= 0:
            issues.append(f"invalid mark or quantity for {p.get('instrument_id')}")
            continue
        if qty != 0 and p.get("instrument_type") != "spot":
            issues.append(f"unsupported or unknown position type for {p.get('instrument_id')}")
            continue
        if qty < 0:
            issues.append(f"short position accounting unsupported for {p.get('instrument_id')}")
            continue
        nav += qty * mark
        if not math.isfinite(nav):
            return None, ["non-finite portfolio equity"]
    if issues:
        return None, issues
    return nav, issues

def is_stale(mark_as_of: datetime | None, now: datetime, hours: float) -> bool:
    if mark_as_of is None or mark_as_of.tzinfo is None or mark_as_of.utcoffset() is None:
        return True
    age = now - mark_as_of
    return age < timedelta(0) or age > timedelta(hours=hours)

def evaluate(
    *,
    portfolio: dict[str, Any],
    config: dict[str, Any],
    plan_request: dict[str, Any],
    now: datetime,
    latest_revision_id: str,
    bound_revision_id: str,
) -> dict[str, Any]:
    """Deterministic, fail-closed risk check. No LLM involvement."""
    reasons: list[str] = []
    flags = {
        "concentration": "unknown",
        "leverage": 0.0,
        "liquidity": "unknown",
        "downside": "unknown",
    }
    cfg = {**DEFAULT_CONFIG, **config}
    try:
        thresholds = [float(cfg[k]) for k in ("max_single_weight", "max_related_weight", "cash_floor", "stale_price_hours")]
        leverage = float(plan_request.get("leverage") if plan_request.get("leverage") is not None else 1.0)
        if not all(math.isfinite(v) and v >= 0 for v in thresholds) or not math.isfinite(leverage) or leverage <= 0:
            raise ValueError("invalid risk limit or leverage")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("timezone-aware current time required")
    except (TypeError, ValueError, OverflowError):
        flags["concentration"] = "fail_closed"
        return _result(False, ["invalid risk limits, leverage, or current time"], flags, None, cfg, {})
    positions = list(portfolio.get("positions") or [])
    try:
        cash = float(portfolio.get("cash") or 0)
        liabilities = float(portfolio.get("liabilities") or 0)
    except (TypeError, ValueError, OverflowError):
        reasons.append("invalid cash or liabilities")
        flags["concentration"] = "fail_closed"
        return _result(False, reasons, flags, None, cfg, {})
    equity, missing = position_equity(positions, cash, liabilities)
    if equity is None or equity <= 0:
        reasons.append("unknown or non-positive equity: " + "; ".join(missing) if missing else "unknown or non-positive equity")
        flags["concentration"] = "fail_closed"
        return _result(False, reasons, flags, equity, cfg, {})

    stale_h = float(cfg["stale_price_hours"])
    weights: dict[str, float] = {}
    liquidity_notes: list[str] = []
    for p in positions:
        iid = p["instrument_id"]
        mark_as_of = p.get("mark_as_of")
        if isinstance(mark_as_of, str):
            try:
                mark_as_of = datetime.fromisoformat(mark_as_of)
            except ValueError:
                mark_as_of = None
        if mark_as_of is not None and not isinstance(mark_as_of, datetime):
            mark_as_of = None
        if is_stale(mark_as_of, now, stale_h):
            reasons.append(f"stale or missing mark for {iid}")
        value = float(p["quantity"]) * float(p["mark"])
        weights[iid] = weights.get(iid, 0) + value / equity
        liq = p.get("liquidity") or "unknown"
        if liq == "unknown":
            liquidity_notes.append(iid)
    flags["liquidity"] = "unknown" if liquidity_notes else "stated"

    max_w = max(weights.values()) if weights else 0.0
    flags["concentration"] = max_w
    if max_w > float(cfg["max_single_weight"]) + 1e-12:
        reasons.append(f"single-asset concentration {max_w:.4f} exceeds {cfg['max_single_weight']}")

    for group in cfg.get("related_groups") or []:
        g = sum(weights.get(i, 0) for i in group)
        if g > float(cfg["max_related_weight"]) + 1e-12:
            reasons.append(f"related-asset concentration {g:.4f} for {group} exceeds {cfg['max_related_weight']}")

    requested_leverage = leverage
    instrument_type = (plan_request.get("instrument_type") or plan_request.get("market_type") or "spot")
    margin_mode = plan_request.get("margin_mode")
    flags["leverage"] = requested_leverage
    flags["margin_mode"] = margin_mode
    flags["side"] = plan_request.get("side") or "long"
    # Schema is futures-ready (side, leverage, margin_mode, stop/exit). Phase 1 does not
    # implement an exact margin or liquidation model, so leveraged/perp executable
    # paper plans fail closed with supported=false rather than a fake fill.
    needs_margin_model = (
        requested_leverage > 1.0 + 1e-12
        or instrument_type in {"perp", "perpetual", "future", "futures"}
        or bool(plan_request.get("margin"))
        or margin_mode in {"isolated", "cross"}
    )
    flags["margin_model_supported"] = False
    flags["liquidation_model_supported"] = False
    if needs_margin_model:
        reasons.append(
            "exact margin and liquidation model unsupported: leveraged, margined, or perpetual/futures executable plans are blocked (supported=false), not simulated"
        )
    if flags["side"] == "short":
        reasons.append("short position accounting unsupported for executable plans")

    cash_weight = cash / equity if equity else 0
    if cash_weight < float(cfg["cash_floor"]) - 1e-12:
        reasons.append(f"cash weight {cash_weight:.4f} below floor {cfg['cash_floor']}")

    if bound_revision_id != latest_revision_id:
        reasons.append("plan is bound to a superseded thesis revision")

    # Illustrative scenario losses — not probabilities.
    scenario_losses: dict[str, float] = {}
    shocks = cfg.get("scenario_shocks") or {}
    for name, shock_map in shocks.items():
        loss = 0.0
        for iid, w in weights.items():
            shock = float(shock_map.get(iid, 0))
            loss += w * shock
        scenario_losses[name] = loss
    if scenario_losses:
        flags["downside"] = {
            "losses_as_equity_fraction": scenario_losses,
            "disclaimer": cfg.get("scenario_disclaimer"),
        }

    if plan_request.get("require_fresh_evidence") and plan_request.get("missing_critical_evidence"):
        reasons.append("missing or stale critical evidence blocks an actionable paper plan")

    ok = not reasons
    return _result(ok, reasons, flags, equity, cfg, scenario_losses)

def _result(ok, reasons, flags, equity, cfg, scenario_losses):
    return {
        "passed": ok,
        "actionable": ok,
        "block_reasons": reasons,
        "equity": equity,
        "flags": flags,
        "scenario_losses": scenario_losses,
        "scenario_disclaimer": cfg.get("scenario_disclaimer"),
        "engine": ENGINE_VERSION,
        "llm_mutable": False,
    }
