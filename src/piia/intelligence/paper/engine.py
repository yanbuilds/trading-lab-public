from __future__ import annotations
from datetime import datetime, timedelta
import math
from typing import Any
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from trading_lab.perps.risk import Spec, exit_pnl, floor_step, size_order, stop_fill

from piia.intelligence.clock import Clock
from piia.intelligence.errors import Conflict, NotFound, ValidationFailed
from piia.intelligence.ids import new_id
from piia.intelligence.models import (
    Decision, KillSwitchEvent, PaperAccount, PaperLedger, PaperOrder, PaperPosition, Plan,
    ResearchRun, ThesisRevision,
)
from piia.intelligence.paper.risk_file import load_owner_risk
from piia.intelligence.paper.specs import CONTRACTS
from piia.intelligence.risk import ENGINE_VERSION

REGISTRY = {
    "local_simplified": {
        "supported": True,
        "live_orders": False,
        "exchange_demo_orders": False,
        "liquidation": "approximate_stress_not_venue_verified",
        "margin_model_supported": False,
    },
    "exchange_live_verified": {
        "supported": False,
        "live_orders": False,
        "exchange_demo_orders": False,
        "liquidation": "unverified",
        "margin_model_supported": False,
    },
}

def spec_for(symbol: str) -> Spec:
    c = CONTRACTS[symbol]
    return Spec(symbol=symbol, min_qty=c["min_qty"], qty_step=c["qty_step"],
                min_notional=c["min_notional"], taker=c["taker"], contract_size=c["contract_size"])

class PaperEngine:
    def __init__(self, session: Session, clock: Clock | None = None):
        self.session = session
        self.clock = clock or Clock()

    def registry(self) -> dict:
        return REGISTRY

    def kill_enabled(self) -> bool:
        row = self.session.scalars(select(KillSwitchEvent).order_by(KillSwitchEvent.created_at.desc())).first()
        return bool(row and row.enabled)

    def set_kill(self, actor: str, enabled: bool, reason: str) -> dict:
        count = int(self.session.scalar(select(func.count()).select_from(KillSwitchEvent)) or 0)
        ev = KillSwitchEvent(id=new_id(), created_at=self.clock.now() + timedelta(microseconds=count + 1), actor=actor, enabled=enabled, reason=reason)
        self.session.add(ev)
        self.session.commit()
        return {"id": ev.id, "enabled": ev.enabled, "reason": ev.reason, "created_at": ev.created_at.isoformat()}

    def open_account(self, starting_cash: float | None = None, label: str = "LABELED_FIXTURE_PAPER") -> dict:
        cfg, _ = load_owner_risk()
        cash = float(starting_cash if starting_cash is not None else cfg["virtual_equity_usdt"])
        row = PaperAccount(
            id=new_id(), created_at=self.clock.now(), label=label,
            starting_cash=cash, cash=cash, liabilities=0, realized_pnl=0,
            engine="local_simplified",
            notes="Local simplified paper sandbox. Not a live account and not real performance.",
        )
        self.session.add(row)
        self._ledger(row, "open_account", cash, "initial virtual cash", provenance="owner_config")
        self.session.commit()
        return self.account_dict(row)

    def account_dict(self, row: PaperAccount) -> dict:
        pos = list(self.session.scalars(select(PaperPosition).where(PaperPosition.account_id==row.id, PaperPosition.status=="open")))
        closed = list(self.session.scalars(select(PaperPosition).where(PaperPosition.account_id==row.id, PaperPosition.status=="closed")))
        closed_nets = [self._position_net(p) for p in closed]
        realized = sum(v for v in closed_nets if v is not None) if all(v is not None for v in closed_nets) else None
        return {
            "id": row.id, "label": row.label, "engine": row.engine,
            "starting_cash": row.starting_cash, "cash": row.cash, "liabilities": row.liabilities,
            "realized_pnl": realized, "realized_pnl_legacy_exit_only": row.realized_pnl,
            "realized_pnl_basis": "closed positions, entry and exit fees, and recorded funding only; funding completeness unverified",
            "notes": row.notes,
            "sandbox": True, "live_orders": False,
            "open_positions": [self._pos_dict(p) for p in pos],
            "kill_switch": self.kill_enabled(),
            "never_real_performance": True,
        }

    def _pos_dict(self, p: PaperPosition) -> dict:
        return {
            "id": p.id, "symbol": p.symbol, "instrument_id": p.instrument_id, "side": p.side,
            "qty": p.qty, "entry": p.entry, "stop_loss": p.stop_loss, "leverage": p.leverage,
            "margin": p.margin, "status": p.status, "opened_at": p.opened_at.isoformat(),
            "closed_at": p.closed_at.isoformat() if p.closed_at else None,
            "realized_pnl": self._position_net(p) if p.status == "closed" else None,
            "realized_pnl_legacy_exit_only": p.realized_pnl,
        }

    def _position_net(self, p: PaperPosition) -> float | None:
        """Derive closed net PnL from the legacy exit amount and recorded costs.

        Existing database columns retain their original meaning so older paper
        accounts need no destructive rewrite. Missing entry orders fail closed.
        """
        if p.status != "closed" or p.realized_pnl is None:
            return None
        entry = self.session.get(PaperOrder, p.entry_order_id)
        if not entry or entry.account_id != p.account_id or entry.status != "filled":
            return None
        funding = self.session.scalars(select(PaperLedger.amount).where(
            PaperLedger.account_id == p.account_id, PaperLedger.kind == "funding", PaperLedger.ref_id == p.id)).all()
        return p.realized_pnl - entry.fee + sum(funding)

    def _ledger(self, account: PaperAccount, kind: str, amount: float, note: str, ref_id: str | None = None, provenance: str = "assumed"):
        self.session.add(PaperLedger(
            id=new_id(), created_at=self.clock.now(), account_id=account.id, kind=kind,
            amount=amount, cash_after=account.cash, note=note, ref_id=ref_id, provenance=provenance,
        ))

    def get_account(self, account_id: str) -> PaperAccount:
        row = self.session.get(PaperAccount, account_id)
        if not row:
            raise NotFound(f"paper account not found: {account_id}")
        return row

    def execute_approved(self, *, account_id: str, decision_id: str, idempotency_key: str | None = None) -> dict:
        """Fill a previously approved local paper plan. Live exchange is impossible here."""
        existing = self.session.scalars(select(PaperOrder).where(PaperOrder.decision_id==decision_id, PaperOrder.reduce_only.is_(False))).first()
        if existing and existing.status == "filled":
            return {"id": existing.id, "status": existing.status, "idempotent": True, "reason": existing.reason, "sandbox": True}
        dec = self.session.get(Decision, decision_id)
        if not dec:
            raise NotFound("decision not found")
        if not dec.valid or dec.user_choice != "approve":
            raise ValidationFailed("decision is not a valid approval")
        plan = self.session.get(Plan, dec.plan_id)
        if not plan or not plan.actionable:
            raise ValidationFailed("plan is not actionable paper")
        if (plan.risk_result_json or {}).get("engine") != ENGINE_VERSION:
            raise ValidationFailed("risk engine version changed; create a new plan")
        if plan.margin_model_supported or plan.liquidation_model_supported:
            raise ValidationFailed("refusing to treat an unverified venue model as supported")
        req = plan.request_json or {}
        if req.get("execution_registry") != "local_simplified" or req.get("live_orders"):
            raise ValidationFailed("only local_simplified paper execution is allowed")
        account = self.get_account(account_id)
        return self._open(account, plan, dec, req)

    def _open(self, account: PaperAccount, plan: Plan, dec: Decision, req: dict) -> dict:
        if self.kill_enabled():
            return self._reject_order(account, plan, dec, req, "kill_switch_enabled")
        open_pos = self.session.scalars(select(PaperPosition).where(PaperPosition.account_id==account.id, PaperPosition.status=="open")).first()
        if open_pos:
            return self._reject_order(account, plan, dec, req, "max_open_positions")
        symbol = req["contract_symbol"]
        if symbol not in CONTRACTS:
            return self._reject_order(account, plan, dec, req, "unknown_symbol")
        cfg, cfg_hash = load_owner_risk()
        if cfg_hash != req.get("owner_risk_hash"):
            return self._reject_order(account, plan, dec, req, "owner_risk_file_changed")
        price = float(req["entry_price"])
        stop = req.get("stop_loss")
        if stop is None:
            return self._reject_order(account, plan, dec, req, "stop_required")
        stop = float(stop)
        leverage = float(req.get("leverage") or 1)
        if leverage > float(cfg["max_leverage"]) + 1e-12:
            return self._reject_order(account, plan, dec, req, "leverage_exceeds_owner_limit")
        side = req.get("side") or "long"
        side_i = 1 if side == "long" else -1
        stop_distance = abs(price - stop)
        if stop_distance <= 0 or (side == "long" and stop >= price) or (side == "short" and stop <= price):
            return self._reject_order(account, plan, dec, req, "invalid_stop")
        if account.cash < 0:
            return self._reject_order(account, plan, dec, req, "negative_cash")
        equity = account.cash  # no mark-to-market of other positions because only one is allowed
        spec = spec_for(symbol)
        qty, reason = size_order(
            equity, account.cash, price, stop_distance, spec,
            leverage=leverage, risk_fraction=float(cfg["risk_fraction"]),
            cash_floor=float(cfg["cash_floor"]), max_margin_fraction=float(cfg["max_margin_fraction"]),
            slippage=float(cfg["slippage"]),
        )
        if qty <= 0:
            return self._reject_order(account, plan, dec, req, reason)
        if req.get("max_qty") is not None:
            try:
                cap = floor_step(float(req["max_qty"]), spec.qty_step)
            except (TypeError, ValueError):
                return self._reject_order(account, plan, dec, req, "invalid_max_qty")
            if cap < qty:
                qty = cap
            if qty < spec.min_qty or qty * price * spec.contract_size < spec.min_notional:
                return self._reject_order(account, plan, dec, req, "minimum_order_exceeds_additional_risk_reserve")
        slip = float(cfg["slippage"])
        fill = price * (1 + slip) if side == "long" else price * (1 - slip)
        notional = qty * fill * spec.contract_size
        fee = notional * spec.taker
        margin = notional / leverage
        if account.cash - (margin + fee) + 1e-12 < equity * float(cfg["cash_floor"]):
            return self._reject_order(account, plan, dec, req, "cash_floor")
        if account.cash < margin + fee:
            return self._reject_order(account, plan, dec, req, "insufficient_cash")
        now = self.clock.now()
        order = PaperOrder(
            id=new_id(), created_at=now, account_id=account.id, plan_id=plan.id,
            decision_id=dec.id, research_run_id=req.get("research_run_id"),
            symbol=symbol, side=side, qty=qty, price=fill, leverage=leverage, fee=fee,
            stop_loss=stop, reduce_only=False, status="filled", reason="ok", sandbox=True,
        )
        self.session.add(order)
        account.cash = account.cash - margin - fee
        self._ledger(account, "margin", -margin, f"reserve margin {symbol}", order.id, "local_simplified")
        self._ledger(account, "fee", -fee, "taker fee on entry", order.id, "public_taker_rate")
        pos = PaperPosition(
            id=new_id(), account_id=account.id, opened_at=now, symbol=symbol,
            instrument_id=CONTRACTS[symbol]["instrument_id"], side=side, qty=qty, entry=fill,
            stop_loss=stop, leverage=leverage, margin=margin, status="open",
            entry_order_id=order.id,
        )
        self.session.add(pos)
        self.session.commit()
        return {"id": order.id, "status": "filled", "sandbox": True, "position": self._pos_dict(pos),
                "account": self.account_dict(account), "fee": fee, "fill_price": fill, "qty": qty,
                "engine": "local_simplified", "margin_model_supported": False,
                "liquidation_model": "approximate_stress_not_venue_verified", "idempotent": False}

    def _reject_order(self, account, plan, dec, req, reason: str) -> dict:
        order = PaperOrder(
            id=new_id(), created_at=self.clock.now(), account_id=account.id,
            plan_id=getattr(plan, "id", None), decision_id=getattr(dec, "id", None),
            research_run_id=req.get("research_run_id"), symbol=req.get("contract_symbol") or "UNKNOWN",
            side=req.get("side") or "long", qty=0, price=float(req.get("entry_price") or 0),
            leverage=float(req.get("leverage") or 1), fee=0, stop_loss=req.get("stop_loss"),
            reduce_only=False, status="rejected", reason=reason, sandbox=True,
        )
        self.session.add(order)
        self.session.commit()
        return {"id": order.id, "status": "rejected", "reason": reason, "sandbox": True, "qty": 0}

    def _close(self, pos: PaperPosition, exit_price: float, reason: str, reduce_only: bool, candle: dict | None = None) -> dict:
        account = self.get_account(pos.account_id)
        spec = spec_for(pos.symbol)
        cfg, _ = load_owner_risk()
        side_i = 1 if pos.side == "long" else -1
        # Conservative: if gap already applied by stop_fill, still charge taker on notional.
        notional = pos.qty * exit_price * spec.contract_size
        fee = notional * spec.taker
        gross = exit_pnl(side_i, pos.qty, pos.entry, exit_price, spec)
        net = gross - fee
        now = self.clock.now()
        order = PaperOrder(
            id=new_id(), created_at=now, account_id=account.id, plan_id=None, decision_id=None,
            research_run_id=None, symbol=pos.symbol, side="sell" if pos.side == "long" else "buy",
            qty=pos.qty, price=exit_price, leverage=pos.leverage, fee=fee, stop_loss=pos.stop_loss,
            reduce_only=True, status="filled", reason=reason, sandbox=True,
        )
        self.session.add(order)
        account.cash = account.cash + pos.margin + net
        account.realized_pnl = account.realized_pnl + net
        self._ledger(account, "margin_release", pos.margin, "release reserved margin", order.id, "local_simplified")
        self._ledger(account, "pnl", net, f"{reason} net of exit fee", order.id, "local_simplified")
        pos.status = "closed"
        pos.closed_at = now
        pos.exit_order_id = order.id
        pos.realized_pnl = net
        self.session.commit()
        trade_net = self._position_net(pos)
        return {
            "status": "closed", "filled": True, "reason": reason, "exit_price": exit_price,
            "fee": fee, "gross_pnl": gross, "net_pnl": trade_net,
            "net_pnl_basis": "entry and exit fees and recorded funding only; funding completeness unverified",
            "net_pnl_before_entry_fee_and_funding": net, "reduce_only": True,
            "simulated": True, "exchange_stop": False, "sandbox": True,
            "position": self._pos_dict(pos), "account": self.account_dict(account),
        }

    def apply_funding(self, position_id: str, rate: float, mark_price: float, *, provenance: str = "bar_open_mark_proxy") -> dict:
        """Apply a historical funding payment. Mark is the disclosed proxy, not a verified venue mark."""
        try:
            if isinstance(rate, bool) or isinstance(mark_price, bool):
                raise ValueError
            rate, mark_price = float(rate), float(mark_price)
            if not math.isfinite(rate) or not math.isfinite(mark_price) or mark_price <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationFailed("funding rate and positive mark must be finite") from exc
        pos = self.session.get(PaperPosition, position_id)
        if not pos or pos.status != "open":
            raise NotFound("open position not found")
        account = self.get_account(pos.account_id)
        spec = spec_for(pos.symbol)
        side_i = 1 if pos.side == "long" else -1
        amount = side_i * pos.qty * spec.contract_size * mark_price * rate
        if not math.isfinite(amount):
            raise ValidationFailed("funding amount is not finite")
        account.cash = account.cash - amount
        self._ledger(account, "funding", -amount, f"funding rate={rate} mark_proxy={mark_price}", pos.id, provenance)
        self.session.commit()
        return {"position_id": pos.id, "amount_paid": amount, "cash": account.cash, "rate": rate, "mark_proxy": mark_price, "provenance": provenance}

    def exit_position(self, position_id: str, price: float, reason: str, *, slippage: float = 0.0) -> dict:
        pos = self.session.get(PaperPosition, position_id)
        if not pos or pos.status != "open":
            raise NotFound("open position not found")
        side_i = 1 if pos.side == "long" else -1
        fill = float(price) * (1 - side_i * float(slippage or 0))
        return self._close(pos, fill, reason=reason, reduce_only=True)

    def check_stop(self, position_id: str, candle: dict, *, slippage: float = 0.0, reason: str | None = None) -> dict:
        """On-demand next-bar / gap stop check. Simulated, not an exchange stop."""
        pos = self.session.get(PaperPosition, position_id)
        if not pos or pos.status != "open":
            raise NotFound("open position not found")
        if pos.stop_loss is None:
            raise ValidationFailed("position has no stop")
        o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
        side_i = 1 if pos.side == "long" else -1
        fill = stop_fill(side_i, pos.stop_loss, o, l, h)
        if fill is None:
            return {"status": "open", "filled": False, "position_id": pos.id, "simulated": True}
        if slippage:
            fill = fill * (1 - side_i * float(slippage))
        return self._close(pos, fill, reason=reason or "stop_or_gap", reduce_only=True, candle=candle)
