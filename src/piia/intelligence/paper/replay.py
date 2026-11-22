"""Finite HISTORICAL_REPLAY adapter for the local paper engine.

Uses only then-known closed 1h bars. Does not loosen live API guards, does not
touch the prospective research database, and is not a contemporaneous prediction.
The locked dataset has already been seen in earlier perps research; this run is
exploratory, not a held-out test.

Contract min-qty and fee specs are the latest snapshot in paper.specs, applied
as a constant retrospective assumption because archival venue specs were not
available. Mark-price funding uses the contemporaneous bar open as a proxy, not
a verified venue mark. Liquidation remains approximate and unverified.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from piia.intelligence.clock import FrozenClock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.hashing import content_hash
from piia.intelligence.models import PaperPosition
from piia.intelligence.paper.engine import PaperEngine, spec_for
from piia.intelligence.paper.risk_file import load_owner_risk
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.schemas import ApprovalIn, EvidenceIn, ThesisCreate
from piia.intelligence.service import IntelligenceService
from trading_lab.perps.risk import exit_pnl

ACTOR = "sandbox_test_harness"
LABEL = "HISTORICAL_REPLAY"
SYMBOL = "BTCUSDT"
INSTRUMENT_ID = "btcusdt-bitget-perp"
SLIPPAGE = 0.0005
LEVERAGE = 2.0
WARMUP = 60
DAILY_LOSS_HALT = 0.03
TOTAL_LOSS_HALT = 0.10
WIN_EPS = 0.01
STRATEGY = (
    "1h EMA20/50 cross long/short on a fully closed bar; fill at the next bar open; "
    "frozen 2*ATR14 stop from the signal close; reverse-cross exits at the next open; "
    "stops precede reverse exits; no take-profit; no parameter search."
)
ADAPTER_DIFFERENCES = (
    "PIIA engine sizes from the plan entry price then applies configured slippage on the fill; "
    "this replay sets the plan entry to the then-known next-open (unslipped) and the stop to "
    "that open +/- 2*ATR14. The earlier offline trading_lab.perps.research.simulate sized and "
    "placed the stop on an already-slipped price. Exit slippage is applied by the replay adapter "
    "before the engine charges the taker fee. PIIA reserves margin in cash while a position is "
    "open; with max one position this does not change terminal cash versus the offline book. "
    "Loss-halt (3% day / 10% total) is applied by the replay adapter, not by a venue kill switch."
)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """TA-Lib compatible EMA: SMA seed, then standard EMA."""
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return out
    out[period - 1] = float(np.mean(values[:period]))
    alpha = 2.0 / (period + 1)
    for i in range(period, len(values)):
        out[i] = alpha * float(values[i]) + (1 - alpha) * out[i - 1]
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """TA-Lib compatible ATR: first value is the mean of TR[1:period+1], then Wilder."""
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    tr = np.zeros(n, dtype=float)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    if n <= period:
        return out
    out[period] = float(np.mean(tr[1 : period + 1]))
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def dataset_hash(csv_path: Path, funding_path: Path | None) -> str:
    h = hashlib.sha256()
    h.update(csv_path.read_bytes())
    if funding_path and funding_path.exists():
        h.update(b"\n")
        h.update(funding_path.read_bytes())
    return h.hexdigest()


def load_btc_frame(data_dir: Path) -> tuple[pd.DataFrame, dict[pd.Timestamp, float], dict[str, Any]]:
    csv_path = data_dir / "BTCUSDT.csv"
    fund_path = data_dir / "BTCUSDT-funding.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("empty BTCUSDT cache")
    gaps = df["date"].diff().dropna()
    if (gaps != pd.Timedelta(hours=1)).any():
        raise ValueError("BTCUSDT cache is not a continuous 1h series")
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["atr14"] = atr(high, low, close, 14)
    signal = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        a0, b0 = df.at[i - 1, "ema20"], df.at[i - 1, "ema50"]
        a1, b1 = df.at[i, "ema20"], df.at[i, "ema50"]
        if not np.isfinite([a0, b0, a1, b1]).all():
            continue
        if a1 > b1 and a0 <= b0:
            signal[i] = 1
        elif a1 < b1 and a0 >= b0:
            signal[i] = -1
    df["signal"] = signal
    fmap: dict[pd.Timestamp, float] = {}
    if fund_path.exists():
        fd = pd.read_csv(fund_path)
        for _, row in fd.iterrows():
            try:
                timestamp, rate = row["timestamp"], float(row["rate"])
                if not math.isfinite(float(timestamp)) or int(timestamp) != float(timestamp) or not math.isfinite(rate):
                    raise ValueError("invalid funding value")
                ts = pd.Timestamp(int(timestamp), unit="ms", tz="UTC")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("invalid funding record") from exc
            if ts in fmap:
                raise ValueError("duplicate funding timestamp")
            fmap[ts] = rate
    meta = {
        "csv_path": str(csv_path),
        "funding_path": str(fund_path) if fund_path.exists() else None,
        "dataset_hash": dataset_hash(csv_path, fund_path if fund_path.exists() else None),
        "bars": int(len(df)),
        "first": str(df["date"].iloc[0]),
        "last": str(df["date"].iloc[-1]),
        "funding_count": len(fmap),
        "funding_complete": None,
        "funding_file_present": fund_path.exists(),
        "funding_method": "historical rate * contemporaneous bar open as mark proxy; not a verified venue mark",
        "spec_assumption": "latest local paper.specs snapshot applied as a constant; archival Bitget specs unavailable",
        "already_seen": True,
        "held_out": False,
        "label": LABEL,
    }
    return df, fmap, meta


def _side_name(side_i: int) -> str:
    return "long" if side_i == 1 else "short"


@dataclass
class ReplayTrade:
    side: str
    signal_bar_open: str
    entry_time: str
    exit_time: str | None
    entry: float
    exit: float | None
    stop: float
    qty: float
    net_usd: float | None
    exit_reason: str | None
    entry_fee: float
    exit_fee: float | None
    funding_paid: float
    thesis_id: str
    plan_id: str
    decision_id: str
    entry_order_id: str
    position_id: str
    exit_order_id: str | None = None
    sandbox_approval_actor: str = ACTOR

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ReplayResult:
    report: dict[str, Any]
    trades: list[ReplayTrade] = field(default_factory=list)


def _equity(paper: PaperEngine, session, account_id: str, pos_id: str | None, mark: float, spec) -> float:
    account = paper.get_account(account_id)
    total = account.cash
    if pos_id:
        row = session.get(PaperPosition, pos_id)
        if row and row.status == "open":
            side_i = 1 if row.side == "long" else -1
            total = account.cash + row.margin + exit_pnl(side_i, row.qty, row.entry, mark, spec)
    return total


def run_btc_historical_replay(
    *,
    db_path: Path,
    evidence_store: Path,
    data_dir: Path,
    starting_cash: float | None = None,
    end_time: datetime | None = None,
    current_quote: float | None = None,
    run_recorded_at: datetime | None = None,
) -> ReplayResult:
    """Replay locked BTC 1h bars through the real paper fill/stop engine.

    current_quote, if supplied, is recorded as provided-but-unused and must not
    enter sizing, stops, or marks. end_time cuts the series without using later bars.
    """
    unused_quote = current_quote
    run_recorded_at = run_recorded_at or datetime.now(timezone.utc)
    df, fmap, meta = load_btc_frame(data_dir)
    if end_time is not None:
        end_ts = pd.Timestamp(end_time)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        df = df[df["date"] <= end_ts].reset_index(drop=True)
        fmap = {k: v for k, v in fmap.items() if k <= end_ts}
        meta = {**meta, "end_time_cutoff": str(end_ts), "bars_used": int(len(df))}

    if db_path.exists():
        raise FileExistsError(f"replay database already exists; choose a new path: {db_path}")

    if len(df) <= WARMUP:
        raise ValueError("not enough bars for warmup")
    cfg, _cfg_hash = load_owner_risk()
    starting_cash = float(cfg["virtual_equity_usdt"] if starting_cash is None else starting_cash)
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting cash must be finite and positive")
    from trading_lab.perps.compare import funding_complete_between
    records = [{"timestamp": int(t.value // 1_000_000), "rate": rate} for t, rate in fmap.items()]
    coverage = funding_complete_between({SYMBOL: records}, {SYMBOL: 8}, df["date"].iloc[WARMUP], df["date"].iloc[-1])
    if not meta["funding_file_present"] or not coverage["complete"]:
        raise ValueError("funding data missing, incomplete or invalid; replay returns withheld")
    meta["funding_complete"] = True
    meta["funding_coverage_under_current_interval_assumption"] = coverage
    meta["funding_interval_limitation"] = "8-hour BTC schedule assumed throughout; historical changes unverified"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_store.mkdir(parents=True, exist_ok=True)
    clock = FrozenClock(df["date"].iloc[WARMUP].to_pydatetime())
    engine = make_engine(f"sqlite:///{db_path.as_posix()}")
    create_all(engine)
    session = session_factory(engine)()
    try:
        core = IntelligenceService(session, clock=clock, evidence_store=evidence_store)
        core.seed_instruments()
        ext = ExtendedService(core)
        paper: PaperEngine = ext.paper
        acc = paper.open_account(starting_cash=starting_cash, label="HISTORICAL_REPLAY_SANDBOX")
        account_id = acc["id"]
        spec = spec_for(SYMBOL)

        trades: list[ReplayTrade] = []
        rejections: Counter[str] = Counter()
        signals_observed = int((df["signal"].iloc[WARMUP - 1 : -1] != 0).sum()) if len(df) > WARMUP else 0
        signals_seen = 0
        nav: list[tuple[datetime, float]] = []
        funding_net_paid = 0.0
        fees = 0.0
        open_trade: ReplayTrade | None = None
        pos_id: str | None = None
        kill = False
        day = None
        day_start = starting_cash
        last_ts = None
        stress_breaches = 0

        def record_exit(result: dict, when: datetime, reason: str) -> None:
            nonlocal pos_id, open_trade, fees, stress_breaches
            if not result.get("filled"):
                return
            fees += float(result["fee"])
            gross = float(result["gross_pnl"])
            if open_trade:
                closed = session.get(PaperPosition, open_trade.position_id)
                net = gross - open_trade.entry_fee - float(result["fee"]) - open_trade.funding_paid
                open_trade.exit = float(result["exit_price"])
                open_trade.exit_time = when.isoformat()
                open_trade.exit_reason = reason
                open_trade.exit_fee = float(result["fee"])
                open_trade.exit_order_id = closed.exit_order_id if closed else None
                if closed and closed.margin is not None and (-gross + float(result["fee"])) > closed.margin + 1e-12:
                    stress_breaches += 1
                open_trade.net_usd = net
            pos_id = None
            open_trade = None

        for i in range(len(df)):
            ts: pd.Timestamp = df.at[i, "date"]
            if i < WARMUP:
                continue
            if last_ts is not None and ts - last_ts != pd.Timedelta(hours=1):
                raise ValueError("stale or gapped replay timeline")
            last_ts = ts
            clock.set(ts.to_pydatetime())
            if ts.date() != day:
                day = ts.date()
                day_start = _equity(paper, session, account_id, pos_id, float(df.at[i, "open"]), spec)

            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                if pos and pos.status == "open":
                    rate = fmap.get(ts, 0.0)
                    if rate:
                        paid = paper.apply_funding(pos_id, rate, float(df.at[i, "open"]))
                        amount = float(paid["amount_paid"])
                        funding_net_paid += amount
                        if open_trade:
                            open_trade.funding_paid += amount
                        if paper.get_account(account_id).cash < 0:
                            raise ValueError("Funding insolvency; exact liquidation model required; returns not published")
                    pos = session.get(PaperPosition, pos_id)
                    side_i = 1 if pos.side == "long" else -1
                    o = float(df.at[i, "open"])
                    h = float(df.at[i, "high"])
                    l = float(df.at[i, "low"])
                    c = float(df.at[i, "close"])
                    gap = (side_i == 1 and o <= pos.stop_loss) or (side_i == -1 and o >= pos.stop_loss)
                    if gap:
                        result = paper.check_stop(
                            pos_id, {"open": o, "high": h, "low": l, "close": c},
                            slippage=SLIPPAGE, reason="gap_stop",
                        )
                        record_exit(result, ts.to_pydatetime(), "gap_stop")
                    else:
                        prev = df.iloc[i - 1]
                        reversed_ = (side_i == 1 and prev.ema20 < prev.ema50) or (side_i == -1 and prev.ema20 > prev.ema50)
                        if reversed_:
                            result = paper.exit_position(pos_id, o, "reverse_signal", slippage=SLIPPAGE)
                            record_exit(result, ts.to_pydatetime(), "reverse_signal")

            # Halt uses mark-to-market equity, not cash after margin reserve.
            eq_open = _equity(paper, session, account_id, pos_id, float(df.at[i, "open"]), spec)
            if eq_open <= day_start * (1 - DAILY_LOSS_HALT) or eq_open <= starting_cash * (1 - TOTAL_LOSS_HALT):
                kill = True

            if pos_id is None and not kill:
                prev = df.iloc[i - 1]
                side_i = int(prev.signal)
                if side_i and np.isfinite(prev.atr14):
                    signals_seen += 1
                    raw_open = float(df.at[i, "open"])
                    stop = raw_open - side_i * (2.0 * float(prev.atr14))
                    side = _side_name(side_i)
                    signal_close = (prev.date + pd.Timedelta(hours=1)).to_pydatetime()
                    evidence_raw = json.dumps({
                        "label": LABEL,
                        "not_a_contemporaneous_publication": True,
                        "historical_observed_at": signal_close.isoformat(),
                        "run_recorded_at": run_recorded_at.isoformat(),
                        "signal_bar_open": str(prev.date),
                        "signal_bar_close_time": signal_close.isoformat(),
                        "signal": side,
                        "close": float(prev.close),
                        "ema20": float(prev.ema20),
                        "ema50": float(prev.ema50),
                        "atr14": float(prev.atr14),
                        "next_open_as_of_entry": raw_open,
                        "indicators_use_prefix_bars_only": True,
                        "current_quote_unused": unused_quote,
                    }, sort_keys=True)
                    clock.set(ts.to_pydatetime())
                    th = core.create_thesis(ThesisCreate(
                        instrument_id=INSTRUMENT_ID,
                        author=ACTOR,
                        direction=side,
                        horizon_days=1,
                        invalidation="2*ATR14 stop, reverse EMA20/50 cross, or loss halt",
                        subjective_confidence=0.0,
                        statement=(
                            f"{LABEL} deterministic EMA/ATR replay of a then-known closed 1h bar. "
                            f"This is not a live prediction and not an independent LLM review. "
                            f"historical_observed_at={signal_close.isoformat()} "
                            f"run_recorded_at={run_recorded_at.isoformat()}."
                        ),
                        label=LABEL,
                        is_example=True,
                        evidence=[EvidenceIn(
                            raw=evidence_raw,
                            provider="bitget_public_ohlcv_cache",
                            source_url=None,
                            quote_locator=f"BTCUSDT 1h {prev.date}",
                            observed_at=signal_close,
                            published_at=None,
                            available_at=signal_close,
                            timezone="UTC",
                            data_quality="unverified",
                            methodology="prefix-only EMA20/50 ATR14 on locked cache; no future bars",
                            methodology_version="historical_replay_v1",
                            vintage=str(prev.date),
                            label=LABEL,
                            relation="supports",
                            note="As-of snapshot of then-known closed bar. Not a claim that this thesis was published at the time.",
                            historic_context=True,
                        )],
                        change_reason=f"{LABEL} initial snapshot",
                    ), idempotency_key=content_hash(f"{LABEL}:{SYMBOL}:{ts}:{side}".encode())[:32])
                    plan = ext.create_sandbox_plan(
                        thesis_id=th["id"], account_id=account_id, symbol=SYMBOL,
                        side=side, entry_price=raw_open, stop_loss=stop, leverage=LEVERAGE,
                        research_run_id=None, stale=False, live_mode=False,
                    )
                    dec = ext.approve_paper(
                        plan["id"],
                        ApprovalIn(actor=ACTOR, user_choice="approve", rejection_reason=None),
                        idempotency_key=content_hash(f"{LABEL}:approve:{SYMBOL}:{ts}:{side}".encode())[:32],
                    )
                    if not dec.get("valid"):
                        rejections[f"approval_invalid:{dec.get('invalid_reason')}"] += 1
                    else:
                        fill = paper.execute_approved(account_id=account_id, decision_id=dec["id"])
                        if fill.get("status") != "filled":
                            rejections[str(fill.get("reason") or "rejected")] += 1
                        else:
                            pos = fill["position"]
                            pos_id = pos["id"]
                            fees += float(fill["fee"])
                            open_trade = ReplayTrade(
                                side=side,
                                signal_bar_open=str(prev.date),
                                entry_time=ts.isoformat(),
                                exit_time=None,
                                entry=float(fill["fill_price"]),
                                exit=None,
                                stop=float(pos["stop_loss"]),
                                qty=float(fill["qty"]),
                                net_usd=None,
                                exit_reason=None,
                                entry_fee=float(fill["fee"]),
                                exit_fee=None,
                                funding_paid=0.0,
                                thesis_id=th["id"],
                                plan_id=plan["id"],
                                decision_id=dec["id"],
                                entry_order_id=fill["id"],
                                position_id=pos_id,
                            )
                            trades.append(open_trade)

            if pos_id:
                pos = session.get(PaperPosition, pos_id)
                if pos and pos.status == "open":
                    o = float(df.at[i, "open"])
                    h = float(df.at[i, "high"])
                    l = float(df.at[i, "low"])
                    c = float(df.at[i, "close"])
                    clock.set((ts + pd.Timedelta(hours=1)).to_pydatetime())
                    result = paper.check_stop(
                        pos_id, {"open": o, "high": h, "low": l, "close": c},
                        slippage=SLIPPAGE, reason="atr_stop",
                    )
                    if result.get("filled"):
                        record_exit(result, (ts + pd.Timedelta(hours=1)).to_pydatetime(), "atr_stop")

            eq = _equity(paper, session, account_id, pos_id, float(df.at[i, "close"]), spec)
            if eq < 0:
                raise ValueError("Negative equity stress path; exact liquidation required")
            nav.append(((ts + pd.Timedelta(hours=1)).to_pydatetime(), eq))
            if eq <= day_start * (1 - DAILY_LOSS_HALT) or eq <= starting_cash * (1 - TOTAL_LOSS_HALT):
                kill = True
            if kill and pos_id:
                pos = session.get(PaperPosition, pos_id)
                if pos and pos.status == "open":
                    clock.set((ts + pd.Timedelta(hours=1)).to_pydatetime())
                    result = paper.exit_position(pos_id, float(df.at[i, "close"]), "loss_halt", slippage=SLIPPAGE)
                    record_exit(result, (ts + pd.Timedelta(hours=1)).to_pydatetime(), "loss_halt")
                    nav[-1] = (nav[-1][0], paper.get_account(account_id).cash)

        account = paper.get_account(account_id)
        closed = [t for t in trades if t.exit_reason is not None]
        still_open = [t for t in trades if t.exit_reason is None]
        pnls = [float(t.net_usd) for t in closed]
        wins = [p for p in pnls if p > WIN_EPS]
        losses = [p for p in pnls if p < -WIN_EPS]
        breakeven = [p for p in pnls if abs(p) <= WIN_EPS]
        eq_series = pd.Series([starting_cash] + [v for _, v in nav], dtype=float)
        final_eq = float(eq_series.iloc[-1]) if len(eq_series) else starting_cash
        open_unrealized = 0.0
        if still_open and pos_id:
            last_close = float(df["close"].iloc[-1])
            open_unrealized = _equity(paper, session, account_id, pos_id, last_close, spec) - account.cash - (
                session.get(PaperPosition, pos_id).margin if session.get(PaperPosition, pos_id) else 0
            )
        closed_net = sum(pnls)
        recon = closed_net + open_unrealized - (final_eq - starting_cash)
        # When a position is open, final equity includes reserved margin + unrealized;
        # cash is starting + closed_net - open_entry_fee - open_funding - open_margin.
        # final_eq - starting = closed_net + open_unrealized - open_entry_fee - open_funding
        # so recon uses cash-path identity below instead if a position remains open.
        if still_open and pos_id:
            pos = session.get(PaperPosition, pos_id)
            expected_final = starting_cash + closed_net + open_unrealized - (still_open[0].entry_fee + still_open[0].funding_paid)
            recon = final_eq - expected_final

        win_trade = next((t for t in closed if t.net_usd is not None and t.net_usd > WIN_EPS), None)
        loss_trade = next((t for t in closed if t.net_usd is not None and t.net_usd < -WIN_EPS), None)

        report = {
            "label": LABEL,
            "sandbox": True,
            "live_orders": False,
            "independent_llm_review": "NOT_WIRED",
            "research_narrative": "deterministic historical replay; not the chat assistant's predictive capability",
            "approval_actor": ACTOR,
            "not_user_approval": True,
            "held_out": False,
            "already_seen_dataset": True,
            "symbol": SYMBOL,
            "venue": "bitget",
            "strategy": STRATEGY,
            "adapter_differences": ADAPTER_DIFFERENCES,
            "liquidation_model": "approximate_stress_not_venue_verified",
            "margin_model_supported": False,
            "dataset": meta,
            "current_quote_provided_but_unused": unused_quote,
            "run_recorded_at": run_recorded_at.isoformat(),
            "starting_cash": starting_cash,
            "ending_equity": final_eq,
            "ending_cash": account.cash,
            "total_return_pct": (final_eq / starting_cash - 1) * 100,
            "max_drawdown_pct": float((1 - eq_series / eq_series.cummax()).max() * 100) if len(eq_series) else 0.0,
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "winrate_pct": (100 * len(wins) / len(pnls)) if pnls else None,
            "winrate_definition": f"closed fills only; win if net_usd > {WIN_EPS}; open and forced terminal closures reported separately; not a predictive or calibrated win rate",
            "fees": fees,
            "funding_net_paid": funding_net_paid,
            "net_costs": fees + funding_net_paid,
            "signals_observed": signals_observed,
            "signals_eligible": signals_seen,
            "rejections": dict(rejections),
            "loss_halt": bool(kill),
            "daily_loss_halt": DAILY_LOSS_HALT,
            "total_loss_halt": TOTAL_LOSS_HALT,
            "open_positions": len(still_open),
            "open_unrealized": open_unrealized,
            "forced_terminal_closures": sum(1 for t in closed if t.exit_reason == "loss_halt"),
            "stress_margin_breaches": stress_breaches,
            "reconciliation_error": recon,
            "period_start": nav[0][0].isoformat() if nav else None,
            "period_end": nav[-1][0].isoformat() if nav else None,
            "account_id": account_id,
            "db_path": str(db_path),
            "sample_winning_trade": win_trade.as_dict() if win_trade else None,
            "sample_losing_trade": loss_trade.as_dict() if loss_trade else None,
            "closed_trade_summaries": [t.as_dict() for t in closed],
            "open_trade_summaries": [t.as_dict() for t in still_open],
            "distinct_from": "previous 3-asset perps offline simulate and Freqtrade 14-day test; BTC-only PIIA paper-engine replay",
        }
        if abs(recon) > 1e-6:
            report["reconciliation_ok"] = False
        else:
            report["reconciliation_ok"] = True
        return ReplayResult(report=report, trades=trades)
    finally:
        session.close()


def write_report(result: ReplayResult, out_json: Path, out_txt: Path | None = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.report, indent=2, default=str))
    if out_txt is None:
        out_txt = out_json.with_suffix(".txt")
    r = result.report
    wr = r["winrate_pct"]
    wr_s = "n/a (no closed fills)" if wr is None else f"{wr:.1f}% ({r['wins']}/{r['closed_trades']})"
    lines = [
        "HISTORICAL_REPLAY — BTCUSDT only, sandbox paper engine, not live trading",
        f"dataset {r['dataset']['first']} -> {r['dataset']['last']} hash={r['dataset']['dataset_hash'][:16]} already_seen=True held_out=False",
        f"funding completeness under assumed 8-hour BTC schedule={r['dataset']['funding_complete']}; historical schedule and funding mark unverified",
        f"start {r['starting_cash']:.2f} U  end {r['ending_equity']:.2f} U  return {r['total_return_pct']:.2f}%  maxDD {r['max_drawdown_pct']:.2f}%",
        f"closed winrate {wr_s}  losses {r['losses']}  breakeven {r['breakeven']}  open {r['open_positions']}  loss_halt={r['loss_halt']}",
        f"fees {r['fees']:.6f}  funding_paid {r['funding_net_paid']:.6f}  net_costs {r['net_costs']:.6f}",
        f"signals observed {r.get('signals_observed')}  attempted {r['signals_eligible']}  rejections {r['rejections']}",
        f"approval actor={r['approval_actor']} (not the user)  independent_llm_review={r['independent_llm_review']}",
        f"reconciliation_ok={r['reconciliation_ok']} error={r['reconciliation_error']}",
        f"run_recorded_at={r['run_recorded_at']}",
        "",
        STRATEGY,
        ADAPTER_DIFFERENCES,
    ]
    out_txt.write_text("\n".join(lines) + "\n")
