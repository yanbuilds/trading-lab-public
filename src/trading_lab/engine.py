from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_lab.config import LabConfig, load_config
from trading_lab.data import PricePanel
from trading_lab.signals import long_cash_signal

# 目标权重变化小于该阈值时不调仓（完全退出除外），避免趋势策略每天微调。
REBALANCE_BAND = 0.01


@dataclass
class BacktestResult:
    nav: pd.Series
    cash: pd.Series
    weights: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    marks: pd.DataFrame  # last used close per asset on each nav date
    peak_nav: pd.Series
    drawdown: pd.Series
    halted: pd.Series
    config_label: str
    notes: list[str] = field(default_factory=list)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    def slice_dates(self, start, end=None) -> "BacktestResult":
        mask = self.nav.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.nav.index <= pd.Timestamp(end)
        sliced_events = self.events
        if not self.events.empty and "timestamp_utc" in self.events.columns:
            ev_ts = pd.to_datetime(self.events["timestamp_utc"], utc=True)
            start_utc = pd.Timestamp(start)
            if start_utc.tzinfo is None:
                start_utc = start_utc.tz_localize("UTC")
            ev_mask = ev_ts >= start_utc
            if end is not None:
                end_utc = pd.Timestamp(end)
                if end_utc.tzinfo is None:
                    end_utc = end_utc.tz_localize("UTC") + pd.Timedelta(days=1)
                ev_mask &= ev_ts <= end_utc
            sliced_events = self.events.loc[ev_mask].copy()
        return BacktestResult(
            nav=self.nav.loc[mask],
            cash=self.cash.loc[mask],
            weights=self.weights.loc[mask],
            holdings=self.holdings.loc[mask],
            trades=self.trades[self.trades["date"].isin(self.nav.loc[mask].index)] if not self.trades.empty else self.trades,
            signals=self.signals.loc[self.signals.index.intersection(self.nav.loc[mask].index)] if not self.signals.empty else self.signals,
            marks=self.marks.loc[mask],
            peak_nav=self.peak_nav.loc[mask],
            drawdown=self.drawdown.loc[mask],
            halted=self.halted.loc[mask],
            config_label=self.config_label,
            notes=list(self.notes),
            events=sliced_events,
        )


def _target_weights(long_flags: dict[str, int], cfg: LabConfig, *, halt_new_entries: bool, current_shares: dict[str, float]) -> dict[str, float]:
    """Cash + longs. Caps are example limits, not advice."""
    investable = min(cfg.max_gross_exposure, 1.0) - cfg.cash_floor
    if investable < 0:
        investable = 0.0
    longs = [k for k, v in long_flags.items() if int(v) == 1]
    if halt_new_entries:
        # 只保留已经持有的多头；信号仍为多但当前仓位为 0 的，不得新开。
        longs = [k for k in longs if current_shares.get(k, 0.0) > 0]
    if not longs or investable <= 0:
        return {k: 0.0 for k in long_flags}
    raw = investable / len(longs)
    w = min(raw, cfg.max_weight)
    # 如果 min 让总仓小于 investable，多余留现金（更保守）。
    return {k: (w if k in longs else 0.0) for k in long_flags}


def market_events(panel, cfg):
    """Actual UTC instants; input dates are session labels, not execution times."""
    from trading_lab.calendar_utils import nyse_calendar
    events = []
    for k, df in panel.frames.items():
        if k not in cfg.instruments or df.empty:
            continue
        inst = cfg.instruments[k]
        sched = None
        if inst.calendar != "crypto_utc":
            sched = nyse_calendar().schedule(start_date=str(df.index.min().date()), end_date=str(df.index.max().date()))
        for d, row in df.iterrows():
            if inst.calendar == "crypto_utc":
                op = pd.Timestamp(d).tz_localize("UTC")
                cl = op + pd.Timedelta(days=1)
            else:
                if d not in sched.index:
                    continue
                op = pd.Timestamp(sched.loc[d, "market_open"]).tz_convert("UTC")
                cl = pd.Timestamp(sched.loc[d, "market_close"]).tz_convert("UTC")
            events.extend([(op, "open", k, d, float(row["open"])), (cl, "close", k, d, float(row["close"]))])
    return sorted(events, key=lambda e: (e[0], 0 if e[1] == "close" else 1, e[2]))


def run_backtest(panel: PricePanel, cfg: LabConfig | None = None, *, capital: float | None = None,
                 trading_start: str | None = None, baseline: bool = False,
                 matched_risk_budget: bool = False) -> BacktestResult:
    """UTC chronological opens/closes; ETF sales T+1 next-session-open settlement.

    Same-instant buy requests share available cash pro rata. Completed bar signals
    may enter only subsequent opens (crypto adjacent close/open boundary permits
    zero-latency processing, an explicit approximation).
    """
    from trading_lab.calendar_utils import nyse_calendar
    cfg = cfg or load_config()
    capital = float(cfg.example_capital_usd if capital is None else capital)
    keys = sorted(k for k in cfg.instruments if k in panel.frames)
    events = market_events(panel, cfg)
    if not events:
        raise RuntimeError("No valid market events")
    import itertools
    batches = [(t, list(g)) for t, g in itertools.groupby(events, key=lambda e: e[0])]
    signals = {k: long_cash_signal(panel.frames[k]["close"], cfg.sma_window) for k in keys}
    shares = dict.fromkeys(keys, 0.0)
    marks = dict.fromkeys(keys, float("nan"))
    flags = dict.fromkeys(keys, 0)
    signal_time = dict.fromkeys(keys, None)
    settled = capital
    receivables = []  # (release UTC timestamp, net proceeds)
    bought = set()
    peak = capital
    daily = {}
    trades = []
    audit = []
    start = pd.Timestamp(trading_start).tz_localize("UTC") if trading_start else batches[0][0]
    for ts, batch in batches:
        # Settled cash never includes future ETF sale proceeds.
        releasing = sum(amount for release, amount in receivables if release <= ts)
        settled += releasing
        receivables = [(release, amount) for release, amount in receivables if release > ts]
        # Closes at the boundary are complete before opens at the same instant.
        for _, typ, k, d, px in batch:
            if typ != "close":
                continue
            marks[k] = px
            val = signals[k].loc[d]
            flags[k] = 0 if pd.isna(val) else int(val)
            signal_time[k] = ts
        opens = {k: (d, px) for _, typ, k, d, px in batch if typ == "open"}
        for k, (_, px) in opens.items():
            marks[k] = px
        unsettled = sum(amount for _, amount in receivables)
        nav_pre = _nav(settled + unsettled, shares, marks)
        peak = max(peak, nav_pre)
        halted = (1-nav_pre/peak) >= cfg.max_drawdown_halt if peak > 0 else True
        targets = _target_weights(flags, cfg, halt_new_entries=halted, current_shares=shares)
        requests = {}
        if ts >= start:
            for k, (d, px) in opens.items():
                if px <= 0:
                    raise ValueError("Nonpositive open")
                if baseline:
                    if k in bought:
                        continue
                    alloc = min(cfg.max_weight, (1-cfg.cash_floor)/len(keys)) if matched_risk_budget else 1/len(keys)
                    requests[k] = max(0, capital*alloc/(px*(1+cfg.instruments[k].slippage)*(1+cfg.instruments[k].fee)))
                else:
                    if signal_time[k] is None:
                        continue
                    desired = targets[k]*nav_pre/px
                    if halted:
                        desired = min(desired, shares[k])
                    if targets[k] > 0 and abs(targets[k]-shares[k]*px/nav_pre) < REBALANCE_BAND:
                        desired = shares[k]
                    requests[k] = max(0, desired)
        # Execute sells only now, with UTC settlement scheduled after this instant.
        for k, desired in requests.items():
            qty = shares[k]-desired
            if qty <= 1e-12:
                continue
            inst = cfg.instruments[k]
            raw = opens[k][1]
            fill = raw*(1-inst.slippage)
            notional = qty*fill
            fee = notional*inst.fee
            shares[k] -= qty
            if inst.calendar == "crypto_utc":
                settled += notional-fee
                release = ts
            else:
                sched = nyse_calendar().schedule(start_date=str(opens[k][0].date()+pd.Timedelta(days=1)),
                                                 end_date=str(opens[k][0].date()+pd.Timedelta(days=15)))
                release = pd.Timestamp(sched.iloc[0]["market_open"]).tz_convert("UTC")
                receivables.append((release, notional-fee))
            trades.append(dict(date=ts.tz_localize(None).normalize(), timestamp_utc=ts.isoformat(),
                               symbol=k, side="sell", qty=qty, open=raw, fill=fill, fee=fee, notional=notional,
                               signal_at_utc=str(signal_time[k]), cash_available_at_utc=release.isoformat()))
        buy_requests = []
        for k, desired in requests.items():
            qty = desired-shares[k]
            if qty <= 1e-12:
                continue
            inst=cfg.instruments[k]; fill=opens[k][1]*(1+inst.slippage)
            buy_requests.append((k,qty,fill,qty*fill*(1+inst.fee)))
        available = settled if baseline else max(0, settled-cfg.cash_floor*nav_pre)
        cost = sum(x[3] for x in buy_requests)
        scale = min(1,available/cost) if cost > 0 else 0
        for k,qty,fill,cost in buy_requests:
            qty *= scale
            if qty <= 1e-12:
                continue
            notional=qty*fill; fee=notional*cfg.instruments[k].fee
            settled -= notional+fee
            shares[k] += qty
            bought.add(k)
            trades.append(dict(date=ts.tz_localize(None).normalize(),timestamp_utc=ts.isoformat(),
                               symbol=k,side="buy",qty=qty,open=opens[k][1],fill=fill,fee=fee,notional=notional,
                               signal_at_utc=str(signal_time[k]),cash_available_at_utc=ts.isoformat()))
        if settled < -1e-7 or min(shares.values()) < -1e-9:
            raise AssertionError("Cash/long-only invariant violated")
        economic_cash = settled+sum(x[1] for x in receivables)
        nav = _nav(economic_cash,shares,marks)
        peak = nav if all(q<=1e-12 for q in shares.values()) else max(peak,nav)
        dd = 1-nav/peak if peak else 0
        w={k: (shares[k]*marks[k]/nav if marks[k]==marks[k] and nav else 0) for k in keys}
        audit.append(dict(timestamp_utc=ts.isoformat(),settled_cash=settled,
                          unsettled_cash=economic_cash-settled,nav=nav,
                          types=",".join(sorted(set(e[1] for e in batch)))))
        daily[ts.tz_localize(None).normalize()] = dict(nav=nav,cash=economic_cash,weights=w,holdings=dict(shares),
                                                    marks=dict(marks),peak=peak,dd=dd,halted=dd>=cfg.max_drawdown_halt)
    idx=pd.DatetimeIndex(sorted(daily))
    def series(field,name): return pd.Series([daily[d][field] for d in idx],index=idx,name=name)
    def frame(field): return pd.DataFrame([daily[d][field] for d in idx],index=idx)
    tdf=pd.DataFrame(trades,columns=["date","timestamp_utc","symbol","side","qty","open","fill","fee","notional","signal_at_utc","cash_available_at_utc"])
    sig=pd.DataFrame(signals)
    return BacktestResult(nav=series("nav","nav"),cash=series("cash","cash"),weights=frame("weights"),
                          holdings=frame("holdings"),trades=tdf,signals=sig,marks=frame("marks"),
                          peak_nav=series("peak","peak_nav"),drawdown=series("dd","drawdown"),halted=series("halted","halted"),
                          config_label="utc_events_tplus1",events=pd.DataFrame(audit),notes=[
                              "UTC chronological events; ETF sales available at next NYSE session open (T+1 conservative assumption).",
                              "Crypto sells settle immediately; no margin lending; same-instant buys pro rata.",
                              "Daily NAV is last observed event per UTC date; ETF prices carry until next observed open/close, not invented weekend fills.",
                              "Zero-latency crypto close/open boundary and full daily-open fills are simplifying assumptions, not live execution guarantees."])


def _nav(cash: float, shares: dict[str, float], last_close: dict[str, float]) -> float:
    return cash + sum(q*last_close[k] for k,q in shares.items() if last_close.get(k)==last_close.get(k))
