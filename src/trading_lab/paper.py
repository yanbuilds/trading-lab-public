from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_lab.config import LabConfig
from trading_lab.data import PricePanel, stale_report
from trading_lab.engine import _target_weights
from trading_lab.paths import ARTIFACTS
from trading_lab.signals import long_cash_signal
from trading_lab.timeutil import format_shanghai, now_utc


class PaperBlocked(RuntimeError):
    pass


def _fingerprint(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def latest_signal_replay(
    panel: PricePanel,
    cfg: LabConfig,
    *,
    holdings: dict[str, float] | None = None,
    cash: float | None = None,
    now=None,
) -> dict:
    """根据最近已完成 K 线重放信号，列出「下一时段开盘」的意向单。

    这不是全天候纸上交易器。若下一根开盘尚未出现，不会虚构成交价，也不会无限成交。
    任一标的行情过期时，停止发出新信号。
    """
    now = now or now_utc()
    stale = stale_report(panel, cfg, now=now)
    stale_names = [k for k, v in stale.items() if v.get("stale")]
    notes = [
        "这是「最新信号回放」，不是一直挂着的模拟交易服务器。",
        "意向单要等到该标的下一根已出现的开盘才能谈成交；当前不会用最新收盘价假装已经成交。",
        f"模拟资金与持仓若未提供，则默认空仓 + 示例资金 {cfg.example_capital_usd:.2f} USD。",
    ]
    blocked = bool(stale_names)
    if blocked:
        notes.append("存在过期行情，已停止发出新信号: " + ", ".join(stale_names))

    holdings = {k: float((holdings or {}).get(k, 0.0)) for k in cfg.instruments}
    cash = float(cfg.example_capital_usd if cash is None else cash)

    signals = {}
    last_bars = {}
    marks = {}
    for k, inst in cfg.instruments.items():
        if k not in panel.frames:
            signals[k] = None
            last_bars[k] = None
            continue
        df = panel.frames[k]
        close = df["close"]
        sig = long_cash_signal(close, cfg.sma_window)
        last = df.index[-1]
        signals[k] = None if pd.isna(sig.loc[last]) else int(sig.loc[last])
        last_bars[k] = str(last.date())
        marks[k] = float(df.iloc[-1]["close"])

    nav = cash + sum(holdings[k] * marks[k] for k in holdings if k in marks)
    flags = {k: (0 if signals[k] is None else signals[k]) for k in cfg.instruments}
    # 过期时不给新目标（保持当前仓位，目标为空仓变化被阻断）
    if blocked:
        targets = {k: None for k in cfg.instruments}
        intended = []
        status = "blocked_stale"
    else:
        targets = _target_weights(flags, cfg, halt_new_entries=False, current_shares=holdings)
        intended = []
        for k, inst in cfg.instruments.items():
            if k not in panel.frames or signals[k] is None:
                continue
            mark = marks[k]
            target_val = targets[k] * nav
            desired = 0.0 if mark <= 0 else target_val / mark
            delta = desired - holdings[k]
            if abs(delta) <= 1e-10:
                continue
            intended.append(
                {
                    "symbol": k,
                    "side": "buy" if delta > 0 else "sell",
                    "qty_estimate_at_last_mark": abs(delta),
                    "not_executable_quantity": True,
                    "signal": signals[k],
                    "target_weight": targets[k],
                    "execute_at": "该标的下一根开盘（尚未出现则不成交）",
                    "last_completed_bar": last_bars[k],
                    "do_not_fill_at_last_close": True,
                }
            )
        status = "signals_ready_pending_next_open"

    core = {
        "kind": "latest_signal_replay",
        "status": status,
        "last_completed_bars": last_bars,
        "signals": signals,
        "target_weights": targets,
        "holdings": holdings,
        "cash": cash,
        "nav_at_last_marks": nav,
        "intended_orders": intended,
        "marks": marks,
        "availability": panel.availability,
    }
    fingerprint = _fingerprint(core)
    snapshot = {
        **core,
        "generated_at_shanghai": format_shanghai(now),
        "fingerprint": fingerprint,
        "stale": stale,
        "blocked_new_signals": blocked,
        "notes": notes + [w for w in panel.warnings],
        "disclaimer": "示例组合，不是买卖建议，也不是真实账户。",
    }
    return snapshot


def write_snapshot(snapshot: dict, path: Path | None = None) -> tuple[Path, bool]:
    """幂等写入：若 fingerprint 已存在且意向单相同，不视为新的一笔交易。"""
    path = path or (ARTIFACTS / "paper_snapshot.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    unchanged = False
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("fingerprint") == snapshot.get("fingerprint"):
                unchanged = True
        except json.JSONDecodeError:
            old = None
    if not unchanged:
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, unchanged
