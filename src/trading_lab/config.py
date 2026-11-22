from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trading_lab.paths import CONFIG_DEFAULT, ROOT


@dataclass(frozen=True)
class InstrumentConfig:
    key: str
    kind: str
    proxy_for: str
    calendar: str
    yahoo_ticker: str
    fee: float
    slippage: float


@dataclass(frozen=True)
class LabConfig:
    timezone: str
    example_capital_usd: float
    instruments: dict[str, InstrumentConfig]
    sma_window: int
    max_weight: float
    cash_floor: float
    max_drawdown_halt: float
    max_gross_exposure: float
    history_start: str
    train_end: str
    test_start: str
    btc_max_age_hours: float
    etf_max_completed_sessions_lag: int
    auto_adjust: bool
    cache_dir: Path
    csv_dir: Path
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def instrument_keys(self) -> list[str]:
        return list(self.instruments.keys())

    @property
    def etf_keys(self) -> list[str]:
        return [k for k, i in self.instruments.items() if i.calendar == "XNYS"]

    @property
    def crypto_keys(self) -> list[str]:
        return [k for k, i in self.instruments.items() if i.calendar == "crypto_utc"]


def load_config(path: Path | str | None = None) -> LabConfig:
    cfg_path = Path(os.environ.get("TRADING_LAB_CONFIG", path or CONFIG_DEFAULT))
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    instruments = {}
    for key, item in (raw.get("instruments") or {}).items():
        instruments[key] = InstrumentConfig(
            key=key,
            kind=item["kind"],
            proxy_for=item.get("proxy_for", key),
            calendar=item["calendar"],
            yahoo_ticker=item.get("yahoo_ticker", key),
            fee=float(item["fee"]),
            slippage=float(item["slippage"]),
        )

    data = raw.get("data") or {}
    risk = raw.get("risk") or {}
    split = raw.get("split") or {}
    stale = raw.get("stale") or {}
    strategy = raw.get("strategy") or {}

    cache_dir = Path(os.environ.get("TRADING_LAB_CACHE_DIR", data.get("cache_dir", "data/cache")))
    csv_dir = Path(os.environ.get("TRADING_LAB_CSV_DIR", data.get("csv_dir", "data/csv")))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    if not csv_dir.is_absolute():
        csv_dir = ROOT / csv_dir

    return LabConfig(
        timezone=os.environ.get("TRADING_LAB_TZ", raw.get("timezone", "Asia/Shanghai")),
        example_capital_usd=float(raw.get("example_capital_usd", 100000)),
        instruments=instruments,
        sma_window=int(strategy.get("sma_window", 200)),
        max_weight=float(risk.get("max_weight", 0.4)),
        cash_floor=float(risk.get("cash_floor", 0.2)),
        max_drawdown_halt=float(risk.get("max_drawdown_halt", 0.2)),
        max_gross_exposure=float(risk.get("max_gross_exposure", 1.0)),
        history_start=str(split.get("history_start", "2018-01-01")),
        train_end=str(split.get("train_end", "2023-12-31")),
        test_start=str(split.get("test_start", "2024-01-01")),
        btc_max_age_hours=float(stale.get("btc_max_age_hours", 36)),
        etf_max_completed_sessions_lag=int(stale.get("etf_max_completed_sessions_lag", 1)),
        auto_adjust=bool(data.get("auto_adjust", True)),
        cache_dir=cache_dir,
        csv_dir=csv_dir,
        raw=raw,
    )
