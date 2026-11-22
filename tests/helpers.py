from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_lab.config import InstrumentConfig, LabConfig
from trading_lab.data import PricePanel, SeriesMeta
from trading_lab.timeutil import UTC

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def _inst(key: str, calendar: str, fee=0.001, slippage=0.001) -> InstrumentConfig:
    return InstrumentConfig(
        key=key,
        kind="crypto_spot" if calendar == "crypto_utc" else "equity_etf",
        proxy_for=f"TEST {key}",
        calendar=calendar,
        yahoo_ticker=key,
        fee=fee,
        slippage=slippage,
    )


def make_cfg(tmp_path: Path, **over) -> LabConfig:
    instruments = {
        "BTC-USD": _inst("BTC-USD", "crypto_utc", fee=0.001, slippage=0.001),
        "GLD": _inst("GLD", "XNYS", fee=0.0005, slippage=0.0005),
        "QQQ": _inst("QQQ", "XNYS", fee=0.0005, slippage=0.0005),
    }
    kwargs = dict(
        timezone="Asia/Shanghai",
        example_capital_usd=100000.0,
        instruments=instruments,
        sma_window=5,
        max_weight=0.40,
        cash_floor=0.20,
        max_drawdown_halt=0.20,
        max_gross_exposure=1.0,
        history_start="2020-01-01",
        train_end="2020-06-30",
        test_start="2020-07-01",
        btc_max_age_hours=36,
        etf_max_completed_sessions_lag=1,
        auto_adjust=True,
        cache_dir=tmp_path / "cache",
        csv_dir=tmp_path / "csv",
        raw={},
    )
    kwargs.update(over)
    return LabConfig(**kwargs)


def ohlc_from_close(index: pd.DatetimeIndex, close: pd.Series, open_shift: float = 0.0) -> pd.DataFrame:
    close = pd.Series(close, index=index, dtype=float)
    # 开盘用「前收 + 偏移」，避免把当日收盘当成开盘（便于检测前视）。
    prev = close.shift(1)
    open_ = (prev.fillna(close.iloc[0]) + open_shift).astype(float)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.01
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.01
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000}, index=index)


def panel_from_frames(frames: dict[str, pd.DataFrame], source="SYNTHETIC_TEST_ONLY") -> PricePanel:
    metas = {}
    for k, df in frames.items():
        metas[k] = SeriesMeta(
            symbol=k,
            source=source,
            adjusted=True,
            fetched_at=datetime(2020, 12, 31, tzinfo=UTC),
            first_bar=df.index[0].date(),
            last_bar=df.index[-1].date(),
            last_bar_complete=True,
            last_complete_bar=df.index[-1].date(),
            rows=len(df),
            dropped_incomplete=False,
            gap_dates=[],
            notes=["SYNTHETIC_TEST_ONLY: 合成数据，仅测试。"],
        )
    return PricePanel(
        frames=frames,
        metas=metas,
        availability={k: "synthetic_test_only" for k in frames},
        warnings=["SYNTHETIC_TEST_ONLY"],
    )


@pytest.fixture
def tmp_cfg(tmp_path):
    return make_cfg(tmp_path)
