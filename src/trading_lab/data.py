from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trading_lab.calendar_utils import (
    crypto_bar_end_utc,
    etf_session_close_ny,
    is_completed_crypto_daily_bar,
    is_completed_etf_bar,
    last_completed_nyse_session,
    nyse_sessions,
)
from trading_lab.config import InstrumentConfig, LabConfig, load_config
from trading_lab.timeutil import SHANGHAI, UTC, format_shanghai, now_shanghai, now_utc, to_shanghai

OHLC_COLS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


@dataclass
class SeriesMeta:
    symbol: str
    source: str
    adjusted: bool
    fetched_at: datetime
    first_bar: date
    last_bar: date
    last_bar_complete: bool
    last_complete_bar: date | None
    rows: int
    dropped_incomplete: bool
    gap_dates: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "adjusted": self.adjusted,
            "fetched_at_utc": self.fetched_at.astimezone(UTC).isoformat(),
            "fetched_at_shanghai": format_shanghai(self.fetched_at),
            "first_bar": str(self.first_bar),
            "last_bar": str(self.last_bar),
            "last_bar_complete": self.last_bar_complete,
            "last_complete_bar": None if self.last_complete_bar is None else str(self.last_complete_bar),
            "rows": self.rows,
            "dropped_incomplete": self.dropped_incomplete,
            "gap_dates": self.gap_dates,
            "notes": self.notes,
        }


@dataclass
class PricePanel:
    frames: dict[str, pd.DataFrame]  # index = session date (midnight, tz-naive), columns OHLC_COLS
    metas: dict[str, SeriesMeta]
    availability: dict[str, str]  # ok / csv_fallback / missing
    warnings: list[str]

    def frame(self, symbol: str) -> pd.DataFrame:
        return self.frames[symbol]


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() if not isinstance(c, str) else c.lower() for c in df.columns]
        # yfinance 1.8: ('Close', 'BTC-USD') after we may have already flattened differently
    else:
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        base = c.split("_")[0]
        if base in OHLC_COLS:
            rename[c] = base
        elif c in {"adj close", "adj_close", "adjusted_close"}:
            rename[c] = "close"
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise DataError(f"行情缺少列: {missing}，实际列={list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0
    out = df[OHLC_COLS].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _flatten_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLC_COLS)
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the ticker level if present, else first level is price field.
        levels = list(df.columns.names)
        if "Ticker" in levels:
            try:
                df = df.xs(ticker, axis=1, level="Ticker")
            except Exception:
                # some versions put ticker in level 0
                df = df.droplevel(-1, axis=1)
        else:
            # columns like (Close, BTC-USD)
            try:
                df = df.xs(ticker, axis=1, level=1)
            except Exception:
                df.columns = [c[0] for c in df.columns]
    return _normalize_ohlc(df)


def _gap_dates(index: pd.DatetimeIndex, calendar: str) -> list[str]:
    if index.empty:
        return []
    days = pd.DatetimeIndex(index.normalize())
    if calendar == "crypto_utc":
        expected = pd.date_range(days.min(), days.max(), freq="D")
        missing = expected.difference(days)
        return [d.date().isoformat() for d in missing]
    return []  # ETF gaps are mostly holidays; we don't invent them. Optional holiday-aware check is elsewhere.


def _download_yahoo(ticker: str, start: str, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    # end is exclusive in some versions; go one day past UTC today.
    end = (now_utc().date() + timedelta(days=2)).isoformat()
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        group_by="column",
    )
    return _flatten_yf(df, ticker)


def _read_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment='#')
    # accept Date or date
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "datetime", "time"}:
            date_col = c
            break
    if date_col is None:
        raise DataError(f"{path} 缺少 Date 列")
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.set_index(date_col)
    return _normalize_ohlc(df)


def _drop_incomplete(df: pd.DataFrame, inst: InstrumentConfig, now: datetime) -> tuple[pd.DataFrame, bool, date | None]:
    if df.empty:
        return df, False, None
    last = df.index[-1].date()
    if inst.calendar == "crypto_utc":
        complete = is_completed_crypto_daily_bar(last, now)
    else:
        complete = is_completed_etf_bar(last, now)
    if complete:
        return df, False, last
    dropped = df.iloc[:-1]
    last_complete = None if dropped.empty else dropped.index[-1].date()
    return dropped, True, last_complete


def _cache_paths(cache_dir: Path, symbol: str) -> tuple[Path, Path]:
    safe = symbol.replace("/", "-")
    return cache_dir / f"{safe}.csv", cache_dir / f"{safe}.meta.json"


def save_cache(cache_dir: Path, symbol: str, df: pd.DataFrame, meta: SeriesMeta) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(cache_dir, symbol)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(csv_path)
    meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache(cache_dir: Path, symbol: str) -> tuple[pd.DataFrame, SeriesMeta] | None:
    csv_path, meta_path = _cache_paths(cache_dir, symbol)
    if not csv_path.exists() or not meta_path.exists():
        return None
    df = _read_csv_file(csv_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = SeriesMeta(
        symbol=raw["symbol"],
        source=raw["source"],
        adjusted=raw["adjusted"],
        fetched_at=datetime.fromisoformat(raw["fetched_at_utc"]),
        first_bar=date.fromisoformat(raw["first_bar"]),
        last_bar=date.fromisoformat(raw["last_bar"]),
        last_bar_complete=raw["last_bar_complete"],
        last_complete_bar=date.fromisoformat(raw["last_complete_bar"]) if raw.get("last_complete_bar") else None,
        rows=raw["rows"],
        dropped_incomplete=raw["dropped_incomplete"],
        gap_dates=raw.get("gap_dates") or [],
        notes=raw.get("notes") or [],
    )
    return df, meta


def fetch_instrument(
    inst: InstrumentConfig,
    cfg: LabConfig,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, SeriesMeta, str]:
    """Return completed bars, metadata, and availability tag. Never invents prices."""
    now = now or now_utc()
    notes: list[str] = []
    source = "none"
    df = pd.DataFrame(columns=OHLC_COLS)
    availability = "missing"

    csv_override = cfg.csv_dir / f"{inst.key}.csv"
    # Also accept yahoo ticker filename
    csv_alt = cfg.csv_dir / f"{inst.yahoo_ticker}.csv"

    yf_error = None
    if not (csv_override.exists() or csv_alt.exists()) or force:
        try:
            df = _download_yahoo(inst.yahoo_ticker, cfg.history_start, cfg.auto_adjust)
            if df.empty:
                raise DataError("yfinance 返回空表")
            source = "yfinance"
            availability = "ok"
        except Exception as e:  # noqa: BLE001 — surface any provider failure
            yf_error = e
            notes.append(f"yfinance 失败: {type(e).__name__}: {e}")

    if availability != "ok":
        path = csv_override if csv_override.exists() else csv_alt if csv_alt.exists() else None
        if path is not None:
            df = _read_csv_file(path)
            source = f"csv:{path.name}"
            availability = "csv_fallback"
            notes.append(f"已改用本地 CSV 回退：{path}")
        elif yf_error is not None:
            # try cache as last resort, clearly labelled stale-cache if present
            cached = load_cache(cfg.cache_dir, inst.key)
            if cached is not None:
                df, meta = cached
                meta.notes = list(meta.notes) + notes + ["使用此前缓存，不是本次新拉取。"]
                return df, meta, "cache_fallback"
            raise DataError(f"{inst.key} 无法取得行情：{notes}")

    # CSV/供应商数据也必须通过交易日历：非交易日不视为可成交时段。
    if inst.calendar == "XNYS" and not df.empty:
        valid_sessions = nyse_sessions(df.index.min().date(), df.index.max().date())
        invalid = df.index.difference(valid_sessions)
        if len(invalid):
            notes.append(f"删除非美股交易日数据: {[str(x.date()) for x in invalid[:5]]}")
        df = df.loc[df.index.intersection(valid_sessions)]
    start = pd.Timestamp(cfg.history_start)
    df = df[df.index >= start]
    if df.empty:
        raise DataError(f"{inst.key} 在 {cfg.history_start} 之后没有数据。")

    raw_last = df.index[-1].date()
    df, dropped, last_complete = _drop_incomplete(df, inst, now)
    if df.empty:
        raise DataError(f"{inst.key} 去掉未完成 K 线后没有可用数据。")

    last_bar = df.index[-1].date()
    complete_flag = True  # remaining last bar is complete by construction
    gaps = _gap_dates(df.index, inst.calendar)
    if gaps:
        # Do not fill gaps. Record them.
        preview = gaps[:8]
        suffix = "" if len(gaps) <= 8 else f" 等共 {len(gaps)} 天"
        notes.append(f"{inst.key} 在日历上存在缺口（未补齐，不是停牌捏造）: {preview}{suffix}")

    if inst.calendar == "crypto_utc":
        expected_last = (now.astimezone(UTC).date() - timedelta(days=1))
        if last_bar < expected_last:
            notes.append(
                f"{inst.key} 最近已完成日线是 {last_bar}，按 UTC 应至少有 {expected_last} 的已完成日线。"
                "数据源存在缺口，未补造，按实际最后一根判断新鲜度。"
            )

    if inst.kind == "equity_etf" and inst.calendar == "XNYS":
        weekend = [d.date().isoformat() for d in df.index if d.weekday() >= 5]
        if weekend:
            notes.append(f"{inst.key} 数据含周末行，已视为异常并删除: {weekend[:5]}")
            df = df[df.index.weekday < 5]
            last_bar = df.index[-1].date()
            last_complete = last_bar

    notes2 = list(notes)
    if dropped:
        notes2.append(f"已丢弃未完成的最后一根（原始最后日期 {raw_last}）")
    if source.startswith("csv"):
        notes2.append("CSV 是否已按公司行为复权，需你自行确认。")
    meta = SeriesMeta(
        symbol=inst.key,
        source=source,
        adjusted=bool(cfg.auto_adjust),
        fetched_at=now,
        first_bar=df.index[0].date(),
        last_bar=last_bar,
        last_bar_complete=complete_flag,
        last_complete_bar=last_complete or last_bar,
        rows=int(len(df)),
        dropped_incomplete=dropped,
        gap_dates=gaps,
        notes=notes2,
    )
    save_cache(cfg.cache_dir, inst.key, df, meta)
    return df, meta, availability


def fetch_all(cfg: LabConfig | None = None, *, force: bool = False, now: datetime | None = None) -> PricePanel:
    cfg = cfg or load_config()
    now = now or now_utc()
    frames: dict[str, pd.DataFrame] = {}
    metas: dict[str, SeriesMeta] = {}
    availability: dict[str, str] = {}
    warnings_list: list[str] = []
    errors: list[str] = []
    for key, inst in cfg.instruments.items():
        try:
            df, meta, tag = fetch_instrument(inst, cfg, force=force, now=now)
            frames[key] = df
            metas[key] = meta
            availability[key] = tag
            warnings_list.extend(meta.notes)
        except Exception as e:  # noqa: BLE001
            availability[key] = "missing"
            errors.append(f"{key}: {e}")
    if errors:
        warnings_list.extend(errors)
    if len(frames) < len(cfg.instruments):
        missing = [k for k in cfg.instruments if k not in frames]
        warnings_list.append(f"以下标的没有可用真实行情，不会用合成数据顶替: {missing}")
    return PricePanel(frames=frames, metas=metas, availability=availability, warnings=warnings_list)


def load_panel_from_cache(cfg: LabConfig | None = None) -> PricePanel:
    cfg = cfg or load_config()
    frames = {}
    metas = {}
    availability = {}
    warnings_list = []
    for key in cfg.instruments:
        cached = load_cache(cfg.cache_dir, key)
        if cached is None:
            availability[key] = "missing"
            warnings_list.append(f"{key} 尚无缓存，请先运行 fetch。")
            continue
        df, meta = cached
        frames[key] = df
        metas[key] = meta
        availability[key] = "cache"
    return PricePanel(frames=frames, metas=metas, availability=availability, warnings=warnings_list)


def stale_report(panel: PricePanel, cfg: LabConfig, now: datetime | None = None) -> dict[str, dict]:
    """Judge freshness from completed bars. Stale series must not emit new signals."""
    now = now or now_utc()
    out: dict[str, dict] = {}
    last_etf_session, last_etf_close = (None, None)
    try:
        last_etf_session, last_etf_close = last_completed_nyse_session(now)
    except Exception as e:  # noqa: BLE001
        cal_error = str(e)
    else:
        cal_error = None

    for key, inst in cfg.instruments.items():
        meta = panel.metas.get(key)
        if meta is None or key not in panel.frames:
            out[key] = {
                "stale": True,
                "reason": "没有可用行情",
                "as_of_shanghai": format_shanghai(now),
            }
            continue
        last = meta.last_complete_bar or meta.last_bar
        if inst.calendar == "crypto_utc":
            end = crypto_bar_end_utc(last)
            age_h = (now.astimezone(UTC) - end).total_seconds() / 3600.0
            stale = age_h > cfg.btc_max_age_hours or last < (now.astimezone(UTC).date() - timedelta(days=1))
            reason = (
                f"最近已完成日线 {last}（UTC 日结束于 {end.isoformat()}），距现在 {age_h:.1f} 小时，"
                f"阈值 {cfg.btc_max_age_hours} 小时；且必须有上一 UTC 自然日的完整日线。"
            )
            if stale:
                reason += " 判定为过期，停止发出新信号。"
        else:
            if cal_error:
                stale = True
                reason = f"无法核对美股日历: {cal_error}"
            else:
                # lag in completed NYSE sessions between data last bar and official last session
                sessions = []
                try:
                    from trading_lab.calendar_utils import nyse_sessions

                    sess = nyse_sessions(last, last_etf_session)
                    # sessions strictly after last bar, up to last completed session
                    sessions = [d.date() for d in sess if d.date() > last]
                    lag = len(sessions)
                except Exception:  # noqa: BLE001
                    lag = max(0, (last_etf_session - last).days)
                stale = lag > cfg.etf_max_completed_sessions_lag
                reason = (
                    f"最近已完成 ETF 日线 {last}；纽约交易所最近已收盘时段为 {last_etf_session} "
                    f"（收盘 {format_shanghai(last_etf_close)}）。滞后 {lag} 个交易日，"
                    f"阈值 {cfg.etf_max_completed_sessions_lag}。"
                )
                if stale:
                    reason += " 判定为过期，停止发出新信号。"
        out[key] = {
            "stale": bool(stale),
            "reason": reason,
            "last_complete_bar": str(last),
            "as_of_shanghai": format_shanghai(now),
        }
    return out
