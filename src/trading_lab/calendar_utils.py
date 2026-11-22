from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from trading_lab.timeutil import NEW_YORK, UTC, now_utc


@lru_cache(maxsize=4)
def nyse_calendar():
    return mcal.get_calendar("XNYS")


def nyse_sessions(start: date | str, end: date | str) -> pd.DatetimeIndex:
    sched = nyse_calendar().schedule(start_date=str(start), end_date=str(end))
    return pd.DatetimeIndex(pd.to_datetime(sched.index).normalize())


def last_completed_nyse_session(now: datetime | None = None) -> tuple[date, datetime]:
    """Return (session_date, session_close_ny) of the most recently completed NYSE session."""
    now = now or now_utc()
    now_ny = now.astimezone(NEW_YORK)
    # Look back a generous window so holidays/weekends are covered.
    start = (now_ny.date() - timedelta(days=14)).isoformat()
    end = now_ny.date().isoformat()
    sched = nyse_calendar().schedule(start_date=start, end_date=end)
    if sched.empty:
        raise RuntimeError("无法取得纽约证券交易所交易日历。")
    completed = []
    for session_day, row in sched.iterrows():
        close = row["market_close"].to_pydatetime()
        if close.tzinfo is None:
            close = close.replace(tzinfo=UTC)
        close = close.astimezone(NEW_YORK)
        if now_ny >= close:
            completed.append((pd.Timestamp(session_day).date(), close))
    if not completed:
        # Before the first close in the window (e.g. early Monday before any prior load):
        # extend further back.
        start2 = (now_ny.date() - timedelta(days=30)).isoformat()
        sched2 = nyse_calendar().schedule(start_date=start2, end_date=end)
        for session_day, row in sched2.iterrows():
            close = row["market_close"].to_pydatetime()
            if close.tzinfo is None:
                close = close.replace(tzinfo=UTC)
            close = close.astimezone(NEW_YORK)
            if now_ny >= close:
                completed.append((pd.Timestamp(session_day).date(), close))
    if not completed:
        raise RuntimeError("当前时间之前没有已完成的美股交易时段。")
    return completed[-1]


def is_completed_etf_bar(bar_date: date, now: datetime | None = None) -> bool:
    now = now or now_utc()
    now_ny = now.astimezone(NEW_YORK)
    sched = nyse_calendar().schedule(start_date=str(bar_date), end_date=str(bar_date))
    if sched.empty:
        # Not a session — a bar on a non-session date is unexpected; treat as incomplete/invalid.
        return False
    close = sched.iloc[0]["market_close"].to_pydatetime()
    if close.tzinfo is None:
        close = close.replace(tzinfo=UTC)
    return now_ny >= close.astimezone(NEW_YORK)


def is_completed_crypto_daily_bar(bar_date: date, now: datetime | None = None) -> bool:
    """Yahoo BTC-USD 日线按 UTC 自然日切分：该日 24:00 UTC 之后才算完成。"""
    now = now or now_utc()
    period_end = datetime(bar_date.year, bar_date.month, bar_date.day, tzinfo=UTC) + timedelta(days=1)
    return now.astimezone(UTC) >= period_end


def crypto_bar_end_utc(bar_date: date) -> datetime:
    return datetime(bar_date.year, bar_date.month, bar_date.day, tzinfo=UTC) + timedelta(days=1)


def etf_session_close_ny(bar_date: date) -> datetime | None:
    sched = nyse_calendar().schedule(start_date=str(bar_date), end_date=str(bar_date))
    if sched.empty:
        return None
    close = sched.iloc[0]["market_close"].to_pydatetime()
    if close.tzinfo is None:
        close = close.replace(tzinfo=UTC)
    return close.astimezone(NEW_YORK)
