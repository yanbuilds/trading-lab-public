from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def now_shanghai() -> datetime:
    return now_utc().astimezone(SHANGHAI)


def to_shanghai(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(SHANGHAI)


def format_shanghai(dt: datetime | None, *, label: str = "Asia/Shanghai") -> str:
    if dt is None:
        return "未知"
    local = to_shanghai(dt)
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} ({label})"
