from __future__ import annotations
from datetime import datetime, timezone
class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)
class FrozenClock(Clock):
    def __init__(self, instant: datetime):
        self._now = instant if instant.tzinfo else instant.replace(tzinfo=timezone.utc)
    def now(self) -> datetime:
        return self._now
    def set(self, instant: datetime) -> None:
        self._now = instant if instant.tzinfo else instant.replace(tzinfo=timezone.utc)
