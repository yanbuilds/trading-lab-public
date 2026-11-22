"""Compare a live ETF flow date to a local NYSE calendar model, without orders."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def annotate_etf_session(context: dict[str, Any], *, as_of: datetime | None = None) -> dict[str, Any]:
    """A calendar match is not verification of issuer publication or flow accuracy."""
    result = dict(context)
    result["latest_exchange_session_verified"] = False
    result["session_check"] = {"status": "unavailable", "calendar": "XNYS local calendar model",
                               "live_exchange_calendar_verified": False, "issuer_publication_verified": False,
                               "note": "Local modeled session close; unscheduled closures and issuer publication require independent confirmation."}
    if result.get("usable_for_btc_research") is not True:
        result["session_check"]["status"] = "flow_unavailable"
        return result
    try:
        from trading_lab.calendar_utils import last_completed_nyse_session
        stamp = as_of if as_of is not None else datetime.fromisoformat(result["completed_at_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("aware collection time required")
        stamp = stamp.astimezone(timezone.utc)
        flow_text = result["asof_complete_flow_date"]
        flow_day = date.fromisoformat(flow_text)
        if flow_day.isoformat() != flow_text:
            raise ValueError("canonical date required")
        session_day, close = last_completed_nyse_session(stamp)
        status = "matches_latest_closed_session" if flow_day == session_day else "not_latest_closed_session"
        result["latest_exchange_session_verified"] = flow_day == session_day
        result["session_check"].update(status=status, last_closed_session=session_day.isoformat(),
                                       last_close_utc=close.astimezone(timezone.utc).isoformat(),
                                       checked_at_utc=stamp.isoformat())
    except (ImportError, KeyError, TypeError, ValueError, RuntimeError, OverflowError, AttributeError):
        result["session_check"]["status"] = "calendar_or_date_unavailable"
    return result
