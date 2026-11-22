"""Read official Bitget settled rate history; coverage/mark price remain unverified."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import httpx

URL = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
MAX_PAGES = 30


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).isdigit():
        raise ValueError("integer field required")
    return int(value)


def _parse(payload, symbol):
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        raise ValueError("invalid history response")
    server_ms = _integer(payload["requestTime"])
    rows = []
    for row in payload["data"]:
        if row.get("symbol") != symbol:
            raise ValueError("history symbol mismatch")
        ts = _integer(row["fundingTime"])
        if isinstance(row["fundingRate"], bool):
            raise ValueError("invalid funding rate")
        value = float(row["fundingRate"])
        if ts <= 0 or ts > server_ms or ts % 3_600_000 or not math.isfinite(value) or abs(value) > 1:
            raise ValueError("invalid or non-hour-aligned funding record")
        rows.append({"timestamp": ts, "rate": value})
    if any(rows[i]["timestamp"] <= rows[i + 1]["timestamp"] for i in range(len(rows) - 1)):
        raise ValueError("history page is not strictly newest first")
    return server_ms, rows


def fetch_history(symbol, start, end, *, client=None):
    if start.tzinfo is None or end.tzinfo is None or not start <= end:
        raise ValueError("ordered timezone-aware window required")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    own = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=False)
    pages, records, first, cutoff_reached = [], [], None, False
    previous_oldest = None
    def get(page):
        response = client.get(URL, params={"symbol": symbol, "productType": "USDT-FUTURES", "pageSize": "100", "pageNo": str(page)})
        response.raise_for_status()
        return response, _parse(response.json(), symbol)
    try:
        for page in range(1, MAX_PAGES + 1):
            response, (server_ms, rows) = get(page)
            if page == 1:
                first = rows
                if int(end.timestamp() * 1000) > server_ms:
                    raise ValueError("funding window ends after provider time")
            if not rows:
                raise ValueError("history ended before window was bracketed")
            if previous_oldest is not None and rows[0]["timestamp"] >= previous_oldest:
                raise ValueError("overlapping or repeated history pages")
            previous_oldest = rows[-1]["timestamp"]
            pages.append({"page": page, "server_at_utc": datetime.fromtimestamp(server_ms / 1000, timezone.utc).isoformat(),
                          "raw_sha256": hashlib.sha256(response.content).hexdigest(), "count": len(rows),
                          "raw_json": response.content.decode("utf-8")})
            records.extend(rows)
            if previous_oldest <= int(start.timestamp() * 1000):
                cutoff_reached = True
                break
        if not cutoff_reached:
            raise ValueError("history page limit reached before window")
        recheck_response, (_, first_again) = get(1)
        if first_again != first:
            raise ValueError("history shifted or revised during pagination")
    finally:
        if own:
            client.close()
    selected = [r for r in records if int(start.timestamp() * 1000) <= r["timestamp"] <= int(end.timestamp() * 1000)]
    return {"kind": "bitget_official_public_funding_history", "symbol": symbol, "source_url": URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(), "pages": pages, "records": selected,
            "first_page_recheck_raw_json": recheck_response.content.decode("utf-8"),
            "first_page_recheck_sha256": hashlib.sha256(recheck_response.content).hexdigest(),
            "history_window_bracketed": True, "schedule_completeness_proven": False,
            "actual_account_funding_paid": None,
            "note": "Public rates only; page bracketing is not proof of schedule completeness, mark price, or actual account settlement."}


def validate_history(history, symbol, start, end):
    """Reconstruct selected rates from frozen raw pages instead of trusting summaries."""
    if (history.get("symbol") != symbol or history.get("source_url") != URL or history.get("history_window_bracketed") is not True
            or datetime.fromisoformat(history["window_start_utc"]) != start
            or datetime.fromisoformat(history["window_end_utc"]) != end):
        raise ValueError("funding history provenance or window invalid")
    pages = history.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError("history pages missing")
    all_rows, first, previous_oldest = [], None, None
    for index, page in enumerate(pages, 1):
        raw = page["raw_json"].encode("utf-8")
        if page.get("page") != index or hashlib.sha256(raw).hexdigest() != page.get("raw_sha256"):
            raise ValueError("funding page integrity mismatch")
        server_ms, rows = _parse(json.loads(raw), symbol)
        if (not rows or page.get("count") != len(rows)
                or datetime.fromisoformat(page["server_at_utc"]) != datetime.fromtimestamp(server_ms / 1000, timezone.utc)):
            raise ValueError("funding page metadata mismatch")
        if index == 1:
            first = rows
            if int(end.timestamp() * 1000) > server_ms:
                raise ValueError("funding window ends after provider time")
        if previous_oldest is not None and rows[0]["timestamp"] >= previous_oldest:
            raise ValueError("overlapping or repeated history pages")
        previous_oldest = rows[-1]["timestamp"]
        all_rows.extend(rows)
    if previous_oldest > int(start.timestamp() * 1000):
        raise ValueError("history window not bracketed")
    recheck = history["first_page_recheck_raw_json"].encode("utf-8")
    if hashlib.sha256(recheck).hexdigest() != history.get("first_page_recheck_sha256"):
        raise ValueError("funding recheck integrity mismatch")
    _, first_again = _parse(json.loads(recheck), symbol)
    if first_again != first:
        raise ValueError("history shifted or revised during pagination")
    selected = [r for r in all_rows if int(start.timestamp() * 1000) <= r["timestamp"] <= int(end.timestamp() * 1000)]
    if history.get("records") != selected:
        raise ValueError("funding selected records mismatch")
    return selected
