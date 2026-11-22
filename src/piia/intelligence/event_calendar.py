"""Current official U.S. economic and policy calendars from Fed, BLS, and BEA.

This is contemporaneous published-schedule evidence, not a historical
point-in-time archive and not a verified exact release-time stamp. Completeness
requires all three official sources, not a single speech or one agency page.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

FED_CALENDAR_URL = "https://www.federalreserve.gov/json/calendar.json"
FED_FOMC_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_RELEASE_INDEX = "https://www.bls.gov/schedule/news_release/"
BEA_ICS_URL = "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics"
BEA_PAGE = "https://www.bea.gov/news/schedule"
ALLOWED_HOSTS = {
    "www.federalreserve.gov", "federalreserve.gov",
    "www.bls.gov", "bls.gov",
    "www.bea.gov", "bea.gov",
}
EASTERN = ZoneInfo("America/New_York")
MONTH_NAMES = ("January","February","March","April","May","June","July","August","September","October","November","December")
MONTH_RE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
QUARTER_RE = r"(?:First|Second|Third|Fourth)\s+Quarter"
RELEASE_RE = re.compile(
    rf"([A-Za-z][A-Za-z0-9 ,.\-\(\)/]+?)\s+((?:{MONTH_RE}\s+\d{{4}})|(?:{QUARTER_RE}\s+\d{{4}})|(?:Annual\s+\d{{4}}))\s+(\d{{1,2}}:\d{{2}}\s*[AP]M)",
    re.I,
)
HOLIDAY_RE = re.compile(r"(.+?\s+Holiday)$", re.I)
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*([ap])\.?m\.?$", re.I)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._in_cell = True
        elif tag == "br" and self._in_cell and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            text = html_lib.unescape(re.sub(r"\s+", " ", "".join(self._cell))).strip()
            self._row.append(text)
            self._cell = None
            self._in_cell = False
        elif tag == "tr" and self._table is not None and self._row is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._cell is not None:
            self._cell.append(data)


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _get(client: httpx.Client, url: str, *, follow: bool = True) -> httpx.Response:
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing non-allowlisted host {host}")
    response = client.get(url, follow_redirects=follow)
    final = urlparse(str(response.url)).hostname
    if final not in ALLOWED_HOSTS:
        raise ValueError(f"redirected off allowlisted host {final}")
    response.raise_for_status()
    return response


def _parse_ampm(text: str) -> tuple[int, int] | None:
    match = TIME_RE.match((text or "").strip())
    if not match:
        return None
    hour, minute, ap = int(match.group(1)), int(match.group(2)), match.group(3).lower()
    if hour == 12:
        hour = 0
    if ap == "p":
        hour += 12
    return hour, minute


def _eastern_iso(year: int, month: int, day: int, time_text: str | None) -> str | None:
    parsed = _parse_ampm(time_text or "")
    if not parsed:
        try:
            return datetime(year, month, day, tzinfo=EASTERN).isoformat()
        except ValueError:
            return None
    hour, minute = parsed
    try:
        return datetime(year, month, day, hour, minute, tzinfo=EASTERN).isoformat()
    except ValueError:
        return None


def parse_fed_calendar(raw: bytes, *, now: datetime) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8-sig"))
    events = list(data.get("events") or [])
    policy = []
    for item in events:
        kind = str(item.get("type") or "")
        if kind not in {"FOMC", "Beige"}:
            continue
        month = str(item.get("month") or "")
        try:
            year, mon = [int(part) for part in month.split("-", 1)]
            day = int(str(item.get("days") or "").split("-")[0])
        except (ValueError, TypeError):
            continue
        scheduled = _eastern_iso(year, mon, day, str(item.get("time") or ""))
        policy.append({
            "agency": "federal_reserve",
            "type": kind,
            "title": html_lib.unescape(re.sub(r"<[^>]+>", " ", str(item.get("title") or ""))).strip(),
            "description": html_lib.unescape(re.sub(r"<[^>]+>", " ", str(item.get("description") or ""))).strip() or None,
            "scheduled_local_date": f"{year:04d}-{mon:02d}-{day:02d}",
            "published_time_text": item.get("time"),
            "timezone_assumption": "America/New_York",
            "scheduled_at_eastern_iso": scheduled,
            "source_url": FED_CALENDAR_URL,
        })
    window_start = (_aware(now) - timedelta(days=7)).date()
    window_end = (_aware(now) + timedelta(days=90)).date()
    in_window = [e for e in policy if window_start <= date.fromisoformat(e["scheduled_local_date"]) <= window_end]
    upcoming = [e for e in in_window if date.fromisoformat(e["scheduled_local_date"]) >= _aware(now).date()]
    return {
        "source_url": FED_CALENDAR_URL,
        "docs_url": FED_FOMC_PAGE,
        "policy_event_count": len(policy),
        "window_events": in_window,
        "upcoming_events": upcoming,
        "ok": bool(in_window),
        "note": "Official Board calendar JSON. Only FOMC and Beige Book events are treated as the required Fed policy set. Speeches and other items are omitted. Times are published clock times assumed America/New_York.",
    }


def parse_bls_schedule_html(html: str, *, year: int, month: int) -> dict[str, Any]:
    parser = _TableParser()
    parser.feed(html)
    weekday_header = None
    body_rows = []
    for table in parser.tables:
        if table and [c.strip() for c in table[0][:5]] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            weekday_header = table[0]
            body_rows = table[1:]
            break
    if not weekday_header:
        raise ValueError("bls monthly schedule table not found")
    releases = []
    holidays = []
    current_year, current_month = year, month
    for row in body_rows:
        for cell in row:
            text = html_lib.unescape(cell or "").strip()
            if not text:
                continue
            match = re.match(r"^(\d{1,2})(.*)$", text, re.S)
            if not match:
                continue
            day = int(match.group(1))
            rest = match.group(2).strip()
            if day < 1:
                continue
            # Trailing days of the next month appear at the end of the grid.
            if body_rows.index(row) > 0 and day <= 7 and current_month == month and row is body_rows[-1]:
                pass
            event_month, event_year = current_month, current_year
            # If this is the last row and day is small, it may belong to next month.
            if row is body_rows[-1] and day <= 7 and any(re.match(r"^(\d{1,2})", c or "") and int(re.match(r"^(\d{1,2})", c or "").group(1)) > 20 for c in row if c and re.match(r"^(\d{1,2})", c)):
                # mixed row: keep large days in current month, small in next
                if day <= 7:
                    event_month = month + 1 if month < 12 else 1
                    event_year = year if month < 12 else year + 1
            if not rest:
                continue
            if HOLIDAY_RE.search(rest) and not RELEASE_RE.search(rest):
                holidays.append({"date": f"{event_year:04d}-{event_month:02d}-{day:02d}", "name": rest})
                continue
            for found in RELEASE_RE.finditer(rest):
                name, period, time_text = found.group(1).strip(), found.group(2).strip(), found.group(3).strip()
                releases.append({
                    "agency": "bls",
                    "title": name,
                    "reference_period": period,
                    "scheduled_local_date": f"{event_year:04d}-{event_month:02d}-{day:02d}",
                    "published_time_text": time_text,
                    "timezone_assumption": "America/New_York",
                    "scheduled_at_eastern_iso": _eastern_iso(event_year, event_month, day, time_text),
                })
    title_match = re.search(r"Schedule of Selected Releases for\s+([A-Za-z]+)\s+(\d{4})", html, re.I)
    return {
        "schedule_title": title_match.group(0) if title_match else None,
        "year": year,
        "month": month,
        "releases": releases,
        "holidays": holidays,
        "ok": bool(releases),
    }


def parse_bea_ics(text: str, *, now: datetime) -> dict[str, Any]:
    # Unfold ICS lines.
    unfolded = re.sub(r"\r\n[ \t]", "", text.replace("\r\n", "\n").replace("\r", "\n"))
    unfolded = re.sub(r"\n[ \t]", "", unfolded)
    events = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", unfolded, re.S):
        fields = dict(re.findall(r"^([^:\n]+):(.*)$", block, re.M))
        raw_start = fields.get("DTSTART;VALUE=DATE-TIME") or fields.get("DTSTART") or ""
        raw_start = raw_start.strip()
        scheduled = None
        if re.match(r"^\d{8}T\d{6}Z$", raw_start):
            scheduled = datetime.strptime(raw_start, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        elif re.match(r"^\d{8}T\d{6}$", raw_start):
            scheduled = datetime.strptime(raw_start, "%Y%m%dT%H%M%S").replace(tzinfo=EASTERN).isoformat()
        elif re.match(r"^\d{8}$", raw_start):
            scheduled = datetime.strptime(raw_start, "%Y%m%d").replace(tzinfo=EASTERN).isoformat()
        summary = html_lib.unescape((fields.get("SUMMARY") or "").replace("\\,", ",").replace("\\n", " ")).strip()
        if not summary or not scheduled:
            continue
        events.append({
            "agency": "bea",
            "title": summary,
            "scheduled_at_utc": scheduled if scheduled.endswith("Z") or "+" in scheduled or scheduled.endswith("+00:00") else None,
            "scheduled_at_iso": scheduled,
            "source_url": BEA_ICS_URL,
        })
    now_u = _aware(now)
    window_start = now_u - timedelta(days=7)
    window_end = now_u + timedelta(days=90)
    def _dt(item: dict[str, Any]) -> datetime:
        return datetime.fromisoformat(item["scheduled_at_iso"])
    in_window = [e for e in events if window_start <= _dt(e) <= window_end]
    upcoming = [e for e in in_window if _dt(e) >= now_u]
    return {
        "source_url": BEA_ICS_URL,
        "docs_url": BEA_PAGE,
        "event_count": len(events),
        "window_events": in_window,
        "upcoming_events": upcoming,
        "ok": bool(in_window),
        "note": "Official BEA release-calendar ICS subscription. Timed events in the file use the published UTC or Eastern values; this is a scheduled calendar, not a verified actual release timestamp.",
    }


def _window_ok(items: list[dict[str, Any]]) -> bool:
    return bool(items)


def fetch_official_event_calendar(timeout: float = 20.0, *, now: datetime | None = None, client: httpx.Client | None = None, snapshot_dir: Any = None) -> dict[str, Any]:
    fixed_time = now is not None
    retrieved = _aware(now)
    out: dict[str, Any] = {
        "kind": "official_us_fed_bls_bea_current_calendars",
        "applies_to": ["BTCUSDT", "XAUUSDT", "NDX100USDT"],
        "retrieved_at_utc": retrieved.isoformat(),
        "historical_replay_supported": False,
        "alpha_claim": False,
        "sources": {},
        "errors": {},
        "raw_sha256": {},
        "complete": False,
        "partial": True,
        "usable_for_research": False,
        "measure": "published_official_release_and_policy_schedules",
        "not_measure": ["verified_intraday_release_timestamps", "non_us_central_bank_calendars", "earnings_calendars"],
    }
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "PIIA-research-lab/0.1"})
        close_client = True
    raw_store: dict[str, bytes] = {}
    year, month = retrieved.year, retrieved.month
    bls_url = f"https://www.bls.gov/schedule/{year}/{month:02d}_sched.htm"
    try:
        fed = _get(client, FED_CALENDAR_URL)
        raw_store["fed_calendar.json"] = fed.content
        out["raw_sha256"]["fed"] = hashlib.sha256(fed.content).hexdigest()
        out["sources"]["fed"] = parse_fed_calendar(fed.content, now=retrieved)
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        out["errors"]["fed"] = f"{type(exc).__name__}: {exc}"
    try:
        bls = _get(client, bls_url)
        raw_store["bls_schedule.html"] = bls.content
        out["raw_sha256"]["bls"] = hashlib.sha256(bls.content).hexdigest()
        parsed = parse_bls_schedule_html(bls.text, year=year, month=month)
        parsed["source_url"] = str(bls.url)
        parsed["requested_url"] = bls_url
        out["sources"]["bls"] = parsed
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        out["errors"]["bls"] = f"{type(exc).__name__}: {exc}"
    try:
        bea = _get(client, BEA_ICS_URL)
        raw_store["bea_calendar.ics"] = bea.content
        out["raw_sha256"]["bea"] = hashlib.sha256(bea.content).hexdigest()
        out["sources"]["bea"] = parse_bea_ics(bea.text, now=retrieved)
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        out["errors"]["bea"] = f"{type(exc).__name__}: {exc}"
    finally:
        if close_client:
            client.close()

    fed_ok = bool((out["sources"].get("fed") or {}).get("ok"))
    bls_ok = bool((out["sources"].get("bls") or {}).get("ok"))
    bea_ok = bool((out["sources"].get("bea") or {}).get("ok"))
    out["present_sources"] = [name for name, flag in (("fed", fed_ok), ("bls", bls_ok), ("bea", bea_ok)) if flag]
    out["missing_sources"] = [name for name, flag in (("fed", fed_ok), ("bls", bls_ok), ("bea", bea_ok)) if not flag]
    out["complete"] = fed_ok and bls_ok and bea_ok and not out["errors"]
    out["partial"] = not out["complete"]
    out["usable_for_research"] = out["complete"]
    out["coverage_note"] = (
        "Official current U.S. calendars only: Federal Reserve Board calendar JSON "
        "(FOMC and Beige Book required), BLS monthly selected-releases schedule, and "
        "BEA release ICS. This is not a complete global macro calendar, not earnings, "
        "and not a confirmed tick-level publication time."
    )
    out["completed_at_utc"] = (retrieved if fixed_time else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None:
        from pathlib import Path
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name, body in raw_store.items():
            (dest / name).write_bytes(body)
        (dest / "context.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
