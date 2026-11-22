"""US spot Bitcoin ETF net fund-flow context from the public Farside Investors page.

Farside compiles issuer disclosures into a daily flow table in US$ millions.
This is not official SEC daily filings, not fund AUM, and not BTC price.
The page is fetched for local research snapshots only; the table is not
republished in the repository. Completeness requires fund-level coverage on
a reported day, not a single total cell, and dashes are unreported, not zero.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

SOURCE_URL = "https://farside.co.uk/btc/"
ALLOWED_HOSTS = {"farside.co.uk", "www.farside.co.uk"}
UNITS = "millions_of_usd"
EXPECTED_FUNDS = (
    "IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC",
)
COMPLETE_MAX_AGE_DAYS = 7
SKIP_ROW_PREFIXES = ("fee", "total", "average", "maximum", "minimum")
DATE_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$")
PAGE_DATE_RE = re.compile(r"Today's date is\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.I)
MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}


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
            text = html_lib.unescape(re.sub(r"\s+", " ", "".join(self._cell))).replace("\xa0", " ").strip()
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


def parse_flow_value(text: str) -> float | None:
    raw = html_lib.unescape(text or "").replace("\xa0", " ").strip()
    if raw in {"", "-", "–", "—", "n/a", "N/A", "."}:
        return None
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg = True
        raw = raw[1:-1].strip()
    raw = raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if neg else value


def parse_flow_date(text: str) -> str | None:
    raw = html_lib.unescape(text or "").replace("\xa0", " ").strip()
    match = DATE_RE.match(raw)
    if not match:
        return None
    day, mon, year = match.groups()
    month = MONTHS.get(mon[:3].title())
    if not month:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def _is_skip_label(text: str) -> bool:
    return text.strip().lower().startswith(SKIP_ROW_PREFIXES)


def parse_farside_btc_html(html: str) -> dict[str, Any]:
    parser = _TableParser()
    parser.feed(html)
    chosen: list[list[str]] | None = None
    header_idx = 0
    for table in parser.tables:
        for i, row in enumerate(table):
            upper = [c.strip().upper() for c in row]
            known = {"IBIT", "FBTC", "BITB", "ARKB", "GBTC", "BTCO", "HODL"}
            if len(set(upper) & known) >= 3:
                chosen = table
                header_idx = i
                break
        if chosen:
            break
    if not chosen:
        raise ValueError("farside btc flow table not found")
    header = chosen[header_idx]
    funds: list[str] = []
    fund_cols: list[int] = []
    total_col: int | None = None
    date_col = 0
    date_col_locked = False
    for i, cell in enumerate(header):
        name = cell.strip().upper()
        if not date_col_locked and (name == "DATE" or (i == 0 and name in {"", "DATE"})):
            date_col = i
            date_col_locked = True
            continue
        if name == "TOTAL":
            total_col = i
            continue
        if re.fullmatch(r"[A-Z0-9.]{2,6}", name):
            funds.append(name)
            fund_cols.append(i)
    if total_col is None:
        # Live Farside page puts "Total" on a previous header row and leaves the last cell blank.
        last = len(header) - 1
        if last >= 0 and last not in fund_cols and last != date_col:
            total_col = last
    if not funds:
        raise ValueError("no fund tickers in farside header")
    days: list[dict[str, Any]] = []
    for row in chosen[header_idx + 1:]:
        if not row:
            continue
        label = row[date_col] if date_col < len(row) else ""
        if _is_skip_label(label):
            continue
        date = parse_flow_date(label)
        if not date:
            continue
        fund_flows: dict[str, float | None] = {}
        reported = 0
        missing = 0
        for fund, col in zip(funds, fund_cols):
            cell = row[col] if col < len(row) else ""
            value = parse_flow_value(cell)
            fund_flows[fund] = value
            if value is None:
                missing += 1
            else:
                reported += 1
        total_reported = parse_flow_value(row[total_col]) if total_col is not None and total_col < len(row) else None
        known = [v for v in fund_flows.values() if v is not None]
        complete = missing == 0 and reported == len(funds)
        days.append({
            "flow_date": date,
            "fund_flows_millions_usd": fund_flows,
            "reported_fund_count": reported,
            "unreported_fund_count": missing,
            "sum_of_reported_funds_millions_usd": round(sum(known), 4) if known else None,
            "table_total_millions_usd": total_reported,
            "complete": complete,
            "partial": not complete,
        })
    page_date = None
    match = PAGE_DATE_RE.search(html)
    if match:
        page_date = parse_flow_date(match.group(1)) or match.group(1)
    # The public table may be oldest-first or newest-first; always select by date.
    ordered = sorted(days, key=lambda item: item["flow_date"], reverse=True)
    latest = ordered[0] if ordered else None
    latest_complete = next((item for item in ordered if item["complete"]), None)
    coverage_missing = [f for f in EXPECTED_FUNDS if f not in funds]
    extra = [f for f in funds if f not in EXPECTED_FUNDS]
    return {
        "funds": funds,
        "days": days,
        "latest_row": latest,
        "latest_complete_row": latest_complete,
        "page_date": page_date,
        "expected_funds": list(EXPECTED_FUNDS),
        "coverage_missing_expected_funds": coverage_missing,
        "additional_funds": extra,
        "disclaimer_excerpt": "Automatically generated. Farside Investors is not responsible or liable for any errors, omissions or inaccuracies.",
    }


def _quality(parsed: dict[str, Any], retrieved: datetime) -> dict[str, Any]:
    latest_complete = parsed.get("latest_complete_row")
    latest = parsed.get("latest_row")
    stale = True
    complete = False
    complete_date = latest_complete["flow_date"] if latest_complete else None
    latest_date = latest["flow_date"] if latest else None
    impossible = bool(latest_date and datetime.fromisoformat(latest_date).date() > retrieved.date())
    age_days = None
    if complete_date:
        asof = datetime.fromisoformat(complete_date).date()
        age_days = (retrieved.date() - asof).days
        nominal_close = datetime.combine(asof, time(16), tzinfo=ZoneInfo("America/New_York"))
        impossible = impossible or asof > retrieved.date() or retrieved < nominal_close
        stale = not 0 <= age_days <= COMPLETE_MAX_AGE_DAYS
        complete = (not stale) and not impossible and not parsed["coverage_missing_expected_funds"]
    latest_partial = bool(latest) and not latest.get("complete")
    return {
        "complete": complete,
        "partial": (not complete) or latest_partial,
        "stale": stale,
        "latest_flow_date": latest["flow_date"] if latest else None,
        "latest_complete_flow_date": complete_date,
        "latest_row_is_partial": latest_partial,
        "usable_for_btc_research": complete,
        "freshness_window_days": COMPLETE_MAX_AGE_DAYS,
        "latest_complete_age_calendar_days": age_days,
        "latest_exchange_session_verified": False,
        "future_or_unclosed_session": impossible,
        "note": (
            "Usable only when a complete fund-level day exists inside the freshness window "
            "and expected US spot tickers are present. A later partial day (issuer dashes) "
            "is kept as lag, not coerced to zero, and does not replace the last complete day. "
            "Table totals include compiler rounding; row completeness does not prove the latest exchange session is covered. "
            "A 16:00 New York cutoff is conservative on early-close days and is not a verified publication time."
        ),
    }


def fetch_us_spot_btc_etf_flows(timeout: float = 20.0, *, now: datetime | None = None, client: httpx.Client | None = None, snapshot_dir: Any = None) -> dict[str, Any]:
    fixed_time = now is not None
    retrieved = now or datetime.now(timezone.utc)
    if retrieved.tzinfo is None:
        raise ValueError("timezone-aware time required")
    retrieved = retrieved.astimezone(timezone.utc)
    out: dict[str, Any] = {
        "kind": "farside_us_spot_btc_etf_net_fund_flows",
        "asset": "BTC",
        "applies_to": ["BTCUSDT"],
        "not_applicable_to": ["XAUUSDT", "NDX100USDT"],
        "source_host": "farside.co.uk",
        "source_url": SOURCE_URL,
        "source_role": "third_party_compiler_of_issuer_disclosures",
        "official_sec_daily_filing": False,
        "units": UNITS,
        "measure": "net_fund_flows",
        "not_measure": ["aum", "nav_price", "btc_price", "share_price"],
        "retrieved_at_utc": retrieved.isoformat(),
        "historical_replay_supported": False,
        "alpha_claim": False,
        "errors": {},
        "complete": False,
        "partial": True,
        "stale": True,
        "usable_for_btc_research": False,
    }
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "PIIA-research-lab/0.1"})
        close_client = True
    try:
        host = urlparse(SOURCE_URL).hostname
        if host not in ALLOWED_HOSTS:
            raise ValueError("refusing non-allowlisted host")
        response = client.get(SOURCE_URL)
        if urlparse(str(response.url)).hostname not in ALLOWED_HOSTS:
            raise ValueError("redirected off allowlisted host")
        response.raise_for_status()
        raw = response.content
        out["http_status"] = response.status_code
        out["final_url"] = str(response.url)
        out["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        out["raw_bytes"] = len(raw)
        if snapshot_dir is not None:
            from pathlib import Path
            dest = Path(snapshot_dir)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "farside_btc.html").write_bytes(raw)
        parsed = parse_farside_btc_html(response.text)
        quality = _quality(parsed, retrieved)
        latest = parsed.get("latest_complete_row") or {}
        out.update({
            "page_date": parsed.get("page_date"),
            "funds": parsed["funds"],
            "expected_funds": parsed["expected_funds"],
            "coverage_missing_expected_funds": parsed["coverage_missing_expected_funds"],
            "additional_funds": parsed["additional_funds"],
            "latest_row": parsed.get("latest_row"),
            "latest_complete_row": parsed.get("latest_complete_row"),
            "complete_daily_flows": [
                {
                    "flow_date": day["flow_date"],
                    "sum_of_reported_funds_millions_usd": day["sum_of_reported_funds_millions_usd"],
                    "complete": True,
                }
                for day in sorted((d for d in parsed.get("days") or [] if d.get("complete")),
                                  key=lambda item: item["flow_date"])
            ],
            "disclaimer_excerpt": parsed.get("disclaimer_excerpt"),
            "coverage_note": (
                "Farside public US Bitcoin ETF flow table, units US$m net fund flows derived "
                "from issuer disclosures. This is a compiler, not a primary SEC official daily "
                "release. Coverage is the listed tickers on the page; a Total row is cumulative "
                "and is not a daily flow. AUM and share price are not taken from this table."
            ),
        })
        out.update(quality)
        if latest:
            out["asof_complete_flow_date"] = latest.get("flow_date")
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        out["errors"]["fetch"] = type(exc).__name__
        out["errors"]["detail"] = str(exc)
        out["complete"] = False
        out["usable_for_btc_research"] = False
    finally:
        if close_client:
            client.close()
    out["completed_at_utc"] = (retrieved if fixed_time else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None:
        from pathlib import Path
        import json
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "context.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
