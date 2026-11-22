"""Current CFTC Commitments of Traders for COMEX gold futures.

Official U.S. government public report. This is futures-only positioning in
contracts of 100 troy ounces, not gold ETF holdings, not official inventory,
and not physical or central-bank demand. It is truthful partial coverage only.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

SOURCE_URL = "https://www.cftc.gov/dea/futures/deacmxsf.htm"
DOCS_URL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
ALLOWED_HOSTS = {"www.cftc.gov", "cftc.gov"}
STALE_DAYS = 14
GOLD_BLOCK_RE = re.compile(
    r"GOLD - COMMODITY EXCHANGE INC\.\s+Code-088691\s+"
    r"FUTURES ONLY POSITIONS AS OF\s+(\d{2}/\d{2}/\d{2}).*?"
    r"\(CONTRACTS OF 100 TROY OUNCES\)\s+OPEN INTEREST:\s+([0-9,]+)\s+"
    r"COMMITMENTS\s+"
    r"([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+"
    r"([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)",
    re.S,
)
CHANGE_RE = re.compile(r"CHANGES FROM\s+(\d{2}/\d{2}/\d{2})\s+\(CHANGE IN OPEN INTEREST:\s+(-?[0-9,]+)\)")


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _num(text: str) -> int:
    return int(text.replace(",", "").strip())


def _asof(text: str) -> str:
    month, day, year = [int(part) for part in text.split("/")]
    if year < 100:
        year += 2000
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_comex_gold_futures_only(html: str) -> dict[str, Any]:
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;", " ", text)
    # Split off MICRO GOLD so the first GOLD block is the 100 oz contract.
    head = re.split(r"MICRO GOLD - COMMODITY EXCHANGE INC\.", text, maxsplit=1)[0]
    match = GOLD_BLOCK_RE.search(head)
    if not match:
        raise ValueError("COMEX gold futures-only COT block not found")
    asof = _asof(match.group(1))
    labels = (
        "noncommercial_long",
        "noncommercial_short",
        "noncommercial_spreads",
        "commercial_long",
        "commercial_short",
        "total_reportable_long",
        "total_reportable_short",
        "nonreportable_long",
        "nonreportable_short",
    )
    commitments = {name: _num(match.group(i + 3)) for i, name in enumerate(labels)}
    change_asof = None
    oi_change = None
    change = CHANGE_RE.search(head)
    if change:
        change_asof = _asof(change.group(1))
        oi_change = _num(change.group(2))
    return {
        "market": "GOLD - COMMODITY EXCHANGE INC.",
        "cftc_code": "088691",
        "report": "futures_only",
        "contract_unit": "contracts_of_100_troy_ounces",
        "asof_date": asof,
        "open_interest_contracts": _num(match.group(2)),
        "commitments_contracts": commitments,
        "previous_asof_date": change_asof,
        "open_interest_change_contracts": oi_change,
        "source_url": SOURCE_URL,
        "docs_url": DOCS_URL,
        "note": "Official CFTC futures-only Commitments of Traders. Positions are contracts, not tonnes of physical gold and not ETF share creation/redemption.",
    }


def verify_gold_snapshot(context: dict, snapshot_dir) -> None:
    """Reparse saved report bytes; the report date is not a publication timestamp."""
    from pathlib import Path
    if context.get("usable_for_gold_research") is not True or context.get("complete") is not True:
        raise ValueError("complete gold positioning required")
    path = Path(snapshot_dir) / "deacmxsf.htm"
    if path.stat().st_size > 5_000_000:
        raise ValueError("gold snapshot too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != context.get("raw_sha256"):
        raise ValueError("gold snapshot hash mismatch")
    parsed = parse_comex_gold_futures_only(raw.decode("utf-8", errors="replace"))
    for name, value in parsed.items():
        if context.get(name) != value:
            raise ValueError("gold summary does not match raw report")
    retrieved = datetime.fromisoformat(context["retrieved_at_utc"].replace("Z", "+00:00"))
    if retrieved.tzinfo is None or retrieved.utcoffset() is None:
        raise ValueError("timezone required")
    asof = datetime.fromisoformat(parsed["asof_date"]).date()
    previous = datetime.fromisoformat(parsed["previous_asof_date"]).date() if parsed["previous_asof_date"] else None
    if not 0 <= (retrieved.astimezone(timezone.utc).date() - asof).days <= STALE_DAYS or (previous and previous >= asof):
        raise ValueError("invalid gold report date")


def fetch_comex_gold_positioning(timeout: float = 20.0, *, now: datetime | None = None, client: httpx.Client | None = None, snapshot_dir: Any = None) -> dict[str, Any]:
    fixed_time = now is not None
    retrieved = _aware(now)
    out: dict[str, Any] = {
        "kind": "cftc_comex_gold_futures_only_cot",
        "asset": "XAU",
        "applies_to": ["XAUUSDT"],
        "not_applicable_to": ["BTCUSDT", "NDX100USDT"],
        "retrieved_at_utc": retrieved.isoformat(),
        "historical_replay_supported": False,
        "alpha_claim": False,
        "measure": "comex_gold_futures_positioning",
        "not_measure": ["gold_etf_holdings_or_flows", "physical_or_central_bank_demand", "exchange_warehouse_inventory"],
        "source_url": SOURCE_URL,
        "docs_url": DOCS_URL,
        "errors": {},
        "complete": False,
        "partial": True,
        "stale": True,
        "usable_for_gold_research": False,
    }
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "PIIA-research-lab/0.1"})
        close_client = True
    raw = b""
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
        parsed = parse_comex_gold_futures_only(response.text)
        asof = datetime.fromisoformat(parsed["asof_date"]).date()
        previous = datetime.fromisoformat(parsed["previous_asof_date"]).date() if parsed["previous_asof_date"] else None
        if asof > retrieved.date() or (previous is not None and previous >= asof):
            raise ValueError("future or invalid COT report date")
        stale = (retrieved.date() - asof) > timedelta(days=STALE_DAYS)
        out.update(parsed)
        out["stale"] = stale
        out["complete"] = (not stale) and all(
            parsed["commitments_contracts"].get(name) is not None
            for name in ("noncommercial_long", "noncommercial_short", "commercial_long", "commercial_short")
        ) and parsed["open_interest_contracts"] > 0
        out["partial"] = not out["complete"]
        out["usable_for_gold_research"] = out["complete"]
        out["coverage_note"] = (
            "Partial gold evidence only: official CFTC COMEX futures-only positioning. "
            "It does not complete gold ETF holdings/net flows or physical/central-bank demand."
        )
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        out["errors"]["fetch"] = type(exc).__name__
        out["errors"]["detail"] = str(exc)
        out["complete"] = False
        out["usable_for_gold_research"] = False
    finally:
        if close_client:
            client.close()
    out["completed_at_utc"] = (retrieved if fixed_time else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None:
        from pathlib import Path
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if raw:
            (dest / "deacmxsf.htm").write_bytes(raw)
        (dest / "context.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
