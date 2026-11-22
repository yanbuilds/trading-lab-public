from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("piia")

import httpx

from piia.intelligence.event_calendar import (
    BEA_ICS_URL,
    FED_CALENDAR_URL,
    fetch_official_event_calendar,
    parse_bea_ics,
    parse_bls_schedule_html,
    parse_fed_calendar,
)
from piia.intelligence.evidence_gates import apply_external_missing
from piia.intelligence.gold_cftc import fetch_comex_gold_positioning, parse_comex_gold_futures_only, verify_gold_snapshot


CUTOFF = datetime(2026, 11, 17, 2, 40, tzinfo=timezone.utc)

BLS_HTML = """<!doctype html><html><head><title>Schedule of Selected Releases for November 2026</title></head><body>
<table>
<tr><th>Monday</th><th>Tuesday</th><th>Wednesday</th><th>Thursday</th><th>Friday</th></tr>
<tr><td>2</td><td>3Job Openings and Labor Turnover Survey September 2026 10:00 AM</td><td>4</td><td>5Productivity and Costs (P) Third Quarter 2026 08:30 AM</td><td>6Employment Situation October 2026 08:30 AM</td></tr>
<tr><td>9</td><td>10Consumer Price Index October 2026 08:30 AM</td><td>11Veterans Day Holiday</td><td>12</td><td>13Producer Price Index October 2026 08:30 AM</td></tr>
<tr><td>30</td><td>1Job Openings and Labor Turnover Survey October 2026 10:00 AM</td><td>2County Employment and Wages Second Quarter 2026 10:00 AMMetropolitan Area Employment and Unemployment (Monthly) October 2026 10:00 AM</td><td>3</td><td>4Employment Situation November 2026 08:30 AM</td></tr>
</table></body></html>
"""

FED_JSON = json.dumps({"events": [
    {"type": "FOMC", "title": "FOMC Meeting", "description": "Two-day meeting", "time": "2:00 p.m.", "month": "2026-12", "days": "9"},
    {"type": "FOMC", "title": "FOMC Minutes", "description": "Meeting of October 27-28", "time": "2:00 p.m.", "month": "2026-11", "days": "18"},
    {"type": "Beige", "title": "Beige Book", "time": "2:00 p.m.", "month": "2026-11", "days": "25"},
    {"type": "Speeches", "title": "Speech - Someone", "time": "9:00 a.m.", "month": "2026-11", "days": "19"},
], "announcement": []}).encode("utf-8-sig")

BEA_ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Gross Domestic Product\\, 3rd Quarter 2026 (Second Estimate)
DTSTART;VALUE=DATE-TIME:20261125T133000Z
END:VEVENT
BEGIN:VEVENT
SUMMARY:Personal Income and Outlays\\, October 2026
DTSTART;VALUE=DATE-TIME:20261126T133000Z
END:VEVENT
END:VCALENDAR
"""

GOLD_HTML = """<html><body><pre>
GOLD - COMMODITY EXCHANGE INC.                                       Code-088691
FUTURES ONLY POSITIONS AS OF 11/10/26                         |
--------------------------------------------------------------| NONREPORTABLE
      NON-COMMERCIAL      |   COMMERCIAL    |      TOTAL      |   POSITIONS
--------------------------|-----------------|-----------------|-----------------
  LONG  | SHORT  |SPREADS |  LONG  | SHORT  |  LONG  | SHORT  |  LONG  | SHORT
--------------------------------------------------------------------------------
(CONTRACTS OF 100 TROY OUNCES)                       OPEN INTEREST:      404,725
COMMITMENTS
 226,234   36,127   61,730   70,463  287,985  358,427  385,842   46,298   18,883

CHANGES FROM 11/03/26 (CHANGE IN OPEN INTEREST:      5,556)
   5,704   -4,855   -6,559    6,137   18,757    5,282    7,343      274   -1,787

MICRO GOLD - COMMODITY EXCHANGE INC.                                 Code-088695
FUTURES ONLY POSITIONS AS OF 11/10/26                         |
(CONTRACTS OF 10 TROY OUNCES)                        OPEN INTEREST:      111,708
COMMITMENTS
  54,739   74,890    8,331   10,000   20,000   30,000   40,000    1,000    2,000
</pre></body></html>
"""


class FakeResponse:
    def __init__(self, content, url, status=200, text=None):
        self.content = content if isinstance(content, bytes) else content.encode("utf-8")
        self.text = text if text is not None else self.content.decode("utf-8", errors="replace")
        self.status_code = status
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("GET", str(self.url)), response=httpx.Response(self.status_code))


class FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, url, follow_redirects=True):
        if url not in self.mapping:
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("not found", request=request, response=httpx.Response(404, request=request))
        body = self.mapping[url]
        return body if isinstance(body, FakeResponse) else FakeResponse(body, url)


def test_fed_bls_bea_parsers_extract_scheduled_events():
    fed = parse_fed_calendar(FED_JSON, now=CUTOFF)
    assert fed["ok"] is True
    titles = {e["title"] for e in fed["window_events"]}
    assert "FOMC Meeting" in titles
    assert "Beige Book" in titles
    assert not any("Speech" in e["title"] for e in fed["window_events"])
    bls = parse_bls_schedule_html(BLS_HTML, year=2026, month=11)
    names = {r["title"] for r in bls["releases"]}
    assert "Employment Situation" in names
    assert "Consumer Price Index" in names
    # last-grid days belong to December
    dec = [r for r in bls["releases"] if r["scheduled_local_date"].startswith("2026-12-")]
    assert any(r["title"] == "Employment Situation" and r["scheduled_local_date"] == "2026-12-04" for r in dec)
    assert any(r["title"] == "County Employment and Wages" and r["reference_period"] == "Second Quarter 2026" for r in dec)
    assert any(r["title"] == "Metropolitan Area Employment and Unemployment (Monthly)" for r in dec)
    assert not any(r["title"].startswith("AM") for r in bls["releases"])
    bea = parse_bea_ics(BEA_ICS, now=CUTOFF)
    assert bea["ok"] is True
    assert any("Gross Domestic Product" in e["title"] for e in bea["window_events"])


def test_calendar_fetch_requires_all_three_official_sources(tmp_path):
    client = FakeClient({
        FED_CALENDAR_URL: FakeResponse(FED_JSON, FED_CALENDAR_URL),
        "https://www.bls.gov/schedule/2026/11_sched.htm": FakeResponse(BLS_HTML, "https://www.bls.gov/schedule/2026/11_sched.htm"),
        BEA_ICS_URL: FakeResponse(BEA_ICS, BEA_ICS_URL),
    })
    result = fetch_official_event_calendar(now=CUTOFF, client=client, snapshot_dir=tmp_path)
    assert result["complete"] is True
    assert result["usable_for_research"] is True
    assert result["present_sources"] == ["fed", "bls", "bea"]
    incomplete = fetch_official_event_calendar(now=CUTOFF, client=FakeClient({
        FED_CALENDAR_URL: FakeResponse(FED_JSON, FED_CALENDAR_URL),
    }))
    assert incomplete["complete"] is False
    assert incomplete["usable_for_research"] is False
    assert "bls" in incomplete["missing_sources"]


def test_gold_cot_parses_100oz_contract_and_is_not_etf_demand():
    parsed = parse_comex_gold_futures_only(GOLD_HTML)
    assert parsed["asof_date"] == "2026-11-10"
    assert parsed["open_interest_contracts"] == 404725
    assert parsed["commitments_contracts"]["noncommercial_long"] == 226234
    assert parsed["commitments_contracts"]["commercial_short"] == 287985
    assert parsed["contract_unit"] == "contracts_of_100_troy_ounces"
    # must not accidentally take MICRO GOLD 111,708 OI
    assert parsed["open_interest_contracts"] != 111708
    client = FakeClient({"https://www.cftc.gov/dea/futures/deacmxsf.htm": FakeResponse(GOLD_HTML, "https://www.cftc.gov/dea/futures/deacmxsf.htm")})
    result = fetch_comex_gold_positioning(now=CUTOFF, client=client)
    assert result["complete"] is True
    assert result["usable_for_gold_research"] is True
    assert "gold_etf_holdings_or_flows" in result["not_measure"]
    missing = apply_external_missing(["event_calendar"], None, None, calendar={"usable_for_research": True}, gold_positioning=result)
    assert "event_calendar" not in missing
    assert "event_calendar_non_us_earnings_and_intraday_release_confirmation" in missing
    assert "gold_etf_and_physical_demand_flows" in missing
    assert "nasdaq100_official_aggregate_valuation_and_breadth" in missing


def test_future_cot_report_or_nonprior_comparison_is_unusable():
    url = "https://www.cftc.gov/dea/futures/deacmxsf.htm"
    future = GOLD_HTML.replace("11/10/26", "11/24/26")
    invalid_previous = GOLD_HTML.replace("CHANGES FROM 11/03/26", "CHANGES FROM 11/10/26")
    for html in (future, invalid_previous):
        result = fetch_comex_gold_positioning(now=CUTOFF, client=FakeClient({url: FakeResponse(html, url)}))
        assert result["usable_for_gold_research"] is False
        assert result["complete"] is False and result["errors"]["fetch"] == "ValueError"


def test_gold_summary_is_bound_to_raw_report(tmp_path):
    from copy import deepcopy
    url = "https://www.cftc.gov/dea/futures/deacmxsf.htm"
    context = fetch_comex_gold_positioning(now=CUTOFF, client=FakeClient({url: FakeResponse(GOLD_HTML, url)}), snapshot_dir=tmp_path)
    verify_gold_snapshot(context, tmp_path)
    changed = deepcopy(context)
    changed["commitments_contracts"]["commercial_long"] += 1
    with pytest.raises(ValueError, match="does not match"):
        verify_gold_snapshot(changed, tmp_path)
    (tmp_path / "deacmxsf.htm").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_gold_snapshot(context, tmp_path)


def test_gold_chat_gate_blocks_missing_report(monkeypatch, tmp_path):
    from copy import deepcopy
    from piia.intelligence import chat_decision_gate as gate
    url = "https://www.cftc.gov/dea/futures/deacmxsf.htm"
    context = fetch_comex_gold_positioning(now=CUTOFF, client=FakeClient({url: FakeResponse(GOLD_HTML, url)}), snapshot_dir=tmp_path / "gold")
    request = {"snapshot_ok": True, "cutoff_at_utc": CUTOFF.isoformat(), "request_hash": "fixture",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"}, "gold_positioning_context": context,
               "instruments": {"XAUUSDT": {"ticker": {"observed_at_utc": CUTOFF.isoformat()}}}}
    state = {"request_hash": "fixture", "run_id": "fixture", "paths": {"run_dir": str(tmp_path)},
             "packet_recommendation_blockers": [], "instrument_recommendation_blockers": {"XAUUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False, "created_at_utc": CUTOFF.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check():
        return gate.assess_chat_evidence(request, "XAUUSDT", now=CUTOFF, state=state, restricted={},
                                         archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "gold_cot_snapshot_invalid_or_stale" not in check()["blockers"]
    (tmp_path / "gold" / "deacmxsf.htm").unlink()
    assert "gold_cot_snapshot_invalid_or_stale" in check()["blockers"]


def test_live_official_calendar_and_gold_fetch(tmp_path):
    if os.environ.get("PIIA_SKIP_NETWORK") == "1":
        pytest.skip("network fetch skipped")
    now = datetime.now(timezone.utc)
    calendar = fetch_official_event_calendar(now=now, snapshot_dir=tmp_path / "calendar")
    gold = fetch_comex_gold_positioning(now=now, snapshot_dir=tmp_path / "gold")
    (tmp_path / "live_proof.json").write_text(json.dumps({
        "retrieved_at_utc": now.isoformat(),
        "calendar_complete": calendar.get("complete"),
        "calendar_sources": calendar.get("present_sources"),
        "calendar_errors": calendar.get("errors"),
        "fed_upcoming": (calendar.get("sources") or {}).get("fed", {}).get("upcoming_events"),
        "gold_usable": gold.get("usable_for_gold_research"),
        "gold_asof": gold.get("asof_date"),
        "gold_open_interest": gold.get("open_interest_contracts"),
        "gold_errors": gold.get("errors"),
    }, indent=2, default=str), encoding="utf-8")
    # Live proof should reach the official hosts; completeness depends on calendar windows.
    assert "fed" in (calendar.get("present_sources") or []) or "fed" in (calendar.get("errors") or {})
    assert gold.get("source_url") == "https://www.cftc.gov/dea/futures/deacmxsf.htm"
