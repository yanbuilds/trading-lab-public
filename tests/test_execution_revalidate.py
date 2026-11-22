from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("piia")

import httpx

from piia.intelligence.execution_revalidate import (
    FIXTURE_LABEL,
    cost_entry_and_exit,
    proposal_still_valid,
    revalidate_run,
    snapshot_frozen_files,
)
from piia.intelligence.paper.risk_file import load_owner_risk
from piia.intelligence.public_orderbook import (
    allowed_url,
    parse_orderbook,
    parse_ticker,
    walk_book,
    _get,
)
from piia.intelligence.research_packet import write_json


NOW = datetime(2026, 11, 17, 3, 20, tzinfo=timezone.utc)


def _book_payload(best_bid="100.0", best_ask="100.2", bid_size="10", ask_size="10", ts_ms=None, extra_asks=None, extra_bids=None):
    ts = ts_ms if ts_ms is not None else int(NOW.timestamp() * 1000)
    bids = [[best_bid, bid_size]] + ([["99.9", "10"]] if extra_bids is None else extra_bids)
    asks = [[best_ask, ask_size]] + ([["100.3", "10"]] if extra_asks is None else extra_asks)
    return {"code": "00000", "msg": "success", "requestTime": ts + 5, "data": {"bids": bids, "asks": asks, "ts": str(ts)}}


def _ticker_payload(symbol="BTCUSDT", last="100.1", mark="100.1", ts_ms=None):
    ts = ts_ms if ts_ms is not None else int(NOW.timestamp() * 1000)
    return {"code": "00000", "msg": "success", "requestTime": ts + 5, "data": {"symbol": symbol, "lastPr": last, "markPrice": mark, "indexPrice": last, "bidPr": "100.0", "askPr": "100.2", "fundingRate": "0.0001", "ts": str(ts)}}


def _contract(symbol="BTCUSDT", min_qty="0.0001", step="0.0001", min_notional="5", taker="0.0006", cap="100"):
    return {
        "symbol": symbol, "status": "normal", "instrument_type": "perpetual",
        "min_qty": min_qty, "qty_step": step, "min_notional": min_notional, "taker": taker,
        "contract_size": "1", "max_market_order_qty": cap,
    }


def _books(now=NOW, symbols=("BTCUSDT", "XAUUSDT", "NDX100USDT"), mutate=None):
    instruments = {}
    for symbol, mid in zip(symbols, ("100.1", "2000.0", "25000")):
        bid = str(Decimal(mid) - Decimal("0.1")) if symbol != "NDX100USDT" else str(Decimal(mid) - Decimal("5"))
        ask = str(Decimal(mid) + Decimal("0.1")) if symbol != "NDX100USDT" else str(Decimal(mid) + Decimal("5"))
        size = "50" if symbol != "NDX100USDT" else "5"
        book = parse_orderbook(_book_payload(best_bid=bid, best_ask=ask, bid_size=size, ask_size=size, extra_bids=[[str(Decimal(bid)-1), "50"]], extra_asks=[[str(Decimal(ask)+1), "50"]]), symbol=symbol, source_url=f"https://api.bitget.com/api/v2/mix/market/orderbook?symbol={symbol}", retrieved=now, raw=b"book"+symbol.encode())
        ticker = parse_ticker(_ticker_payload(symbol=symbol, last=mid, mark=mid), symbol=symbol, source_url=f"https://api.bitget.com/api/v2/mix/market/ticker?symbol={symbol}", retrieved=now, raw=b"tick"+symbol.encode())
        instruments[symbol] = {"symbol": symbol, "ok": book["ok"] and ticker["ok"], "orderbook": book, "ticker": ticker, "contract": _contract(symbol, min_qty="0.01" if symbol=="XAUUSDT" else "0.0001", step="0.01" if symbol=="XAUUSDT" else "0.0001")}
    payload = {"ok": True, "retrieved_at_utc": now.isoformat(), "instruments": instruments, "errors": {}}
    if mutate:
        mutate(payload)
    return payload


def _risk():
    cfg, digest = load_owner_risk()
    return cfg, digest



def _tier_snapshot(symbol="BTCUSDT", now=NOW, end="200000", leverage="150", mmr="0.0040"):
    ts = int(now.timestamp() * 1000)
    payload = {"code": "00000", "msg": "success", "requestTime": ts, "data": [
        {"symbol": symbol, "level": "1", "startUnit": "0", "endUnit": end, "leverage": leverage, "keepMarginRate": mmr},
        {"symbol": symbol, "level": "2", "startUnit": end, "endUnit": "1000000", "leverage": "100", "keepMarginRate": "0.0050"},
    ]}
    from piia.intelligence.public_position_tiers import parse_position_tiers
    return parse_position_tiers(payload, symbol=symbol, source_url=f"https://api.bitget.com/api/v2/mix/market/query-position-lever?productType=USDT-FUTURES&symbol={symbol}", retrieved=now, raw=b"tiers"+symbol.encode())


def _tiers(now=NOW, symbols=("BTCUSDT", "XAUUSDT", "NDX100USDT")):
    inst = {}
    ends = {"BTCUSDT": ("200000", "150", "0.0040"), "XAUUSDT": ("50000", "100", "0.0050"), "NDX100USDT": ("10000", "50", "0.0100")}
    for symbol in symbols:
        end, lev, mmr = ends[symbol]
        inst[symbol] = _tier_snapshot(symbol, now=now, end=end, leverage=lev, mmr=mmr)
    return {"ok": True, "instruments": inst, "errors": {}, "retrieved_at_utc": now.isoformat()}

def _write_cycle(tmp_path, *, direction="flat", action="need_more_evidence", stop=None, close_at=None, expires=None, fixture=False, extra_request=None):
    run = tmp_path / "run"
    close_at = close_at or (NOW - timedelta(minutes=30)).isoformat()
    ticker_at = (NOW - timedelta(minutes=1)).isoformat()
    inst = {}
    for symbol, iid in (("BTCUSDT", "btcusdt-bitget-perp"), ("XAUUSDT", "xauusdt-bitget-perp"), ("NDX100USDT", "ndx100usdt-bitget-perp")):
        inst[symbol] = {
            "instrument_id": iid,
            "ticker": {"last": 100, "mark": 100, "observed_at_utc": ticker_at, "source_url": "https://api.bitget.com/fixture"},
            "last_closed_close_at_utc": close_at,
            "evidence_hash": "a" * 64,
        }
        if expires:
            inst[symbol]["thesis_expires_at_utc"] = expires
    cfg, digest = _risk()
    request = {
        "packet_type": "piia_chat_assisted_research_request",
        "integrity_version": 2,
        "snapshot_ok": True,
        "request_hash": "b" * 64,
        "cutoff_at_utc": ticker_at,
        "paper_risk_context": {"available": True, "owner_config_hash": digest, "captured_at_utc": ticker_at},
        "instruments": inst,
        "missing_datasets": [],
    }
    if extra_request:
        request.update(extra_request)
    write_json(run / "evidence" / "request.json", request)
    write_json(run / "STATE.json", {
        "run_id": "fixture-run",
        "phase": "imported_unapproved",
        "request_hash": request["request_hash"],
        "restricted_packet_hash": "c" * 64,
        "packet_recommendation_blockers": [],
        "instrument_recommendation_blockers": {"BTCUSDT": [], "XAUUSDT": [], "NDX100USDT": []},
        "plans_created": [],
    })
    write_json(run / "dispatch" / "RESTRICTED_EVIDENCE.json", {"restricted_packet_hash": "c" * 64, "request_hash": request["request_hash"]})
    synth_inst = {}
    imported = []
    for symbol in inst:
        item = {"direction": direction if symbol == "BTCUSDT" else "flat",
                "suggested_action": action if symbol == "BTCUSDT" else "need_more_evidence",
                "proposed_stop": stop if symbol == "BTCUSDT" else None}
        if expires:
            item["thesis_expires_at_utc"] = expires
        synth_inst[symbol] = item
        imported.append({
            "symbol": symbol, "direction": item["direction"], "suggested_action": item["suggested_action"],
            "proposed_stop": item["proposed_stop"], "thesis_id": "thesis-fixture", "revision_id": "rev-fixture",
            "research_run_id": "run-fixture", "plan": None, "approved": False, "executed": False,
        })
    write_json(run / "inbox" / "synthesizer.json", {"instruments": synth_inst, "role": "synthesizer"})
    write_json(run / "inbox" / "researcher.json", {"role": "researcher"})
    write_json(run / "inbox" / "critic_independent.json", {"role": "critic_independent"})
    write_json(run / "import" / "result.json", {"import": {"instruments": imported}, "plans_created": [], "approved": False, "quote_treated_as_current": False})
    if fixture:
        (run / "FIXTURE_LABEL.txt").write_text(FIXTURE_LABEL, encoding="utf-8")
    return run


def test_book_times_are_distinct_and_ssrf_is_rejected():
    payload = _book_payload(ts_ms=int(NOW.timestamp() * 1000))
    retrieved = NOW + timedelta(seconds=1)
    parsed = parse_orderbook(payload, symbol="BTCUSDT", source_url="https://api.bitget.com/api/v2/mix/market/orderbook?symbol=BTCUSDT", retrieved=retrieved, raw=b"abc")
    assert parsed["ok"] is True
    assert parsed["book_generated_at_utc"] != parsed["retrieved_at_utc"]
    assert parsed["endpoint_request_time_utc"] != parsed["book_generated_at_utc"]
    assert parsed["mark_is_not_executable"] is True
    assert allowed_url("https://api.bitget.com/api/v2/mix/market/orderbook") is True
    for url in ("http://127.0.0.1/api/v2/mix/market/orderbook", "https://evil.example/orderbook", "file:///etc/passwd"):
        assert allowed_url(url) is False
        with pytest.raises(ValueError, match="allowlisted"):
            _get(httpx.Client(), url)


def test_stale_future_crossed_and_illiquid_books_are_rejected():
    stale = parse_orderbook(_book_payload(ts_ms=int((NOW - timedelta(minutes=6)).timestamp() * 1000)), symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW)
    assert stale["ok"] is False and stale["reason"] == "stale_book"
    future = parse_orderbook(_book_payload(ts_ms=int((NOW + timedelta(minutes=2)).timestamp() * 1000)), symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW)
    assert future["reason"] == "future_book_timestamp"
    crossed = parse_orderbook(_book_payload(best_bid="101.0", best_ask="100.0"), symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW)
    assert crossed["reason"] == "crossed_book"
    empty = parse_orderbook({"code": "00000", "requestTime": int(NOW.timestamp()*1000), "data": {"bids": [], "asks": [], "ts": str(int(NOW.timestamp()*1000))}}, symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW)
    assert empty["reason"] == "empty_book"
    short = walk_book([{"price": "100.2", "size": "0.001"}], Decimal("1"))
    assert short["ok"] is False and short["reason"] == "insufficient_visible_depth"
    assert short["did_not_assume_unseen_tail"] is True


def test_wrong_side_and_precision_and_entry_uses_correct_book():
    book = parse_orderbook(_book_payload(), symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW)
    bad = cost_entry_and_exit(book, side="buy", qty=Decimal("0.01"), stop=Decimal("99"), taker=Decimal("0.0006"), contract_size=Decimal("1"))
    assert bad["reason"] == "wrong_side"
    long = cost_entry_and_exit(book, side="long", qty=Decimal("0.01"), stop=Decimal("99"), taker=Decimal("0.0006"), contract_size=Decimal("1"))
    assert long["ok"] is True
    assert long["entry_side"] == "asks"
    assert Decimal(long["entry_vwap"]) >= Decimal("100.2")
    short = cost_entry_and_exit(book, side="short", qty=Decimal("0.01"), stop=Decimal("101"), taker=Decimal("0.0006"), contract_size=Decimal("1"))
    assert short["entry_side"] == "bids"
    assert Decimal(short["entry_vwap"]) <= Decimal("100.0")


def test_flat_cycle_stays_no_plan(tmp_path):
    run = _write_cycle(tmp_path)
    before = snapshot_frozen_files(run)
    result = revalidate_run(run, now=NOW, books=_books(), fetch_public=False, fixture_label=None)
    assert result["approved"] is False
    assert result["plans_created"] == []
    for symbol, item in result["instruments"].items():
        assert item["decision"] == "structured_rejection"
        assert item["proposal"] is None
        assert "synthesizer_" in " ".join(item["research_blockers"])
        assert item["fresh_quote_cures_stale_research"] is False
    assert snapshot_frozen_files(run) == before
    assert result["frozen_files_unchanged"] is True


def test_unverified_role_chain_never_creates_nonfixture_proposal(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0")
    result = revalidate_run(run, now=NOW, books=_books(), tiers=_tiers(), fetch_public=False)
    assert result["plans_created"] == []
    assert "research_blocker:research_role_chain_unverified" in result["instruments"]["BTCUSDT"]["research_blockers"]


def test_fresh_quote_does_not_cure_stale_research(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", close_at=(NOW - timedelta(hours=3)).isoformat(), fixture=True)
    result = revalidate_run(run, now=NOW, books=_books(), fetch_public=False, fixture_label=FIXTURE_LABEL)
    btc = result["instruments"]["BTCUSDT"]
    assert "research_candle_stale_or_missing" in btc["research_blockers"]
    assert btc["decision"] == "structured_rejection"
    assert btc["proposal"] is None
    assert result["fresh_quote_cures_stale_research"] is False
    assert result["books_ok"] is True


def test_thesis_expired(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", expires=(NOW - timedelta(minutes=1)).isoformat(), fixture=True)
    result = revalidate_run(run, now=NOW, books=_books(), fetch_public=False, fixture_label=FIXTURE_LABEL)
    assert "thesis_expired" in result["instruments"]["BTCUSDT"]["research_blockers"]
    assert result["instruments"]["BTCUSDT"]["proposal"] is None


def test_missing_new_context(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    result = revalidate_run(run, now=NOW, books={"ok": False, "instruments": {}, "errors": {"fetch": "missing"}}, fetch_public=False, fixture_label=FIXTURE_LABEL)
    assert "missing_new_execution_context" in result["instruments"]["BTCUSDT"]["execution_blockers"] or "ticker_" in " ".join(result["instruments"]["BTCUSDT"]["execution_blockers"])


def test_tamper_and_unknown_hash(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    expected = snapshot_frozen_files(run)
    # Tamper frozen request after hashing.
    request = json.loads((run / "evidence" / "request.json").read_text())
    request["notes"] = "tampered"
    (run / "evidence" / "request.json").write_text(json.dumps(request), encoding="utf-8")
    result = revalidate_run(run, now=NOW, books=_books(), fetch_public=False, expected_frozen_hashes=expected, fixture_label=FIXTURE_LABEL)
    assert any(item.startswith("tampered_or_unknown_frozen_hash:") for item in result["instruments"]["BTCUSDT"]["research_blockers"])
    unknown = dict(expected)
    unknown["evidence/request.json"] = "d" * 64
    run2 = _write_cycle(tmp_path / "other", direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    result2 = revalidate_run(run2, now=NOW, books=_books(), fetch_public=False, expected_frozen_hashes=unknown, fixture_label=FIXTURE_LABEL)
    assert any("tampered_or_unknown_frozen_hash" in item for item in result2["instruments"]["BTCUSDT"]["research_blockers"])


def test_cross_symbol_book_is_rejected(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    books = _books()
    books["instruments"]["BTCUSDT"]["orderbook"]["symbol"] = "XAUUSDT"
    result = revalidate_run(run, now=NOW, books=books, fetch_public=False, fixture_label=FIXTURE_LABEL)
    assert "cross_symbol_book" in result["instruments"]["BTCUSDT"]["execution_blockers"]


def test_stale_ticker_in_new_context(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    books = _books()
    books["instruments"]["BTCUSDT"]["ticker"] = parse_ticker(
        _ticker_payload(ts_ms=int((NOW - timedelta(minutes=8)).timestamp() * 1000)),
        symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW,
    )
    books["instruments"]["BTCUSDT"]["ok"] = False
    result = revalidate_run(run, now=NOW, books=books, fetch_public=False, fixture_label=FIXTURE_LABEL)
    assert any("stale_ticker" in item for item in result["instruments"]["BTCUSDT"]["execution_blockers"])
    assert result["instruments"]["BTCUSDT"]["proposal"] is None


def test_risk_or_portfolio_change_invalidates(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    cfg, digest = _risk()
    ok = revalidate_run(run, now=NOW, books=_books(), tiers=_tiers(), fetch_public=False, current_risk=cfg, current_risk_hash=digest, fixture_label=FIXTURE_LABEL)
    assert ok["instruments"]["BTCUSDT"]["decision"] == "unapproved_paper_proposal"
    changed = revalidate_run(run, now=NOW, books=_books(), tiers=_tiers(), fetch_public=False, current_risk=cfg, current_risk_hash="e" * 64, fixture_label=FIXTURE_LABEL)
    assert "risk_config_changed" in changed["instruments"]["BTCUSDT"]["execution_blockers"]
    previous = ok["instruments"]["BTCUSDT"]["proposal"]
    valid = proposal_still_valid(previous, now=NOW + timedelta(minutes=1), current_bindings=previous["bindings"], current_book_hash=previous["bindings"].get("orderbook_raw_sha256"))
    assert valid["valid"] is True
    missing = proposal_still_valid(previous, now=NOW + timedelta(minutes=1), current_bindings={}, current_book_hash=None)
    assert not missing["valid"] and "book_hash_missing" in missing["reasons"]
    tampered = dict(previous, qty="999999")
    invalid = proposal_still_valid(tampered, now=NOW + timedelta(minutes=1), current_bindings=previous["bindings"], current_book_hash=previous["bindings"].get("orderbook_raw_sha256"))
    assert "proposal_hash_missing_or_mismatch" in invalid["reasons"]
    tier_changed = dict(previous["bindings"], position_tiers_raw_sha256="e" * 64)
    assert "position_tiers_raw_sha256_changed" in proposal_still_valid(previous, now=NOW + timedelta(minutes=1), current_bindings=tier_changed, current_book_hash=previous["bindings"].get("orderbook_raw_sha256"))["reasons"]
    later = proposal_still_valid(previous, now=NOW + timedelta(minutes=6), current_bindings=previous["bindings"], current_book_hash=previous["bindings"].get("orderbook_raw_sha256"))
    assert later["valid"] is False and "proposal_expired" in later["reasons"]
    moved = dict(previous["bindings"])
    moved["portfolio_state_hash"] = "f" * 64
    port = proposal_still_valid(previous, now=NOW + timedelta(minutes=1), current_bindings=moved, current_book_hash=previous["bindings"].get("orderbook_raw_sha256"))
    assert port["valid"] is False and "portfolio_state_hash_changed" in port["reasons"]
    assert later["approved"] is False


def test_fixture_positive_path_is_unapproved_and_labelled(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    cfg, digest = _risk()
    result = revalidate_run(run, now=NOW, books=_books(), tiers=_tiers(), fetch_public=False, current_risk=cfg, current_risk_hash=digest, fixture_label=FIXTURE_LABEL)
    btc = result["instruments"]["BTCUSDT"]
    assert result["fixture_label"] == FIXTURE_LABEL
    assert btc["decision"] == "unapproved_paper_proposal"
    proposal = btc["proposal"]
    assert proposal["approved"] is False
    assert proposal["executed"] is False
    assert proposal["actionable"] is False
    assert proposal["proposal_hash"]
    cloned = {k: v for k, v in proposal.items() if k != "proposal_hash"}
    from piia.intelligence.hashing import content_hash
    assert proposal["proposal_hash"] == content_hash(json.dumps(cloned, sort_keys=True, default=str).encode("utf-8"))
    # Other symbols stay flat / no plan.
    assert result["instruments"]["XAUUSDT"]["proposal"] is None
    assert result["approved"] is False
    # Frozen files still unchanged after writing execution_context.
    assert (run / "execution_context" / result["revalidation_id"] / "RESULT.json").is_file()
    # STATE and evidence still the fixture originals, not rewritten as approved.
    state = json.loads((run / "STATE.json").read_text())
    assert state.get("plans_created") == []


def test_illiquid_min_size_rejects_fixture_path(tmp_path):
    run = _write_cycle(tmp_path, direction="long", action="review_paper_plan", stop="99.0", fixture=True)
    books = _books()
    books["instruments"]["BTCUSDT"]["orderbook"] = parse_orderbook(
        _book_payload(ask_size="0.0000001", bid_size="0.0000001", extra_asks=[], extra_bids=[]),
        symbol="BTCUSDT", source_url="https://api.bitget.com/x", retrieved=NOW,
    )
    cfg, digest = _risk()
    result = revalidate_run(run, now=NOW, books=books, fetch_public=False, current_risk=cfg, current_risk_hash=digest, fixture_label=FIXTURE_LABEL)
    blockers = " ".join(result["instruments"]["BTCUSDT"]["execution_blockers"])
    assert "insufficient_visible_depth" in blockers or "size_" in blockers
    assert result["instruments"]["BTCUSDT"]["proposal"] is None
