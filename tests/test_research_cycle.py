from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("piia")

from piia.intelligence import research_cycle as cycle
from piia.intelligence.chronos_forecast import env_blocker, run_chronos_tiny
from piia.intelligence.errors import ValidationFailed
from piia.intelligence.macro_data import EXTENDED_SERIES, parse_vintage_csv
from piia.intelligence.research_packet import (
    PacketResponseIn, build_request, frozen_hashes_for_symbol, restricted_content_hash, validate_response,
)


CUTOFF = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)


def test_restricted_packet_distinguishes_forecast_archive_from_market_cutoff():
    request = {"request_hash": "fixture", "cutoff_at_utc": CUTOFF.isoformat(), "instruments": {}}
    archived = (CUTOFF + timedelta(seconds=30)).isoformat()
    packet = cycle._build_restricted(request, {"archive_receipt": {"recorded_at_utc": archived}})
    assert packet["cutoff_at_utc"] == CUTOFF.isoformat()
    assert packet["forecast_available_no_earlier_than_utc"] == archived
    assert cycle._build_restricted(request, {})["forecast_available_no_earlier_than_utc"] is None


def _candles(n=80, last=100.0):
    start = CUTOFF - timedelta(hours=n)
    rows = []
    px = last - n
    for i in range(n):
        open_at = start + timedelta(hours=i)
        close_at = open_at + timedelta(hours=1)
        o, c = px, px + 0.3
        rows.append({
            "open_at_utc": open_at.isoformat(), "close_at_utc": close_at.isoformat(),
            "open": o, "high": c + 0.2, "low": o - 0.2, "close": c, "volume": 1.0,
        })
        px = c
    return rows


def _snapshot(missing=None):
    candles = _candles()
    last = candles[-1]
    ticker = {
        "last": last["close"], "mark": last["close"], "index": last["close"],
        "funding_rate": "0.0001", "observed_at_utc": CUTOFF.isoformat(),
        "provenance": "fixture", "source_url": "https://api.bitget.com/fixture",
    }
    inst = {
        "instrument_id": "btcusdt-bitget-perp",
        "ticker": ticker,
        "funding": {"funding_rate": "0.0001", "provenance": "fixture", "retrieved_at_utc": CUTOFF.isoformat()},
        "candles_closed_1h": candles,
        "candles_closed_1d": [],
    }
    return {
        "ok": True,
        "retrieved_at_utc": CUTOFF.isoformat(),
        "cutoff_at_utc": CUTOFF.isoformat(),
        "missing_datasets": list(missing or ["onchain", "btc_etf_flows", "macro", "dxy", "tga", "rrp", "real_rates"]),
        "errors": [],
        "instruments": {
            "BTCUSDT": inst,
            "XAUUSDT": {**inst, "instrument_id": "xauusdt-bitget-perp"},
            "NDX100USDT": {**inst, "instrument_id": "ndx100usdt-bitget-perp"},
        },
    }


def _macro(stale=False):
    series = {}
    for name, meta in {
        "DGS10": ("percent_per_annum", "rates", 4.1),
        "DGS2": ("percent_per_annum", "rates", 3.8),
        "DFII10": ("percent_per_annum", "rates", 1.9),
        "DFF": ("percent_per_annum", "rates", 4.0),
        "DTWEXBGS": ("index", "dollar", 120.5),
        "RRPONTSYD": ("billions_of_dollars", "liquidity", 2.1),
        "WDTGAL": ("millions_of_dollars", "liquidity", 800000),
        "WALCL": ("millions_of_dollars", "liquidity", 6700000),
    }.items():
        series[name] = {
            "series_id": name, "unit": meta[0], "topic": meta[1], "value": meta[2],
            "observation_date": "2026-11-13", "vintage_date": "2026-11-14",
            "retrieved_at_utc": CUTOFF.isoformat(), "age_days": 3, "stale": stale,
            "source_page": f"https://fred.stlouisfed.org/series/{name}",
            "time_limit": "date-vintage only; not for historical replay",
        }
    return {
        "kind": "alfred_conservative_date_vintage_extended",
        "retrieved_at_utc": CUTOFF.isoformat(),
        "vintage_date": "2026-11-14",
        "historical_replay_supported": False,
        "intraday_release_verified": False,
        "series": series,
        "errors": {},
        "complete": not stale,
    }


def _forecast(ok=True):
    if not ok:
        return {"ok": False, "weights_ran": False, "reason": "forced_unavailable", "model_id": "amazon/chronos-t5-tiny"}
    def q(start):
        return [start + i * 0.1 for i in range(24)]
    item = {
        "ok": True, "weights_ran": True, "bars_used": 80, "last_close": _candles()[-1]["close"],
        "last_close_at_utc": _candles()[-1]["close_at_utc"], "timeframe": "1H_closed",
        "quantile_10": q(99), "quantile_50": q(100), "quantile_90": q(101),
        "num_samples": 20, "prediction_length": 24, "samples": [[100.0] * 24] * 2,
    }
    return {
        "ok": True, "weights_ran": True, "model_id": "amazon/chronos-t5-tiny",
        "revision": "29d808298f1a62493e7b9a5e08529d0d930fa189", "license": "apache-2.0",
        "device": "cpu", "gpu_used": False, "trust_remote_code": False,
        "training_leakage": "unknown", "calibration_verified": False, "alpha_claim": False,
        "forecasts": {s: dict(item) for s in ("BTCUSDT", "XAUUSDT", "NDX100USDT")},
    }


def _evidence_hashes(state):
    request = cycle._load_json(Path(state["paths"]["run_dir"]) / "evidence" / "request.json")
    return {symbol: raw["evidence_hash"] for symbol, raw in request["instruments"].items()}


def _role_payload(role, state, action="need_more_evidence", direction="flat"):
    inst = {}
    hashes = _evidence_hashes(state)
    for symbol in cycle.SYMBOLS:
        inst[symbol] = {
            "direction": direction,
            "summary": f"{role} test writeup for {symbol}.",
            "invalidation": "A close beyond the last frozen 1h extreme would invalidate this test stance.",
            "claims": [{"text": "uses frozen snapshot only", "evidence_hash": hashes[symbol], "status": "supported"}],
            "disagreements": [],
            "missing_data": ["onchain"],
            "suggested_action": action,
            "proposed_stop": 99.0 if action == "review_paper_plan" else None,
        }
    payload = {
        "packet_type": cycle.ROLE_RESPONSE,
        "role": role,
        "run_id": state["run_id"],
        "request_hash": state["request_hash"],
        "restricted_packet_hash": state["restricted_packet_hash"],
        "cutoff_at_utc": state["cutoff_at_utc"],
        "actor": "external_fresh_context",
        "generation": "external_fresh_context",
        "independent_run_mode": cycle.INDEPENDENT_RUN_MODE,
        "instruments": inst,
    }
    if role == "synthesizer":
        synth_path = Path(state["paths"]["run_dir"]) / "dispatch" / "SYNTHESIZER_REQUEST.json"
        if synth_path.is_file():
            payload["followup_request_hash"] = cycle._load_json(synth_path)["followup_request_hash"]
    return payload


def test_alfred_extended_parser_accepts_dollar_series():
    raw = b"observation_date,DTWEXBGS_20261114\n2026-11-12,120.5\n"
    result = parse_vintage_csv(raw, "DTWEXBGS", "2026-11-14", CUTOFF)
    assert result["unit"] == "index"
    assert result["topic"] == "dollar"
    assert "not the ICE" in (result["series_note"] or "")
    assert "DTWEXBGS" in EXTENDED_SERIES


def test_prepare_writes_isolated_role_files(tmp_path):
    state = cycle.prepare_run(
        tmp_path,
        snapshot=_snapshot(),
        macro=_macro(),
        forecast_payload=_forecast(),
    )
    assert state["phase"] == "awaiting_independent_roles"
    assert state["exploratory_evidence_archive"] is None  # synthetic injected inputs are never real observations
    assert state["forecast_weights_ran"] is True
    assert any("onchain" in r for r in state["trading_recommendation_blockers"])
    run_dir = Path(state["paths"]["run_dir"])
    researcher = cycle._load_json(run_dir / "dispatch" / "RESEARCHER_REQUEST.json")
    critic = cycle._load_json(run_dir / "dispatch" / "CRITIC_INDEPENDENT_REQUEST.json")
    assert cycle._evidence_only(researcher) == cycle._evidence_only(critic)
    assert "researcher_opinion" not in critic
    evidence = cycle._load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json")
    for inst in evidence["instruments"].values():
        assert "all_closed_candles" not in inst
        assert inst["candles_hash"]
        assert inst["candles_closed_1h_recent"]
    # samples must not leak into the dispatch file
    for item in evidence["forecast"]["forecasts"].values():
        assert "samples" not in item
    assert (run_dir / "dispatch" / "ROLE_SCHEMA.json").is_file()
    assert (run_dir / "dispatch" / "INSTRUCTIONS.md").is_file()
    assert "awaiting_independent_roles" == cycle.status(run_dir)["phase"]



def test_critic_independent_rejects_researcher_fields(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    payload = _role_payload("critic_independent", state)
    payload["researcher_opinion"] = {"leaked": True}
    with pytest.raises(ValueError, match="researcher"):
        cycle._validate_role_response(payload, role="critic_independent", state=state)


def test_followups_label_opinions_separately(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    (run_dir / "inbox" / "researcher.json").write_text(
        __import__("json").dumps(_role_payload("researcher", state)), encoding="utf-8")
    (run_dir / "inbox" / "critic_independent.json").write_text(
        __import__("json").dumps(_role_payload("critic_independent", state)), encoding="utf-8")
    follow = cycle.materialize_followups(run_dir)
    assert follow["phase"] == "awaiting_debate_and_synthesis"
    debate = cycle._load_json(run_dir / "dispatch" / "CRITIC_DEBATE_REQUEST.json")
    synth = cycle._load_json(run_dir / "dispatch" / "SYNTHESIZER_REQUEST.json")
    assert debate["researcher_opinion_labeled"]["not_evidence"] is True
    assert "frozen_evidence" in debate["allowed_context"]
    assert "researcher_opinion_labeled" in debate["allowed_context"]
    assert synth["critic_independent_opinion_labeled"]["not_evidence"] is True


def test_status_blocker_drift_refuses_followups_and_import(tmp_path):
    import json
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role}.json").write_text(json.dumps(_role_payload(role, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    (run_dir / "inbox" / "synthesizer.json").write_text(json.dumps(_role_payload("synthesizer", state)), encoding="utf-8")
    status_path = run_dir / "STATE.json"
    status = cycle._load_json(status_path)
    assert status["instrument_recommendation_blockers"]["BTCUSDT"]
    status["instrument_recommendation_blockers"]["BTCUSDT"] = []
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="state and restricted evidence differ"):
        cycle.materialize_followups(run_dir)
    with pytest.raises(ValueError, match="state and restricted evidence differ"):
        cycle.import_roles(run_dir, db_path=run_dir / "db" / "blocked.sqlite")
    assert not (run_dir / "db" / "blocked.sqlite").exists()


def test_synthesis_refuses_source_opinion_changed_after_followups(tmp_path):
    import json
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role}.json").write_text(json.dumps(_role_payload(role, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    (run_dir / "inbox" / "synthesizer.json").write_text(json.dumps(_role_payload("synthesizer", state)), encoding="utf-8")
    path = run_dir / "inbox" / "critic_independent.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instruments"]["BTCUSDT"]["summary"] = "Changed after synthesis request."
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches"):
        cycle.import_roles(run_dir, db_path=run_dir / "db" / "blocked.sqlite")
    assert not (run_dir / "db" / "blocked.sqlite").exists()


def test_regenerated_followup_rejects_synthesis_of_old_opinions(tmp_path):
    import json
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role}.json").write_text(json.dumps(_role_payload(role, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    synth = _role_payload("synthesizer", state)
    path = run_dir / "inbox" / "critic_independent.json"
    critic = json.loads(path.read_text(encoding="utf-8"))
    critic["instruments"]["BTCUSDT"]["summary"] = "A revised synthetic counterargument."
    path.write_text(json.dumps(critic), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    (run_dir / "inbox" / "synthesizer.json").write_text(json.dumps(synth), encoding="utf-8")
    with pytest.raises(ValueError, match="followup request hash"):
        cycle.import_roles(run_dir, db_path=run_dir / "db" / "blocked.sqlite")
    assert not (run_dir / "db" / "blocked.sqlite").exists()


def test_import_downgrades_recommendation_and_never_approves(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role}.json").write_text(
            __import__("json").dumps(_role_payload(role, state, action="stand_aside")), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    (run_dir / "inbox" / "synthesizer.json").write_text(
        __import__("json").dumps(_role_payload("synthesizer", state, action="review_paper_plan", direction="long")),
        encoding="utf-8")
    imported = cycle.import_roles(run_dir, db_path=run_dir / "db" / "cycle.sqlite")
    assert imported["phase"] == "imported_unapproved"
    from piia.intelligence.execution_revalidate import _role_chain_verified
    request = cycle._load_json(run_dir / "evidence" / "request.json")
    assert _role_chain_verified(run_dir, imported, request)
    result_path = run_dir / "import" / "result.json"
    original = result_path.read_text(encoding="utf-8")
    result_path.write_text(original + " ", encoding="utf-8")
    assert not _role_chain_verified(run_dir, imported, request)
    result_path.write_text(original, encoding="utf-8")
    assert imported["recommendation_downgrades"]
    payload = cycle._load_json(run_dir / "import" / "result.json")
    assert payload["approved"] is False
    assert payload["executed"] is False
    for inst in payload["import"]["instruments"]:
        assert inst.get("approved") is False
        plan = inst.get("plan")
        assert not plan or plan.get("approved") is False
    with pytest.raises(ValueError, match="not awaiting synthesis import"):
        cycle.import_roles(run_dir, db_path=run_dir / "db" / "cycle.sqlite")
    with pytest.raises(ValueError, match="not awaiting role followups"):
        cycle.materialize_followups(run_dir)


def test_missing_forecast_is_an_explicit_blocker(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast(ok=False))
    assert any("neural_forecast_unavailable" in r for r in state["trading_recommendation_blockers"])


@pytest.mark.parametrize("invalid", [None, {"ok": False, "weights_ran": False},
                                   {"ok": True, "weights_ran": True, "prediction_length": 1,
                                    "quantile_10": [float("nan")], "quantile_50": [1.0], "quantile_90": [2.0]},
                                   {"ok": True, "weights_ran": True, "prediction_length": 1,
                                    "quantile_10": [2.0], "quantile_50": [1.0], "quantile_90": [3.0]}])
def test_partial_or_invalid_forecast_blocks_only_affected_instrument(tmp_path, invalid):
    forecast = _forecast()
    if invalid is not None:
        invalid = {"last_close_at_utc": _candles()[-1]["close_at_utc"],
                   "last_close": _candles()[-1]["close"], "timeframe": "1H_closed", **invalid}
    forecast["forecasts"]["XAUUSDT"] = invalid
    # A failed item may be omitted instead of represented as null by a runner.
    if invalid is None:
        del forecast["forecasts"]["XAUUSDT"]
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=forecast)
    reasons = state["instrument_recommendation_blockers"]
    assert "neural_forecast_missing_or_invalid:XAUUSDT" in reasons["XAUUSDT"]
    assert not any("neural_forecast_missing_or_invalid" in r for r in reasons["NDX100USDT"])


@pytest.mark.parametrize("field,value", [("last_close_at_utc", "2026-01-01T00:00:00+00:00"),
                                        ("last_close", 99999.0), ("timeframe", "1D")])
def test_forecast_must_match_frozen_candle_input(tmp_path, field, value):
    forecast = _forecast()
    forecast["forecasts"]["BTCUSDT"][field] = value
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=forecast)
    assert "neural_forecast_missing_or_invalid:BTCUSDT" in state["instrument_recommendation_blockers"]["BTCUSDT"]


def test_run_chronos_reports_blocker_when_python_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("piia.intelligence.chronos_forecast.default_forecast_python", lambda: None)
    assert env_blocker()["reason"] == "forecast_env_python_not_found"
    out = run_chronos_tiny({"BTCUSDT": {"closes": [1.0] * 40}}, tmp_path / "out.json")
    assert out["weights_ran"] is False
    assert "forecast_env_python_not_found" in out["reason"]


def test_live_db_is_refused(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent", "synthesizer"):
        if role != "synthesizer":
            (run_dir / "inbox" / f"{role}.json").write_text(
                __import__("json").dumps(_role_payload(role, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    (run_dir / "inbox" / "synthesizer.json").write_text(
        __import__("json").dumps(_role_payload("synthesizer", state)), encoding="utf-8")
    from piia.intelligence.paths import DEFAULT_SQLITE
    with pytest.raises(ValueError, match="live research database"):
        cycle.import_roles(run_dir, db_path=DEFAULT_SQLITE)


def test_frozen_subhash_claims_import_and_fake_hashes_fail(tmp_path):
    import json
    from piia.intelligence.research_packet import InstrumentResponseIn, RoleIn, import_response
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    request = cycle._load_json(run_dir / "evidence" / "request.json")
    restricted = cycle._load_json(run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json")
    assert restricted_content_hash(restricted) == restricted["restricted_packet_hash"]
    btc = request["instruments"]["BTCUSDT"]
    allowed = frozen_hashes_for_symbol(request, "BTCUSDT", restricted_packet=restricted)
    assert restricted["restricted_packet_hash"] in allowed
    assert btc["candles_hash"] in allowed

    def role(name, ev):
        return {
            "role": name, "actor": "external_fresh_context", "generation": "external_fresh_context",
            "direction": "flat", "summary": f"{name} frozen-hash test writeup.",
            "invalidation": "A move through the frozen 1h extreme would invalidate this test stance.",
            "claims": [{"text": "cites frozen hash", "evidence_hash": ev, "status": "supported"}],
            "disagreements": [], "missing_data": ["onchain"], "suggested_action": "need_more_evidence",
            "proposed_stop": None,
        }

    for role_name, ev in (
        ("researcher", restricted["restricted_packet_hash"]),
        ("critic_independent", request["instruments"]["BTCUSDT"]["candles_hash"]),
        ("synthesizer", request["instruments"]["BTCUSDT"]["macro_context_hash"]),
    ):
        payload = _role_payload(role_name, state)
        for symbol in cycle.SYMBOLS:
            inst_hash = {
                "researcher": restricted["restricted_packet_hash"],
                "critic_independent": request["instruments"][symbol]["candles_hash"],
                "synthesizer": request["instruments"][symbol]["macro_context_hash"],
            }[role_name]
            payload["instruments"][symbol]["claims"] = [
                {"text": "cites frozen hash", "evidence_hash": inst_hash, "status": "supported"}
            ]
            payload["instruments"][symbol]["suggested_action"] = "need_more_evidence"
            payload["instruments"][symbol]["direction"] = "flat"
        (run_dir / "inbox" / f"{'critic_independent' if role_name=='critic_independent' else role_name}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    # researcher and critic written; followups then overwrite? write all three after followups
    cycle.materialize_followups(run_dir)
    # rewrite after followups so files stay as designed
    for role_name in ("researcher", "critic_independent", "synthesizer"):
        payload = _role_payload(role_name, state)
        for symbol in cycle.SYMBOLS:
            inst_hash = {
                "researcher": restricted["restricted_packet_hash"],
                "critic_independent": request["instruments"][symbol]["candles_hash"],
                "synthesizer": request["instruments"][symbol]["macro_context_hash"],
            }[role_name]
            payload["instruments"][symbol]["claims"] = [
                {"text": "cites frozen hash", "evidence_hash": inst_hash, "status": "supported"}
            ]
            payload["instruments"][symbol]["suggested_action"] = "need_more_evidence"
            payload["instruments"][symbol]["direction"] = "flat"
        filename = "critic_independent.json" if role_name == "critic_independent" else f"{role_name}.json"
        (run_dir / "inbox" / filename).write_text(json.dumps(payload), encoding="utf-8")
    imported = cycle.import_roles(run_dir, db_path=run_dir / "db" / "subhash.sqlite")
    assert imported["phase"] == "imported_unapproved"
    assert imported["plans_created"] == []
    assert imported["evidence_refreshed"] is False
    assert imported["quote_treated_as_current"] is False

    fake = _role_payload("researcher", state)
    fake["instruments"]["BTCUSDT"]["claims"] = [
        {"text": "unknown hash", "evidence_hash": "b" * 64, "status": "supported"}
    ]
    instruments = []
    from piia.intelligence.research_packet import InstrumentResponseIn
    # direct validate
    from copy import deepcopy
    response = {
        "packet_type": "piia_chat_assisted_research_response",
        "request_hash": request["request_hash"],
        "cutoff_at_utc": request["cutoff_at_utc"],
        "independent_llm_review": "NOT_WIRED",
        "actor": "test",
        "instruments": [],
    }
    def pack(claim_hash):
        insts = []
        for symbol in cycle.SYMBOLS:
            ev = request["instruments"][symbol]["evidence_hash"]
            h = claim_hash if symbol == "BTCUSDT" else ev
            body = {
                "role": "researcher", "actor": "t", "generation": "external_fresh_context",
                "direction": "flat", "summary": "researcher test writeup.",
                "invalidation": "Invalidated by a close through the last frozen extreme.",
                "claims": [{"text": "x", "evidence_hash": h, "status": "supported"}],
                "disagreements": [], "missing_data": [], "suggested_action": "need_more_evidence",
            }
            critic = dict(body, role="critic", summary="critic test writeup.")
            synth = dict(body, role="synthesizer", summary="synthesizer test writeup.")
            insts.append({
                "symbol": symbol, "request_evidence_hash": ev,
                "researcher": body, "critic": critic, "synthesizer": synth,
            })
        return {"packet_type": "piia_chat_assisted_research_response",
                "request_hash": request["request_hash"], "cutoff_at_utc": request["cutoff_at_utc"],
                "independent_llm_review": "NOT_WIRED", "actor": "test", "instruments": insts}
    with pytest.raises(ValidationFailed, match="claim evidence_hash"):
        validate_response(request, PacketResponseIn.model_validate(pack("c" * 64)), restricted_packet=restricted)
    # verified restricted hash accepted
    validate_response(request, PacketResponseIn.model_validate(pack(restricted["restricted_packet_hash"])),
                      restricted_packet=restricted)
    # forged restricted packet rejected
    forged = dict(restricted, restricted_packet_hash="d" * 64)
    with pytest.raises(ValidationFailed, match="restricted packet hash"):
        validate_response(request, PacketResponseIn.model_validate(pack("d" * 64)), restricted_packet=forged)


def test_missing_synthesizer_schema_fails_import(tmp_path):
    import json
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role}.json").write_text(json.dumps(_role_payload(role, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    payload = _role_payload("synthesizer", state)
    del payload["instruments"]["BTCUSDT"]
    (run_dir / "inbox" / "synthesizer.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must cover"):
        cycle.import_roles(run_dir, db_path=run_dir / "db" / "missing.sqlite")


def test_instrument_specific_gates_do_not_block_gold_or_index(tmp_path):
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    assert any("onchain" in r for r in state["trading_recommendation_blockers"])
    assert any("btc_etf_flows" in r for r in state["trading_recommendation_blockers"])
    inst = state["instrument_recommendation_blockers"]
    assert inst["XAUUSDT"] == []
    assert inst["NDX100USDT"] == []
    assert inst["BTCUSDT"]
    request = cycle._load_json(Path(state["paths"]["run_dir"]) / "evidence" / "request.json")
    assert "equities_fundamentals" not in (request.get("missing_datasets") or [])
    assert "gold_etf_and_physical_demand_flows" in request["missing_datasets"]
    assert "nasdaq100_official_aggregate_valuation_and_breadth" in request["missing_datasets"]


def test_import_downgrades_only_blocked_instruments(tmp_path):
    import json
    state = cycle.prepare_run(tmp_path, snapshot=_snapshot(), macro=_macro(), forecast_payload=_forecast())
    run_dir = Path(state["paths"]["run_dir"])
    for role_name in ("researcher", "critic_independent"):
        (run_dir / "inbox" / f"{role_name}.json").write_text(json.dumps(_role_payload(role_name, state)), encoding="utf-8")
    cycle.materialize_followups(run_dir)
    payload = _role_payload("synthesizer", state)
    for symbol in cycle.SYMBOLS:
        payload["instruments"][symbol]["suggested_action"] = "review_paper_plan"
        payload["instruments"][symbol]["direction"] = "long"
        payload["instruments"][symbol]["proposed_stop"] = 90.0
    (run_dir / "inbox" / "synthesizer.json").write_text(json.dumps(payload), encoding="utf-8")
    imported = cycle.import_roles(run_dir, db_path=run_dir / "db" / "downgrade.sqlite")
    result = cycle._load_json(run_dir / "import" / "result.json")
    downgraded = {item["symbol"] for item in result["recommendation_downgrades"]}
    assert downgraded == {"BTCUSDT"}
    assert imported["phase"] == "imported_unapproved"
    assert result["approved"] is False
    assert result["executed"] is False
    assert imported.get("approved") in {None, False}

