"""Record an on-demand prospective model shadow opinion; never execute orders."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from piia.intelligence.paths import ROOT

LEDGER = ROOT / "data/piia/frozen_policy_v1.sqlite"
APP_ID = 0x46505631
START = datetime(2026, 11, 18, tzinfo=timezone.utc)
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
API = "https://api.bitget.com/api/v3/market/candles"
PROTOCOL_SHA256 = "2e6e098f1adde13b9c6151c566c66086cb46460f4aff0800e5e3fd12bebb4c3b"
QUESTION = ("针对 focus_symbol，依据提供的已收盘行情和机械候选，独立陈述支持与反对理由；"
            "综合结论必须为 long、short 或 flat，并选择 review_paper_plan、stand_aside 或 need_more_evidence。"
            "不要假定真实持仓，不给出价格目标、胜率或订单。")


def now_utc():
    return datetime.now(timezone.utc)


def _timestamp(text):
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("aware timestamp required")
    return stamp.astimezone(timezone.utc)


def verify_protocol():
    text = (ROOT / "AI_SHADOW_V1.md").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(text).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("AI shadow protocol changed; use a new version")


def latest_observation(path: Path, now: datetime) -> tuple[int, dict, str]:
    if not path.is_file():
        raise FileNotFoundError("no prospective observation ledger")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT id, cutoff, created_at, payload_json FROM observations ORDER BY cutoff DESC LIMIT 1").fetchone()
    if not row:
        raise ValueError("no prospective observation")
    ident, cutoff_text, created_text, raw = row
    cutoff, created = _timestamp(cutoff_text), _timestamp(created_text)
    event = json.loads(raw)
    if (cutoff < START or not cutoff <= created <= now < cutoff + timedelta(minutes=55)
            or event.get("cutoff_utc") != cutoff_text or event.get("created_at_utc") != created_text
            or event.get("policy_commit") != "69fd853" or event.get("source") != API
            or event.get("order_permissions") is not False or set(event.get("symbols") or []) != set(SYMBOLS)):
        raise ValueError("observation is stale or invalid")
    for symbol in SYMBOLS:
        state, evidence = event["states"][symbol], event["evidence"][symbol]
        if (type(state.get("candidate_signal")) is not int or state["candidate_signal"] not in (-1, 0, 1)
                or len(evidence.get("bars") or []) != 300
                or not cutoff <= _timestamp(evidence["received_at_utc"]) <= created):
            raise ValueError("observation evidence incomplete")
    return ident, event, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _store(path: Path, ident: int, event: dict, observation_sha: str, result: dict):
    cutoff = _timestamp(event["cutoff_utc"])
    created = _timestamp(result["created_at_utc"])
    if not _timestamp(event["created_at_utc"]) <= created < cutoff + timedelta(hours=1) or abs((now_utc() - created).total_seconds()) > 120:
        raise ValueError("model result arrived too late")
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
            raise ValueError("unrelated database")
        original = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        if not original or hashlib.sha256(original[0].encode("utf-8")).hexdigest() != observation_sha:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS ai_shadow_events (observation_id INTEGER PRIMARY KEY, "
                   "created_at TEXT NOT NULL, payload_json TEXT NOT NULL, FOREIGN KEY(observation_id) REFERENCES observations(id))")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_ai_shadow_{action.lower()} BEFORE {action} ON ai_shadow_events "
                       "BEGIN SELECT RAISE(ABORT, 'append-only ai shadow'); END")
        db.execute("INSERT INTO ai_shadow_events(observation_id, created_at, payload_json) VALUES (?, ?, ?)",
                   (ident, created.isoformat(), json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)))


def _claim(path: Path, ident: int, observation_sha: str):
    """Claim once before any potentially billable calls; no retry selection."""
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
            raise ValueError("unrelated database")
        original = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        if not original or hashlib.sha256(original[0].encode("utf-8")).hexdigest() != observation_sha:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS ai_shadow_claims (observation_id INTEGER PRIMARY KEY, "
                   "started_at TEXT NOT NULL, observation_sha256 TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_ai_claim_{action.lower()} BEFORE {action} ON ai_shadow_claims "
                       "BEGIN SELECT RAISE(ABORT, 'append-only ai claim'); END")
        db.execute("INSERT INTO ai_shadow_claims(observation_id, started_at, observation_sha256) VALUES (?, ?, ?)",
                   (ident, now_utc().isoformat(), observation_sha))


def run(path: Path = LEDGER) -> dict:
    from piia.intelligence.agents.workflow import run_workflow
    from piia.intelligence.settings import model_configured
    verify_protocol()
    script_path = Path(__file__)
    script_sha = hashlib.sha256(script_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    ident, observation, observation_sha = latest_observation(path, now_utc())
    signals = {s: observation["states"][s]["candidate_signal"] for s in SYMBOLS}
    if any(signals.values()) and not model_configured():
        return {"status": "needs_model_configuration", "order_permissions": False, "model_called": False}
    _claim(path, ident, observation_sha)
    evidence = [{"id": s, "raw": json.dumps({"symbol": s, "state": observation["states"][s],
                 "last_eight_closed_bars": observation["evidence"][s]["bars"][-8:]}, sort_keys=True)} for s in SYMBOLS]
    decisions, outputs = {}, {}
    for symbol, signal in signals.items():
        if not signal:
            decisions[symbol] = {"mechanical_signal": 0, "ai_pass": False, "reason": "no_mechanical_signal"}
            continue
        payload = {"statement": QUESTION, "focus_symbol": symbol, "invalidation": "缺失证据或不满足机械规则时观望",
                   "evidence": evidence, "mechanical_signal": signal}
        response = None
        try:
            if now_utc() >= _timestamp(observation["cutoff_utc"]) + timedelta(minutes=55):
                raise ValueError("insufficient time for a new model request")
            response = run_workflow(mode="model", payload=payload, legal_evidence_ids=list(SYMBOLS)).model_dump()
            synth = response.get("synthesizer") or {}
            if (response.get("mode") != "model" or response.get("status") != "completed"
                    or synth.get("generation") != "model" or synth.get("direction") not in ("long", "short", "flat")
                    or not all(response.get(k) for k in ("model_provider", "model_name", "prompt_hash"))):
                raise ValueError("model role did not complete")
            direction = "long" if signal == 1 else "short"
            passed = synth["direction"] == direction and synth.get("suggested_action") == "review_paper_plan"
            decisions[symbol] = {"mechanical_signal": signal, "ai_pass": passed, "reason": "model_agrees" if passed else "model_did_not_confirm"}
            outputs[symbol] = response
        except Exception as exc:
            decisions[symbol] = {"mechanical_signal": signal, "ai_pass": False, "reason": "model_failed"}
            outputs[symbol] = {"status": "failed", "error_type": type(exc).__name__,
                               "usage_records": (response or {}).get("usage_records", []),
                               "usage_complete": False, "monetary_cost": "unknown"}
    created = now_utc()
    failed = any(item["reason"] == "model_failed" for item in decisions.values())
    hashes = {x.get("prompt_hash") for x in outputs.values() if x.get("status") == "completed"}
    providers = {(x.get("model_provider"), x.get("model_name")) for x in outputs.values() if x.get("status") == "completed"}
    if len(hashes) > 1 or len(providers) > 1:
        failed = True
    result = {"status": "model_failed" if failed else "completed" if outputs else "no_candidate",
              "created_at_utc": created.isoformat(), "observation_id": ident, "observation_sha256": observation_sha,
              "protocol_sha256": PROTOCOL_SHA256, "question_sha256": hashlib.sha256(QUESTION.encode("utf-8")).hexdigest(),
              "shadow_code_sha256": script_sha,
              "decisions": decisions, "model_outputs": outputs, "order_permissions": False,
              "independent_llm_review": "NOT_WIRED", "monetary_cost": "unknown"}
    verify_protocol()
    if hashlib.sha256(script_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != script_sha:
        raise ValueError("AI shadow code changed during research")
    _store(path, ident, observation, observation_sha, result)
    return {"status": result["status"], "observation_id": ident, "model_called": bool(outputs),
            "order_permissions": False, "monetary_cost": "unknown"}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"AI shadow unavailable: {type(exc).__name__}: {exc}") from exc
