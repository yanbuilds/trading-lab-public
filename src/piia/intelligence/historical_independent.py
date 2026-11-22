"""Ingest isolated researcher/critic/synthesizer runs and replay them historically.

These files were produced in externally orchestrated, distinct assistant contexts.
That is not program-enforced inference isolation, not a configured official model
API, and not statistically independent validation. Same provider / correlated model
risk remains. The official PIIA automatic model adapter stays unconfigured.

TradingAgents-style role separation (research, debate, decision) is used as a
workflow pattern only; that library is not a project dependency.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from piia.intelligence.hashing import content_hash
from piia.intelligence.historical_packets import PROTOCOL
from piia.intelligence.historical_phase2 import (
    DB_PATH as PRIOR_PHASE2_DB,
    FORBIDDEN_DB,
    PACKET_DIR,
    FrozenDecisionError,
    run_phase2,
    sha256_file,
    write_report,
)
from piia.intelligence.paper.engine import spec_for
from piia.intelligence.paths import ROOT
from trading_lab.perps.risk import floor_step, size_order

REVIEW_MARKER = "ISOLATED_CONTEXT_RUNS_SAME_PROVIDER_NOT_STATISTICALLY_INDEPENDENT"
AUTHOR = "invest-desk_isolated_role_runs"
LABEL = "HISTORICAL_INDEPENDENT_ROLES_V1"
EXPECTED_PACKET_BYTES_SHA256 = "b3cc862d791eba2ac2e0f0690067db460c81d877a0e550f67eefca54ab1b7a75"
EXPECTED_EMBEDDED_PACKET_HASH = "facde07f273828814050d910455fa20e218f0286404a34b2dc9f31ee57fed4ed"
ROLE_FILES = {
    "researcher": {
        "sha256": "cc2c99c692854093e6a205b1cb1a425ecf2eb0a482b775e29841c90c2cec5ae1",
        "default": Path("/workspace/piia-independent-researcher.json"),
    },
    "critic": {
        "sha256": "0c55aea31fb18ee379d5cb7a505697db31a4ced8f8103111ed5b652fd667826e",
        "default": Path("/workspace/piia-independent-critic.json"),
    },
    "synthesizer": {
        "sha256": "fd9ec2c9d6daef2a2d498edfe725e79e3f41b919d7789605794327f4e2bb7f99",
        "default": Path("/workspace/piia-independent-synthesis.json"),
    },
}
OUT_DIR = ROOT / "artifacts" / "research_packets" / "historical_v1_independent"
RAW_DIR = OUT_DIR / "raw"
DB_PATH = ROOT / "data" / "piia" / "piia.historical_independent_v1.sqlite"
EVIDENCE_STORE = ROOT / "data" / "piia" / "historical_independent_v1_evidence"
PRIOR_DECISIONS = PACKET_DIR / "CHAT_DECISIONS_FROZEN.json"
PRIOR_RESULTS = PACKET_DIR / "PHASE2_RESULTS.json"

FUNDING_RESERVE_USDT = 0.06  # explicit writeup reserve; not a future-funding forecast
PLANNED_LOSS_USDT = 0.30
ROUND_TRIP_COST = 0.0022
MAX_NOTIONAL = 30.0
SLIPPAGE = 0.0005
LEVERAGE = 2.0
PACKET_IDS = ["P-05", "P-04", "P-03", "P-09", "P-02", "P-10", "P-07", "P-01", "P-06", "P-08"]


class IndependentRoleError(FrozenDecisionError):
    pass


def _load_role(path: Path, expected_role: str, expected_hash: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise IndependentRoleError(f"missing {expected_role} file: {path}")
    digest = sha256_file(path)
    if digest != expected_hash:
        raise IndependentRoleError(f"{expected_role} hash mismatch: {digest}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise IndependentRoleError(f"{expected_role} is not an object")
    role = payload.get("role") or payload.get("actor")
    # Synthesizer file uses protocol + actor, not a top-level role field.
    if expected_role != "synthesizer":
        if payload.get("role") != expected_role:
            raise IndependentRoleError(
                f"role field must be {expected_role!r}; renaming another file cannot fake independence"
            )
    else:
        if payload.get("protocol") != PROTOCOL:
            raise IndependentRoleError("synthesizer protocol mismatch")
        if payload.get("independent_llm_review") != REVIEW_MARKER:
            raise IndependentRoleError("synthesizer marker must describe isolated same-provider runs")
        if payload.get("independent_llm_review") == "NOT_WIRED":
            raise IndependentRoleError("these isolated historical role runs are not NOT_WIRED")
    rows = payload.get("decisions")
    if not isinstance(rows, list) or not rows:
        raise IndependentRoleError(f"{expected_role} has no decisions")
    return payload, digest


def _packet_map(payload: dict[str, Any], expected_role: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in payload["decisions"]:
        if expected_role == "synthesizer":
            if not isinstance(row.get("researcher"), dict) or not isinstance(row.get("critic"), dict) or not isinstance(row.get("synthesizer"), dict):
                raise IndependentRoleError(f"{row.get('packet_id')} synthesizer row missing role sections")
            body = row
            pid = row.get("packet_id")
        else:
            body = row
            pid = row.get("packet_id")
        if pid not in PACKET_IDS:
            raise IndependentRoleError(f"unexpected packet_id {pid} in {expected_role}")
        if pid in out:
            raise IndependentRoleError(f"duplicate {pid} in {expected_role}")
        out[pid] = body
    missing = [p for p in PACKET_IDS if p not in out]
    if missing:
        raise IndependentRoleError(f"{expected_role} incomplete, fail-closed: missing {missing}")
    return out


def _require_fields(pid: str, role: str, body: dict[str, Any]) -> None:
    if body.get("direction") not in {"long", "short", "flat"}:
        raise IndependentRoleError(f"{pid} {role} missing direction")
    if not str(body.get("summary") or "").strip() or not str(body.get("invalidation") or "").strip():
        raise IndependentRoleError(f"{pid} {role} missing rationale")
    if body["direction"] in {"long", "short"} and not body.get("asset"):
        raise IndependentRoleError(f"{pid} {role} directional without asset")


def load_independent_roles(
    *,
    researcher_path: Path | None = None,
    critic_path: Path | None = None,
    synthesizer_path: Path | None = None,
    packet_path: Path | None = None,
) -> dict[str, Any]:
    packet_path = packet_path or (PACKET_DIR / "PACKETS_FOR_REVIEW.json")
    packet_bytes_hash = sha256_file(packet_path)
    if packet_bytes_hash != EXPECTED_PACKET_BYTES_SHA256:
        raise IndependentRoleError(f"source packet bytes hash mismatch: {packet_bytes_hash}")
    parent = json.loads(packet_path.read_text())
    if parent.get("packet_file_hash") != EXPECTED_EMBEDDED_PACKET_HASH:
        raise IndependentRoleError("embedded packet_file_hash mismatch")

    paths = {
        "researcher": researcher_path or ROLE_FILES["researcher"]["default"],
        "critic": critic_path or ROLE_FILES["critic"]["default"],
        "synthesizer": synthesizer_path or ROLE_FILES["synthesizer"]["default"],
    }
    raw: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for role, meta in ROLE_FILES.items():
        payload, digest = _load_role(paths[role], role, meta["sha256"])
        raw[role] = payload
        hashes[role] = digest
        # All three must cite the same source packet bytes.
        cited = (
            payload.get("source_packet_hash")
            or ((payload.get("provenance") or {}).get("sources") or [{}])[0].get("sha256")
        )
        if cited and cited != EXPECTED_PACKET_BYTES_SHA256:
            raise IndependentRoleError(f"{role} source_packet_hash does not match frozen packets")

    r_map = _packet_map(raw["researcher"], "researcher")
    c_map = _packet_map(raw["critic"], "critic")
    s_map = _packet_map(raw["synthesizer"], "synthesizer")
    combined = []
    for pid in PACKET_IDS:
        r, c, srow = r_map[pid], c_map[pid], s_map[pid]
        _require_fields(pid, "researcher", r)
        _require_fields(pid, "critic", c)
        syn = srow["synthesizer"]
        _require_fields(pid, "synthesizer", syn)
        if syn.get("accept_default_2atr_stop_and_72h_horizon") is not True and syn["direction"] in {"long", "short"}:
            raise IndependentRoleError(f"{pid} directional synthesizer must accept frozen stop/horizon")
        if syn["direction"] in {"long", "short"} and syn.get("suggested_action") != "review_paper_plan":
            raise IndependentRoleError(f"{pid} directional action must remain review_paper_plan")
        # Keep exact original researcher/critic records in provenance; use synthesizer copies
        # for the role-body fields it retained, and attach original extra fields separately.
        combined.append({
            "packet_id": pid,
            "researcher": {
                "direction": r["direction"], "asset": r.get("asset"),
                "summary": r["summary"], "invalidation": r["invalidation"],
                "missing_data": r.get("missing_data") or [],
                "claims": r.get("claims") or [],
                "disagreements": r.get("disagreements") or [],
                "execution_status": r.get("execution_status"),
                "counter_risk": r.get("counter_risk"),
            },
            "critic": {
                "direction": c["direction"], "asset": c.get("asset"),
                "summary": c["summary"], "invalidation": c["invalidation"],
                "missing_data": c.get("missing_data") or [],
                "claims": c.get("claims") or [],
                "disagreements": c.get("disagreements") or [],
                "execution_gate": c.get("execution_gate"),
            },
            "synthesizer": syn,
            "original_researcher_extra": {k: r[k] for k in r if k not in {
                "packet_id", "direction", "asset", "summary", "invalidation",
                "missing_data", "claims", "disagreements",
            }},
            "original_critic_extra": {k: c[k] for k in c if k not in {
                "packet_id", "direction", "asset", "summary", "invalidation",
                "missing_data", "claims", "disagreements",
            }},
            "role_disagreements": {
                "researcher": r.get("disagreements") or [],
                "critic": c.get("disagreements") or [],
                "synthesizer": syn.get("disagreements") or [],
                "direction_conflict": {
                    "researcher": r["direction"], "researcher_asset": r.get("asset"),
                    "critic": c["direction"], "critic_asset": c.get("asset"),
                    "synthesizer": syn["direction"], "synthesizer_asset": syn.get("asset"),
                },
            },
        })

    decisions = {
        "protocol": PROTOCOL,
        "actor": AUTHOR,
        "generation": "manual_orchestrated_independent_runs",
        "independent_llm_review": REVIEW_MARKER,
        "official_model_api": "not_configured",
        "isolation": (
            "Externally orchestrated distinct assistant contexts. "
            "Not program-enforced inference isolation, not a statistically independent model, "
            "and not a configured official PIIA model API."
        ),
        "decisions": combined,
    }
    return {
        "decisions": decisions,
        "parent": parent,
        "hashes": hashes,
        "paths": {k: str(v) for k, v in paths.items()},
        "packet_bytes_sha256": packet_bytes_hash,
        "combined_digest": content_hash(json.dumps({
            "hashes": hashes, "packet": packet_bytes_hash, "marker": REVIEW_MARKER,
        }, sort_keys=True).encode()),
    }


def archive_raw_roles(loaded: dict[str, Any], dest: Path = RAW_DIR) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = {}
    for role, meta in ROLE_FILES.items():
        src = Path(loaded["paths"][role])
        out = dest / f"{role}.json"
        shutil.copy2(src, out)
        digest = sha256_file(out)
        if digest != meta["sha256"]:
            raise IndependentRoleError(f"archive hash mismatch for {role}")
        (dest / f"{role}.sha256").write_text(digest + "\n")
        written[role] = str(out)
    (dest / "PROVENANCE.txt").write_text(
        "\n".join([
            f"marker={REVIEW_MARKER}",
            f"packet_bytes_sha256={loaded['packet_bytes_sha256']}",
            f"researcher={loaded['hashes']['researcher']}",
            f"critic={loaded['hashes']['critic']}",
            f"synthesizer={loaded['hashes']['synthesizer']}",
            "isolation=externally_orchestrated_distinct_contexts",
            "program_enforced_inference_isolation=false",
            "official_model_api=not_configured",
            "statistically_independent=false",
        ]) + "\n"
    )
    return written


def funding_inclusive_entry_prepare(
    *, raw_open: float, stop_distance: float, side: str, symbol: str, packet_id: str,
    decision: dict[str, Any], event: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit additional conditions; do not convert them into an unconditioned entry.

    Uses the writeups' 0.30 USDT planned-loss cap and 0.06 USDT adverse-funding reserve.
    The reserve is the specified allowance, not a forecast of subsequent funding.
    Quantity is rounded down only. Failed validation stays flat.
    """
    syn = decision["synthesizer"]
    notes = {
        "planned_loss_usdt": PLANNED_LOSS_USDT,
        "funding_reserve_usdt": FUNDING_RESERVE_USDT,
        "round_trip_cost_fraction": ROUND_TRIP_COST,
        "reserve_is_not_a_future_funding_forecast": True,
    }
    if syn.get("suggested_action") != "review_paper_plan" or syn.get("direction") not in {"long", "short"}:
        return {"proceed": False, "reason": "not_a_review_paper_plan", "max_qty": None, "notes": notes}
    # Researcher recorded operational flat unless entry-time gates are verified.
    spec = spec_for(symbol)
    fill = raw_open * (1 + SLIPPAGE) if side == "long" else raw_open * (1 - SLIPPAGE)
    known_frac = stop_distance / raw_open + ROUND_TRIP_COST
    notes["known_cost_fraction"] = known_frac
    if known_frac <= 0:
        return {"proceed": False, "reason": "invalid_cost_fraction", "max_qty": None, "notes": notes}
    remaining = PLANNED_LOSS_USDT - FUNDING_RESERVE_USDT
    if remaining <= 0:
        return {"proceed": False, "reason": "funding_inclusive_risk_unverified", "max_qty": None, "notes": notes}
    max_notional = min(MAX_NOTIONAL, remaining / known_frac)
    qty, reason = size_order(
        30.0, 30.0, raw_open, stop_distance, spec,
        leverage=LEVERAGE, risk_fraction=0.01, cash_floor=0.20,
        max_margin_fraction=0.50, slippage=SLIPPAGE,
    )
    notes["unconstrained_qty"] = qty
    notes["unconstrained_reason"] = reason
    if qty <= 0:
        return {"proceed": False, "reason": reason, "max_qty": None, "notes": notes}
    cap = floor_step(max_notional / (fill * spec.contract_size), spec.qty_step)
    qty = floor_step(min(qty, cap), spec.qty_step)
    notes["capped_qty"] = qty
    notes["max_notional"] = max_notional
    if qty < spec.min_qty or qty * fill * spec.contract_size < spec.min_notional:
        return {
            "proceed": False,
            "reason": "minimum_order_exceeds_funding_inclusive_budget",
            "max_qty": None,
            "notes": notes,
        }
    known_cost = qty * fill * spec.contract_size * known_frac
    notes["known_cost_usdt"] = known_cost
    notes["known_cost_plus_reserve"] = known_cost + FUNDING_RESERVE_USDT
    if known_cost + FUNDING_RESERVE_USDT > PLANNED_LOSS_USDT + 1e-9:
        return {"proceed": False, "reason": "funding_inclusive_risk_cap", "max_qty": None, "notes": notes}
    return {"proceed": True, "reason": "ok", "max_qty": qty, "notes": notes}


def compare_prior_run(current: dict[str, Any]) -> dict[str, Any]:
    if not PRIOR_RESULTS.exists():
        return {"available": False}
    prior = json.loads(PRIOR_RESULTS.read_text())
    return {
        "available": True,
        "warning": (
            "Same already-seen sample, same costs, same model-provider family. "
            "The isolated-context process is not independent statistical validation. "
            "Do not treat a better or worse number as proof that independence works."
        ),
        "prior": {
            "label": prior.get("label"),
            "mode": prior.get("mode"),
            "independent_llm_review": prior.get("independent_llm_review"),
            "starting_cash": prior.get("starting_cash"),
            "final_equity": prior.get("final_equity"),
            "return_pct": prior.get("return_pct"),
            "max_drawdown_pct": prior.get("max_drawdown_pct"),
            "closed_trades": prior.get("closed_trades"),
            "wins": prior.get("wins"),
            "losses": prior.get("losses"),
            "fees": prior.get("fees"),
            "funding_net_paid": prior.get("funding_net_paid"),
            "rejections": prior.get("rejections"),
        },
        "current": {
            "final_equity": current.get("final_equity"),
            "return_pct": current.get("return_pct"),
            "max_drawdown_pct": current.get("max_drawdown_pct"),
            "closed_trades": current.get("closed_trades"),
            "wins": current.get("wins"),
            "losses": current.get("losses"),
            "fees": current.get("fees"),
            "funding_net_paid": current.get("funding_net_paid"),
            "rejections": current.get("rejections"),
        },
        "window_match": (
            prior.get("window_start_utc") == current.get("window_start_utc")
            and prior.get("window_end_utc") == current.get("window_end_utc")
        ),
    }


def run_independent(*, packet_dir: Path = PACKET_DIR, data_dir: Path | None = None,
                    db_path: Path = DB_PATH) -> dict[str, Any]:
    from piia.intelligence.historical_phase2 import DATA_DIR
    if db_path.resolve() in {FORBIDDEN_DB.resolve(), PRIOR_PHASE2_DB.resolve()}:
        raise ValueError("refusing a protected research database")
    if db_path.exists():
        raise FileExistsError("replay database must be a new path")
    loaded = load_independent_roles(packet_path=packet_dir / "PACKETS_FOR_REVIEW.json")
    archive_raw_roles(loaded)
    if sha256_file(PRIOR_DECISIONS) != "d881a688aba853b7c8cd05f2028ba0bfe4a1ffcedfa0bb8abc42fe61624888ae":
        raise IndependentRoleError("refusing to continue; prior frozen single-assistant decisions changed")
    sealed = json.loads((packet_dir / "BACKEND_SEALED.json").read_text())
    bundle = (loaded["decisions"], loaded["parent"], sealed, loaded["combined_digest"])
    # Confirm role files still match after archival, before outcomes exist in this process's report.
    integrity = all(sha256_file(Path(loaded["paths"][r])) == ROLE_FILES[r]["sha256"] for r in ROLE_FILES)
    extra = {
        "role_input_hashes": loaded["hashes"],
        "source_packet_bytes_sha256": loaded["packet_bytes_sha256"],
        "official_model_api": "not_configured",
        "isolation_boundary": {
            "externally_orchestrated_distinct_contexts": True,
            "program_enforced_inference_isolation": False,
            "statistically_independent": False,
            "same_provider": True,
            "same_already_seen_sample": True,
            "held_out": False,
            "tradingagents_library_integrated": False,
            "marker": REVIEW_MARKER,
        },
        "prior_single_assistant_decisions_preserved": True,
        "funding_reserve_usdt": FUNDING_RESERVE_USDT,
    }
    report = run_phase2(
        packet_dir=packet_dir,
        data_dir=data_dir or DATA_DIR,
        db_path=db_path,
        evidence_store=EVIDENCE_STORE,
        starting_cash=30.0,
        decisions_bundle=bundle,
        label=LABEL,
        author=AUTHOR,
        review_marker=REVIEW_MARKER,
        mode="isolated_context_role_runs_same_provider",
        label_quality="retrospective_exploratory_isolated_contexts_not_statistically_independent",
        entry_prepare=funding_inclusive_entry_prepare,
        report_extra=extra,
        integrity_ok=integrity,
        protected_dbs=(FORBIDDEN_DB, PRIOR_PHASE2_DB),
    )
    report["prior_single_assistant_comparison"] = compare_prior_run(report)
    # Re-check inputs were not rewritten after the run.
    report["role_files_unmodified_after_run"] = all(
        sha256_file(Path(loaded["paths"][r])) == ROLE_FILES[r]["sha256"] for r in ROLE_FILES
    )
    report["prior_decisions_unmodified_after_run"] = (
        sha256_file(PRIOR_DECISIONS) == "d881a688aba853b7c8cd05f2028ba0bfe4a1ffcedfa0bb8abc42fe61624888ae"
    )
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Replay frozen role research into a new local database")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="new replay database; existing files are never replaced")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="generated report directory")
    args = parser.parse_args()
    report = run_independent(db_path=args.db)
    paths = write_report(report, args.out_dir)
    if not report.get("role_files_unmodified_after_run") or not report.get("prior_decisions_unmodified_after_run"):
        raise SystemExit("input decision files changed during the run")
    print("wrote", paths["txt"])
    print("wrote", paths["json"])
    print("return_pct", round(report["return_pct"], 4), "dd", round(report["max_drawdown_pct"], 4))
    print("wins/closed", report["wins"], report["closed_trades"], "rejections", report["rejections"])
    print("recon_ok", report["reconciliation_ok"], "ledger_ok", report["ledger_ok"])
    print("marker", report["independent_llm_review"])
    print("role_files_unmodified", report["role_files_unmodified_after_run"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
