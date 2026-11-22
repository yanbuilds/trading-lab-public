"""All post-activation contiguous segments; independent resets, no aggregate returns."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from experiments import chronos2_shadow, chronos2_shadow_report, frozen_policy_capture as capture, prospective_paper_report as report
from piia.intelligence import ai_shadow
from piia.intelligence.shadow_claim_audit import audit_claims

ACTIVATION = datetime(2026, 11, 18, 7, tzinfo=timezone.utc)
SOURCES = ("SEGMENTED_FORWARD_V2.md", "experiments/segmented_forward_report_v2.py", "src/piia/intelligence/shadow_claim_audit.py")


def verify_precommitted():
    for name in SOURCES:
        disk = (capture.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=capture.ROOT,
                                 capture_output=True, check=True, timeout=15).stdout.replace(b"\r\n", b"\n")
        stamp = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", name], cwd=capture.ROOT,
                               capture_output=True, check=True, timeout=15).stdout.strip()
        if disk != tracked or int(stamp) >= int(ACTIVATION.timestamp()):
            raise ValueError("report rule not committed before activation or changed")


def partition(records):
    """Return every post-activation segment and an excluded pre-activation count."""
    eligible, excluded, previous = [], 0, None
    for record in records:
        cutoff = report._timestamp(record["cutoff_utc"])
        if cutoff != cutoff.replace(minute=0, second=0, microsecond=0) or (previous is not None and cutoff <= previous):
            raise ValueError("unordered, duplicate, or unaligned observations")
        previous = cutoff
        if cutoff < ACTIVATION:
            excluded += 1
        else:
            eligible.append((cutoff, record))
    segments = []
    for cutoff, record in eligible:
        if not segments or cutoff - segments[-1][-1][0] != timedelta(hours=1):
            segments.append([])
        segments[-1].append((cutoff, record))
    return segments, excluded


def build_report(records, ledger: Path) -> dict:
    segments, excluded = partition(records)
    try:
        tiny = report.read_chronos_events(ledger)
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        tiny = {}
    try:
        ai = report.read_shadow_events(ledger)
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        ai = {}
    try:
        chronos2 = chronos2_shadow_report.read_events(ledger)
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        chronos2 = {}
    output = []
    for part in segments:
        rows = [row for _, row in part]
        info = {"first_cutoff_utc": part[0][0].isoformat(), "last_cutoff_utc": part[-1][0].isoformat(),
                "observations": len(rows), "initial_cash_usdt": 100, "initial_positions": 0}
        if len(rows) < 3:
            info.update(status="insufficient_contiguous_observations", result=None)
        else:
            try:
                specs = report.recorded_specs(rows)
                cutoffs = [row["cutoff_utc"] for row in rows]
                audited_ai = ai if audit_claims(ledger, "ai_shadow", cutoffs=cutoffs)["ok"] else {}
                audited_tiny = tiny if audit_claims(ledger, "chronos_shadow", cutoffs=cutoffs)["ok"] else {}
                value = report.evaluate(rows, specs, shadow=audited_ai, chronos_events=audited_tiny)
                try:
                    model = chronos2_shadow_report.evaluate(rows, specs, chronos2, tiny_events=audited_tiny)
                    value["chronos2_shadow"] = {"status": "complete", "result": model["chronos2_shadow"]}
                except (ValueError, TypeError, KeyError, OSError):
                    value["chronos2_shadow"] = {"status": "unavailable_or_incomplete", "result": None}
                info.update(status="price_only_local_diagnostic", result=value)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                info.update(status="segment_invalid_or_unavailable", reason=type(exc).__name__, result=None)
        output.append(info)
    gaps = sum(int((b[0][0] - a[-1][0]).total_seconds() // 3600 - 1) for a, b in zip(segments, segments[1:]))
    return {"status": "all_segments_no_aggregate", "activation_utc": ACTIVATION.isoformat(),
            "excluded_pre_activation_observations": excluded, "missing_hours_between_segments": gaps,
            "missing_hours_before_first": int((segments[0][0][0] - ACTIVATION).total_seconds() // 3600) if segments else None,
            "segments": output, "aggregate_return": None, "order_permissions": False,
            "limitations": ["Every segment resets cash and starts flat; this is not a continuous account.",
                            "Local timestamps and Git metadata are not independent proof.",
                            "Funding unknown; fills, liquidation and terminal closing costs unverified.",
                            "Short segments and sparse trading cannot establish an edge."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=capture.ROOT / "artifacts/segmented_forward_report_v2.json")
    args = parser.parse_args()
    if datetime.now(timezone.utc) < ACTIVATION:
        raise ValueError("report policy is not active")
    verify_precommitted()
    capture.verify_policy()
    report.verify_baseline()
    ai_shadow.verify_protocol()
    chronos2_shadow.source_pins_before(ACTIVATION)
    if not args.ledger.is_file():
        raise FileNotFoundError("prospective ledger missing")
    with tempfile.TemporaryDirectory(prefix="segmented-forward-") as temp:
        snapshot = Path(temp) / "ledger.sqlite"
        with closing(sqlite3.connect(f"{args.ledger.resolve().as_uri()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        try:
            records = report.read_observations(snapshot)
            now = datetime.now(timezone.utc)
            if any(report._timestamp(row["cutoff_utc"]) > now or report._timestamp(row["created_at_utc"]) > now for row in records):
                raise ValueError("future observation in ledger")
            result = build_report(records, snapshot)
        finally:
            gc.collect()  # release unreferenced SQLite handles before Windows removes the temporary copy
    verify_precommitted()
    capture.verify_policy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
