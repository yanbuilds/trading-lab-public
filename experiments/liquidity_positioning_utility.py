"""Pre-registered diagnostic skeleton for liquidity-state utility.

This is not an optimizer and not a claim of historical validation. Metrics are
false positives, missed risk, and lead time versus a leverage-only baseline.
They are not backtest alpha. Event labels must be fixed before measuring.
Records without a known available_at at or before the decision cutoff are dropped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from piia.intelligence.liquidity_positioning import assess_liquidity_positioning, load_config, usable_for_replay

HYPOTHESIS = load_config()["preregistered_validation"]["hypothesis"]
HORIZON = timedelta(days=7)
# Synthetic fixture labels only. These events are not real market history.
SYNTHETIC_EVENT_LABELS = (
    {
        "label_id": "SYNTHETIC_FIXTURE_NOT_REAL_MARKET:drawdown_event",
        "decision_cutoff_utc": "2026-01-10T00:00:00+00:00",
        "event_start_utc": "2026-01-12T00:00:00+00:00",
        "large_drawdown_within_7d": True,
    },
    {
        "label_id": "SYNTHETIC_FIXTURE_NOT_REAL_MARKET:non_event",
        "decision_cutoff_utc": "2026-02-10T00:00:00+00:00",
        "event_start_utc": None,
        "large_drawdown_within_7d": False,
    },
)


def leakage_safe(decision_cutoff: datetime, outcome_start: datetime | None) -> bool:
    if outcome_start is None:
        return True
    return outcome_start > decision_cutoff


def leverage_only_alert(features: dict[str, Any]) -> bool:
    oi = (features.get("open_interest") or {}).get("BTCUSDT") or {}
    return bool(oi.get("status") == "ok" and (oi.get("base_change_24h_pct") or 0) > 0)


def evaluate_labeled_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score pre-labeled fixtures. Refuses to treat results as validated history."""
    alerts = {"state": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "lead_hours": []},
              "leverage_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "lead_hours": []}}
    rejected = []
    for case in cases:
        if not case.get("label_id", "").startswith("SYNTHETIC_FIXTURE_NOT_REAL_MARKET"):
            rejected.append({"reason": "non_synthetic_label_refused_until_protocol_freeze", "label_id": case.get("label_id")})
            continue
        cutoff = datetime.fromisoformat(case["decision_cutoff_utc"])
        outcome = datetime.fromisoformat(case["event_start_utc"]) if case.get("event_start_utc") else None
        if not leakage_safe(cutoff, outcome):
            rejected.append({"reason": "outcome_not_after_cutoff", "label_id": case["label_id"]})
            continue
        snapshot = case.get("snapshot")
        if not snapshot:
            rejected.append({"reason": "missing_snapshot", "label_id": case["label_id"]})
            continue
        assessment = assess_liquidity_positioning(snapshot, cutoff=cutoff, mode="historical_replay",
                                                  prior_state=case.get("prior_state"))
        event = bool(case.get("large_drawdown_within_7d"))
        state_alert = assessment["official_state"] in {"fragility_building", "deleveraging"}
        base_alert = leverage_only_alert(assessment["features"])
        for name, alert in (("state", state_alert), ("leverage_only", base_alert)):
            bucket = alerts[name]
            if alert and event:
                bucket["tp"] += 1
                if outcome:
                    bucket["lead_hours"].append((outcome - cutoff).total_seconds() / 3600)
            elif alert and not event:
                bucket["fp"] += 1
            elif (not alert) and event:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1
    def _rate(bucket: dict[str, Any]) -> dict[str, Any]:
        pos = bucket["tp"] + bucket["fp"]
        ev = bucket["tp"] + bucket["fn"]
        return {
            "false_positive_rate": (bucket["fp"] / pos) if pos else None,
            "missed_risk_rate": (bucket["fn"] / ev) if ev else None,
            "mean_lead_time_hours": (sum(bucket["lead_hours"]) / len(bucket["lead_hours"])) if bucket["lead_hours"] else None,
            "counts": {k: bucket[k] for k in ("tp", "fp", "fn", "tn")},
        }
    return {
        "kind": "liquidity_positioning_preregistered_diagnostic",
        "hypothesis": HYPOTHESIS,
        "horizon": "7_utc_days",
        "not_backtest_alpha": True,
        "historical_states_validated": False,
        "historical_data_insufficient": True,
        "synthetic_fixtures_only": True,
        "usable_for_replay_requires_available_at": True,
        "rejected": rejected,
        "metrics": {"state_joint_conditions": _rate(alerts["state"]),
                    "leverage_only_baseline": _rate(alerts["leverage_only"])},
        "note": "Fixture results do not validate live states. Collect a PIT archive before measuring real events.",
    }


def usable_record_count(records: list[dict[str, Any]], cutoff: datetime) -> int:
    return sum(1 for item in records if usable_for_replay(item, cutoff))
