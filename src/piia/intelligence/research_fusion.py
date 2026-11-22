"""Assemble independently tagged research evidence without voting or trading.

Technical regime, macro snapshot, statistical forecast, and AI adapter status
remain separate. Contradictions are retained. This is not an autonomous agent
and does not approve or execute orders.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from piia.intelligence.forecast_benchmark import neural_forecast_status, random_walk_drift_forecast
from piia.intelligence.macro_regime import classify_macro_snapshot
from piia.intelligence.settings import model_configuration_reasons, model_status


def _forecast_direction(forecast: dict[str, Any]) -> str | None:
    if not forecast or not forecast.get("available"):
        return None
    last, point = forecast.get("last_close"), forecast.get("point_horizon")
    try:
        last_f, point_f = float(last), float(point)
    except (TypeError, ValueError):
        return None
    if not last_f:
        return None
    if point_f > last_f:
        return "up"
    if point_f < last_f:
        return "down"
    return "unchanged"


def build_research_bundle(*, as_of_utc: str, symbol: str | None,
                          technical_regime: dict[str, Any] | None,
                          macro_regime: dict[str, Any] | None,
                          statistical_forecast: dict[str, Any] | None,
                          neural_status: dict[str, Any] | None,
                          ai_adapter: dict[str, Any] | None,
                          open_interest: dict[str, Any] | None = None,
                          liquidity_positioning: dict[str, Any] | None = None) -> dict[str, Any]:
    technical = technical_regime or {
        "kind": "deterministic_closed_bar_classifier_not_agent",
        "regime": "insufficient_data",
        "reason": "bars_or_indicator_library_unavailable",
        "order_permissions": False,
    }
    macro = macro_regime or classify_macro_snapshot(None)
    forecast = statistical_forecast or {"kind": "statistical_benchmark_not_neural_model", "available": False,
                                        "reason": "closes_not_provided", "calibration_status": "not_calibrated"}
    neural = neural_status or neural_forecast_status()
    adapter = ai_adapter or {**model_status(), "request_schema": "not_executed"}
    contradictions: list[str] = []
    tech_label = technical.get("regime")
    curve = macro.get("curve_label")
    forecast_dir = _forecast_direction(forecast)
    if tech_label == "trend" and technical.get("completed_4h_ema20_vs_ema50") == "up" and forecast_dir == "down":
        contradictions.append("technical_uptrend_vs_statistical_forecast_down")
    if tech_label == "trend" and technical.get("completed_4h_ema20_vs_ema50") == "down" and forecast_dir == "up":
        contradictions.append("technical_downtrend_vs_statistical_forecast_up")
    if tech_label == "range" and forecast_dir in {"up", "down"}:
        contradictions.append("range_label_vs_directional_statistical_forecast")
    if curve == "inverted" and technical.get("completed_4h_ema20_vs_ema50") == "up":
        contradictions.append("inverted_yield_curve_vs_upward_completed_4h_trend")
    if macro.get("macro_enhanced_historical_alpha") != "blocked" and not macro.get("historical_replay_supported"):
        contradictions.append("macro_alpha_claim_without_historical_replay")
        macro = {**macro, "macro_enhanced_historical_alpha": "blocked"}
    return {
        "kind": "research_evidence_bundle_not_trade",
        "as_of_utc": as_of_utc,
        "symbol": symbol,
        "order_permissions": False,
        "voting": False,
        "suggested_action": "research_only",
        "independent_sections": {
            "technical_regime": technical,
            "macro_regime": macro,
            "statistical_forecast": forecast,
            "neural_forecast_status": neural,
            "ai_adapter_status": adapter,
            "open_interest": open_interest or {"status": "not_provided"},
            "liquidity_positioning": liquidity_positioning or {"status": "not_provided", "not_a_trading_signal": True},
        },
        "contradictions_retained": contradictions,
        "confidence_is_not_accuracy": True,
        "forecast_calibration_status": forecast.get("calibration_status", "not_calibrated"),
        "limitations": [
            "Sections are independent evidence tags, not a voted trading decision.",
            "LLM confidence, if later added, is not a calibrated forecast accuracy.",
            "Missing historical macro vintages block claims of macro-enhanced historical alpha.",
            "Statistical intervals are uncalibrated; neural models stay unused unless separately configured.",
        ],
    }


def default_bundle_from_closes(closes: list[float], *, symbol: str | None = None,
                               as_of_utc: str | None = None,
                               macro_context: dict[str, Any] | None = None,
                               technical_regime: dict[str, Any] | None = None) -> dict[str, Any]:
    as_of = as_of_utc or datetime.now(timezone.utc).isoformat()
    return build_research_bundle(
        as_of_utc=as_of,
        symbol=symbol,
        technical_regime=technical_regime,
        macro_regime=classify_macro_snapshot(macro_context),
        statistical_forecast=random_walk_drift_forecast(closes),
        neural_status=neural_forecast_status(),
        ai_adapter={**model_status(), "configuration_reasons": model_configuration_reasons(model_status()),
                    "request_schema": "built_only_when_user_configures_credentials"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a research evidence bundle without placing orders")
    parser.add_argument("--closes-json", help="JSON list of closed prices; statistical benchmark only")
    parser.add_argument("--symbol")
    parser.add_argument("--macro-json", help="optional saved ALFRED context JSON; never invents missing values")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    closes = json.loads(open(args.closes_json, encoding="utf-8").read()) if args.closes_json else []
    macro = json.loads(open(args.macro_json, encoding="utf-8").read()) if args.macro_json else None
    bundle = default_bundle_from_closes(closes, symbol=args.symbol, macro_context=macro)
    text = json.dumps(bundle, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
