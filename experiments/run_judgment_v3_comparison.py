"""Run PIIA Judgment v3 four-way comparison & 5-domain leave-one-out ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from piia.intelligence.judgment.evaluation import (
    ROOT,
    run_domain_ablation,
    run_four_way_comparison,
    run_predetermined_benchmark,
)
from piia.intelligence.judgment.evidence import (
    collect_judgment_inputs,
    generate_demo_judgment_inputs,
)
from piia.intelligence.judgment.manifest import verify_judgment_v3_frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PIIA Judgment v3 AI-comparison & 5-domain ablation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action="store_true", help="Collect current live public data once")
    group.add_argument("--demo-synthetic", action="store_true", help="Run on explicitly labelled offline synthetic demo data")
    group.add_argument("--replay-eval", action="store_true", help="Run predetermined UTC schedule benchmark on local historical eval dataset")
    parser.add_argument("--eval-dir", default=None, help="Directory with {SYM}_1h.csv, {SYM}_funding.json, alfred_macro.json")
    parser.add_argument("--starting-cash", type=float, default=100.0, help="Identical starting cash in USDT across all arms")
    parser.add_argument("--call-live-model", action="store_true", help="Invoke configured official AI model for A1/A2 comparison")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "judgment_v3_comparison.json"))
    args = parser.parse_args(argv)

    fz = verify_judgment_v3_frozen()
    if args.replay_eval:
        bench = run_predetermined_benchmark(
            eval_dir=Path(args.eval_dir) if args.eval_dir else None,
            starting_cash=float(args.starting_cash),
            call_live_model=bool(args.call_live_model),
        )
        report = {
            "frozen_verification": fz,
            "mode": "predetermined_eval_replay",
            "starting_cash_usdt": float(args.starting_cash),
            "benchmark": bench,
            "order_permissions": False,
        }
    else:
        bundle = generate_demo_judgment_inputs() if args.demo_synthetic else collect_judgment_inputs()
        four_way = run_four_way_comparison(
            bundle,
            call_live_model=bool(args.call_live_model),
            starting_cash_usdt=float(args.starting_cash),
        )
        ablation = run_domain_ablation(bundle, starting_cash=float(args.starting_cash))
        report = {
            "frozen_verification": fz,
            "is_synthetic_demo": bool(bundle.get("is_synthetic_demo")),
            "starting_cash_usdt": float(args.starting_cash),
            "four_way_comparison": four_way,
            "domain_ablation": ablation,
            "order_permissions": False,
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
