from __future__ import annotations
import argparse, os, sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m piia", description="PIIA Phase-1 intelligence service")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="run Alembic upgrade to head (SQLite default)")
    sub.add_parser("downgrade", help="run Alembic downgrade base (destroys local schema)")
    p_serve = sub.add_parser("serve", help="run localhost API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("PIIA_API_PORT", "8792")))
    p_ui = sub.add_parser("ui", help="run localhost Streamlit UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=int(os.environ.get("PIIA_UI_PORT", "8793")))
    p_replay = sub.add_parser("replay-btc", help="HISTORICAL_REPLAY of locked Bitget BTCUSDT 1h through the local paper engine")
    p_replay.add_argument("--data-dir", default=None, help="directory with BTCUSDT.csv and optional BTCUSDT-funding.csv")
    p_replay.add_argument("--db", default=None, help="isolated replay sqlite path; never the live research db")
    p_replay.add_argument("--out", default=None, help="JSON report path")
    p_replay.add_argument("--starting-cash", type=float, default=None, help="virtual USDT (defaults to owner risk configuration)")
    p_exp = sub.add_parser("research-export", help="export a frozen public-data research request packet")
    p_exp.add_argument("--out", default=None)
    p_exp.add_argument("--include-macro", action="store_true", help="include conservative-date ALFRED macro context")
    p_imp = sub.add_parser("research-import", help="import a chat-assisted research response into an isolated database")
    p_imp.add_argument("--request", required=True)
    p_imp.add_argument("--response", required=True)
    p_imp.add_argument("--db", default=None)
    p_imp.add_argument("--out", default=None)
    p_cycle = sub.add_parser("research-cycle", help="operational research cycle: evidence, isolated roles, unapproved paper plans")
    cycle = p_cycle.add_subparsers(dest="cycle_cmd", required=True)
    p_prep = cycle.add_parser("prepare", help="freeze public evidence, run Chronos-t5-tiny if available, write isolated role requests")
    p_prep.add_argument("--out", default=None, help="parent directory for a new run")
    p_prep.add_argument("--allow-forecast-download", action="store_true", help="allow a bounded official Chronos-t5-tiny snapshot download if the local cache is empty")
    p_follow = cycle.add_parser("followups", help="after independent roles are filed, write labeled debate and synthesizer requests")
    p_follow.add_argument("--run", required=True, help="run directory created by prepare")
    p_imp2 = cycle.add_parser("import-roles", help="import isolated role responses and create UNAPPROVED paper plans only")
    p_imp2.add_argument("--run", required=True)
    p_imp2.add_argument("--db", default=None, help="isolated sqlite path; never the live research db")
    p_stat = cycle.add_parser("status", help="print the run state and any execution revalidations")
    p_stat.add_argument("--run", required=True)
    p_rev = cycle.add_parser("revalidate", help="fetch a fresh public book/ticker and re-check entry risk without changing frozen research")
    p_rev.add_argument("--run", required=True, help="existing research-cycle run directory")
    p_tiers = cycle.add_parser("tiers", help="fetch official public Bitget position-tier snapshots only; no orders")
    p_tiers.add_argument("--out", default=None, help="optional directory to write raw snapshots")
    p_liq = sub.add_parser("liquidity-positioning", help="deterministic liquidity and positioning research assessment; no orders")
    p_liq.add_argument("--out", default=None, help="optional JSON path under artifacts or another local path")
    p_liq.add_argument("--prior-json", default=None, help="optional prior official-state JSON with as_of_utc")

    p_scan = sub.add_parser("scan", help="v2 trading assistant: live/demo multi-contract regime, signal radar & auto trade tickets")
    p_scan.add_argument("--demo", action="store_true", help="use labeled synthetic offline snapshot instead of live network")
    p_scan.add_argument("--equity", type=float, default=None, help="account equity in USDT (defaults to trading_policy_v2.yaml)")
    p_scan.add_argument("--leverage", type=float, default=None, help="isolated leverage (defaults to trading_policy_v2.yaml)")
    p_scan.add_argument("--risk-pct", type=float, default=None, help="per-trade risk budget percentage, e.g. 1.0")
    p_scan.add_argument("--order-type", choices=["limit", "market"], default="limit")
    p_scan.add_argument("--json", action="store_true", help="output full JSON instead of formatted radar table")
    p_scan.add_argument("--out", default=None, help="optional JSON file path to save scan output")

    p_plan = sub.add_parser("plan-trade", help="v2 pre-trade position sizing, breakeven, TP ladder & liquidation calculator")
    p_plan.add_argument("--symbol", required=True, help="e.g. BTCUSDT, XAUUSDT, NDX100USDT")
    p_plan.add_argument("--side", required=True, choices=["long", "short"])
    p_plan.add_argument("--equity", type=float, default=500.0, help="account equity in USDT")
    p_plan.add_argument("--entry", type=float, default=None, help="planned entry price (fetches live mark if omitted with --live)")
    p_plan.add_argument("--stop", type=float, default=None, help="stop-loss price (uses 2*ATR14 if omitted with --live)")
    p_plan.add_argument("--tp", type=float, default=None, help="optional custom take-profit target price")
    p_plan.add_argument("--rr", type=float, default=2.5, help="default R-multiple target if --tp is omitted")
    p_plan.add_argument("--leverage", type=float, default=2.0)
    p_plan.add_argument("--risk-pct", type=float, default=1.0, help="risk budget percentage of equity, e.g. 1.0")
    p_plan.add_argument("--order-type", choices=["limit", "market"], default="limit")
    p_plan.add_argument("--live", action="store_true", help="fetch live Bitget orderbook, tiers, funding & candles")

    p_pos = sub.add_parser("position-advice", help="v2 open-position PnL, R-multiple, breakeven & trailing stop advisor")
    p_pos.add_argument("--symbol", required=True)
    p_pos.add_argument("--side", required=True, choices=["long", "short"])
    p_pos.add_argument("--entry", type=float, required=True)
    p_pos.add_argument("--qty", type=float, required=True)
    p_pos.add_argument("--stop", type=float, required=True)
    p_pos.add_argument("--leverage", type=float, default=2.0)
    p_pos.add_argument("--mark", type=float, default=None, help="current mark price (or pass --live)")
    p_pos.add_argument("--live", action="store_true", help="fetch live Bitget mark price and closed 1h candles")

    sub.add_parser("agent-verify-freeze", help="verify SHA-256 freeze manifest of PIIA v2 protocol and modules")
    p_arun = sub.add_parser("agent-run", help="run one prospective PIIA v2 6-pillar cycle and append to continuous paper ledger")
    p_arun.add_argument("--demo", action="store_true", help="run on labeled synthetic snapshot without network")
    p_arun.add_argument("--starting-cash", type=float, default=500.0)
    p_arun.add_argument("--db", default=None, help="optional sqlite path (defaults to data/piia/piia_v2_continuous.sqlite)")

    p_astat = sub.add_parser("agent-status", help="inspect PIIA v2 continuous prospective paper ledger & shadow baseline comparison")
    p_astat.add_argument("--db", default=None)

    p_aapp = sub.add_parser("agent-approve", help="explicitly approve or reject a pending PIIA v2 human paper proposal (never live)")
    p_aapp.add_argument("--proposal-id", required=True)
    p_aapp.add_argument("--choice", required=True, choices=["approve", "reject"])
    p_aapp.add_argument("--actor", default="local-investor")
    p_aapp.add_argument("--note", default="")
    p_aapp.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    if args.cmd in {"migrate", "downgrade"}:
        from alembic.config import Config
        from alembic import command
        cfg = Config(str(root / "alembic.ini"))
        if args.cmd == "migrate":
            command.upgrade(cfg, "head")
        else:
            command.downgrade(cfg, "base")
        return 0
    if args.cmd == "serve":
        if args.host not in {"127.0.0.1", "localhost"}:
            print("refusing to bind a non-localhost host", file=sys.stderr)
            return 2
        import uvicorn
        uvicorn.run("piia.intelligence.api:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.cmd == "ui":
        if args.host not in {"127.0.0.1", "localhost"}:
            print("refusing to bind a non-localhost host", file=sys.stderr)
            return 2
        app = root / "app" / "piia_app.py"
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app),
                                  "--server.address", args.host, "--server.port", str(args.port),
                                  "--server.headless", "true"])
        return 0
    if args.cmd == "replay-btc":
        from piia.intelligence.paper.replay import run_btc_historical_replay, write_report
        data_dir = Path(args.data_dir) if args.data_dir else root / "data" / "perps"
        db = Path(args.db) if args.db else root / "data" / "piia" / "piia.historical_replay.sqlite"
        out = Path(args.out) if args.out else root / "artifacts" / "btc_historical_replay.json"
        live = root / "data" / "piia" / "piia.dev.sqlite"
        if db.resolve() == live.resolve():
            print("refusing to write historical replay into the live research database", file=sys.stderr)
            return 2
        result = run_btc_historical_replay(
            db_path=db,
            evidence_store=db.parent / "replay_evidence",
            data_dir=data_dir,
            starting_cash=args.starting_cash,
        )
        write_report(result, out)
        print(out.read_text() if False else (out.with_suffix(".txt")).read_text())
        return 0
    if args.cmd == "research-export":
        from piia.intelligence.public_data import fetch_public_snapshot
        from piia.intelligence.research_packet import build_request, capture_paper_risk_context, default_packet_dir, write_json
        risk_context = capture_paper_risk_context()
        snap = fetch_public_snapshot()
        snap["paper_risk_context"] = risk_context
        if args.include_macro:
            from piia.intelligence.macro_data import fetch_alfred_context
            snap["macro_context"] = fetch_alfred_context()
        req = build_request(snap)
        out = Path(args.out) if args.out else default_packet_dir() / "request.json"
        write_json(out, req)
        # also write a slim request (no full candle arrays duplicated in stdout)
        print(f"wrote {out} request_hash={req['request_hash']} cutoff={req['cutoff_at_utc']} ok={req['snapshot_ok']}")
        return 0 if req["snapshot_ok"] else 2
    if args.cmd == "research-import":
        import json
        from piia.intelligence.research_packet import import_response, write_json, default_packet_dir
        live = root / "data" / "piia" / "piia.dev.sqlite"
        db = Path(args.db) if args.db else root / "data" / "piia" / "piia.research_packets.sqlite"
        if db.resolve() == live.resolve():
            print("refusing to import into the live research database", file=sys.stderr)
            return 2
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        response = json.loads(Path(args.response).read_text(encoding="utf-8"))
        result = import_response(
            request, response, db_path=db,
            evidence_store=db.parent / "research_packet_evidence",
        )
        out = Path(args.out) if args.out else default_packet_dir() / "import_result.json"
        write_json(out, result)
        print(json.dumps({k: result[k] for k in result if k != "instruments"}, indent=2, default=str))
        for inst in result["instruments"]:
            print(f"{inst['symbol']}: direction={inst['direction']} action={inst['suggested_action']} thesis={inst['thesis_id']} plan={inst['plan']}")
        return 0
    if args.cmd == "research-cycle":
        import json
        from piia.intelligence.research_cycle import default_cycle_dir, import_roles, materialize_followups, prepare_run, revalidate_execution, status as cycle_status
        if args.cycle_cmd == "prepare":
            state = prepare_run(Path(args.out) if args.out else default_cycle_dir(),
                                allow_forecast_download=args.allow_forecast_download)
            printable = {k: state[k] for k in state}
            print(json.dumps(printable, indent=2, default=str, ensure_ascii=False))
            return 0 if state.get("phase") == "awaiting_independent_roles" else 2
        if args.cycle_cmd == "followups":
            state = materialize_followups(Path(args.run))
            print(json.dumps(state, indent=2, default=str, ensure_ascii=False))
            return 0
        if args.cycle_cmd == "import-roles":
            state = import_roles(Path(args.run), db_path=Path(args.db) if args.db else None)
            print(json.dumps(state, indent=2, default=str, ensure_ascii=False))
            return 0
        if args.cycle_cmd == "status":
            print(json.dumps(cycle_status(Path(args.run)), indent=2, default=str, ensure_ascii=False))
            return 0
        if args.cycle_cmd == "revalidate":
            result = revalidate_execution(Path(args.run))
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
            return 0
        if args.cycle_cmd == "tiers":
            from piia.intelligence.public_position_tiers import fetch_position_tiers
            out = Path(args.out) if args.out else None
            result = fetch_position_tiers(snapshot_dir=out)
            # Compact report: do not dump every ladder row twice.
            compact = {k: result[k] for k in result if k != "instruments"}
            compact["instruments"] = {}
            for symbol, item in (result.get("instruments") or {}).items():
                compact["instruments"][symbol] = {
                    "ok": item.get("ok"),
                    "reason": item.get("reason"),
                    "endpoint_request_time_utc": item.get("endpoint_request_time_utc"),
                    "retrieved_at_utc": item.get("retrieved_at_utc"),
                    "raw_sha256": item.get("raw_sha256"),
                    "unit": item.get("unit"),
                    "tier_count": item.get("tier_count"),
                    "tiers": item.get("tiers"),
                    "exact_liquidation_verified": False,
                    "historical_replay_supported": False,
                }
            print(json.dumps(compact, indent=2, default=str, ensure_ascii=False))
            return 0 if result.get("ok") else 2
        return 2
    if args.cmd == "liquidity-positioning":
        import json
        from piia.intelligence.liquidity_positioning import fetch_and_assess
        prior = None
        if args.prior_json:
            prior = json.loads(Path(args.prior_json).read_text(encoding="utf-8"))
        result = fetch_and_assess(prior_state=prior)
        printable = {k: result[k] for k in result if k != "features"}
        features = result.get("features") or {}
        printable["feature_status"] = {
            "real_yield": (features.get("real_yield") or {}).get("status"),
            "etf_flows": (features.get("etf_flows") or {}).get("status"),
            "btc_oi": ((features.get("open_interest") or {}).get("BTCUSDT") or {}).get("status"),
            "btc_liquidations": ((features.get("liquidations") or {}).get("BTCUSDT") or {}).get("status"),
            "gold_sensitivity": (features.get("gold_real_rate_sensitivity") or {}).get("status"),
        }
        text_out = json.dumps(printable, indent=2, default=str, ensure_ascii=False)
        print(text_out)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text_out + "\n", encoding="utf-8")
        return 0
    if args.cmd == "scan":
        import json
        from piia.intelligence.trade_assistant import format_scan_table, scan_market_signals
        result = scan_market_signals(
            equity_usdt=args.equity,
            leverage=args.leverage,
            risk_fraction=(args.risk_pct / 100.0) if args.risk_pct is not None else None,
            order_type=args.order_type,
            demo_mode=args.demo,
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            print(format_scan_table(result))
        return 0
    if args.cmd == "plan-trade":
        import json
        from piia.intelligence.trade_assistant import build_trade_plan
        ticket = build_trade_plan(
            symbol=args.symbol,
            side=args.side,
            equity_usdt=args.equity,
            entry_price=args.entry,
            stop_loss=args.stop,
            take_profit=args.tp,
            target_rr=args.rr,
            leverage=args.leverage,
            risk_pct=args.risk_pct,
            order_type=args.order_type,
            fetch_live_context=args.live,
        )
        print(json.dumps(ticket, indent=2, ensure_ascii=False, default=str))
        return 0 if ticket.get("actionable_for_manual_review") else 2
    if args.cmd == "position-advice":
        import json
        from piia.intelligence.trade_assistant import advise_open_position
        advice = advise_open_position(
            symbol=args.symbol,
            side=args.side,
            entry_price=args.entry,
            qty=args.qty,
            stop_loss=args.stop,
            leverage=args.leverage,
            mark_price=args.mark,
            fetch_live=args.live,
        )
        print(json.dumps(advice, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.cmd == "agent-verify-freeze":
        import json
        from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
        res = verify_piia_v2_frozen()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["ok"] else 2
    if args.cmd == "agent-run":
        import json
        from piia.intelligence.agent_v2.continuous_paper import DEFAULT_LEDGER_PATH, run_prospective_cycle
        db_p = Path(args.db) if args.db else DEFAULT_LEDGER_PATH
        res = run_prospective_cycle(db_path=db_p, starting_cash=args.starting_cash, demo_mode=args.demo)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.cmd == "agent-status":
        import json
        from piia.intelligence.agent_v2.continuous_paper import DEFAULT_LEDGER_PATH, get_continuous_paper_status
        db_p = Path(args.db) if args.db else DEFAULT_LEDGER_PATH
        res = get_continuous_paper_status(db_path=db_p)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.cmd == "agent-approve":
        import json
        from piia.intelligence.agent_v2.continuous_paper import DEFAULT_LEDGER_PATH, approve_human_proposal
        db_p = Path(args.db) if args.db else DEFAULT_LEDGER_PATH
        res = approve_human_proposal(
            args.proposal_id, choice=args.choice, actor=args.actor, note=args.note, db_path=db_p
        )
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
