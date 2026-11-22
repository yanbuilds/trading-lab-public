from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trading_lab import __version__
from trading_lab.baselines import buy_and_hold
from trading_lab.config import load_config
from trading_lab.data import fetch_all, load_panel_from_cache, stale_report
from trading_lab.engine import run_backtest
from trading_lab.metrics import performance_to_dict, summarize
from trading_lab.paper import latest_signal_replay, write_snapshot
from trading_lab.paths import ARTIFACTS, ROOT
from trading_lab.scenarios import scenario_report
from trading_lab.timeutil import format_shanghai, now_shanghai


def _print(msg: str) -> None:
    print(msg, flush=True)


def _require_panel(cfg, fetch: bool):
    if fetch:
        return fetch_all(cfg, force=True)
    panel = load_panel_from_cache(cfg)
    missing = [k for k in cfg.instruments if k not in panel.frames]
    if missing:
        _print("缓存不完整，开始拉取公开行情: " + ", ".join(missing))
        return fetch_all(cfg, force=False)
    return panel


def cmd_fetch(args) -> int:
    cfg = load_config()
    panel = fetch_all(cfg, force=args.force)
    _print(f"拉取时间 {format_shanghai(now_shanghai())}")
    for k, meta in panel.metas.items():
        _print(
            f"- {k}: source={meta.source} rows={meta.rows} "
            f"{meta.first_bar} → 最近已完成 {meta.last_complete_bar} "
            f"availability={panel.availability.get(k)}"
        )
        for n in meta.notes:
            _print(f"  说明: {n}")
    for w in panel.warnings:
        _print(f"警告: {w}")
    missing = [k for k, a in panel.availability.items() if a == "missing"]
    return 1 if missing else 0


def cmd_backtest(args) -> int:
    cfg = load_config()
    panel = _require_panel(cfg, fetch=args.fetch)
    if len(panel.frames) != len(cfg.instruments):
        _print("有标的缺失，仍对可用标的回测。" if panel.frames else "没有数据，中止。")
        if not panel.frames:
            return 1
    result = run_backtest(panel, cfg)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nav_path = ARTIFACTS / "backtest_nav.csv"
    result.nav.to_csv(nav_path, header=["nav"])
    if not result.trades.empty:
        result.trades.to_csv(ARTIFACTS / "backtest_trades.csv", index=False)

    train = result.slice_dates(cfg.history_start, cfg.train_end)
    test = result.slice_dates(cfg.test_start)
    reports = []
    for label, sl in (("全样本", result), ("训练段（锁定参数，含均线热身）", train), ("测试段（样本外）", test)):
        if sl.nav.empty:
            _print(f"{label}: 无净值")
            continue
        perf = summarize(sl.nav, sl.trades, label=label, holdings=sl.holdings, notes=sl.notes)
        reports.append(performance_to_dict(perf))
        _print(f"=== {label} {perf.start} → {perf.end} ===")
        _print(
            f"净收益 {perf.total_return:.2%}  最大回撤 {perf.max_drawdown:.2%}  "
            f"换手 {perf.turnover:.2f}  成交 {perf.trade_count} 笔"
        )
    try:
        bh_nav, bh_trades, bh_perf = buy_and_hold(panel, cfg, start=cfg.test_start)
        _print(
            f"=== 测试段买住基准 {bh_perf.start} → {bh_perf.end} ===  "
            f"净收益 {bh_perf.total_return:.2%}  最大回撤 {bh_perf.max_drawdown:.2%}"
        )
        reports.append(performance_to_dict(bh_perf))
        bh_nav.to_csv(ARTIFACTS / "buyhold_test_nav.csv", header=["nav"])
    except Exception as e:  # noqa: BLE001
        _print(f"买住基准失败: {e}")

    (ARTIFACTS / "backtest_summary.json").write_text(
        json.dumps({"reports": reports, "notes": result.notes, "as_of_shanghai": format_shanghai(now_shanghai())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print(f"已写入 {nav_path}")
    _print("以上为研究回测，不是实盘，也不保证未来收益。")
    return 0


def cmd_paper(args) -> int:
    cfg = load_config()
    panel = _require_panel(cfg, fetch=args.fetch)
    snap = latest_signal_replay(panel, cfg)
    path, unchanged = write_snapshot(snap)
    _print(f"状态: {snap['status']}")
    _print(f"生成时间: {snap['generated_at_shanghai']}")
    _print(f"指纹: {snap['fingerprint'][:16]}…  与已有快照相同={unchanged}")
    _print(f"写入: {path}")
    for k, s in (snap.get("stale") or {}).items():
        flag = "过期" if s.get("stale") else "新鲜"
        _print(f"- {k} [{flag}] {s.get('reason')}")
    _print("意向单:")
    if snap["blocked_new_signals"]:
        _print("  （已停止新信号）")
    elif not snap["intended_orders"]:
        _print("  无再平衡意向")
    else:
        for o in snap["intended_orders"]:
            _print(f"  {o['side']} {o['symbol']} qty={o['qty_estimate_at_last_mark']:.6f}  {o['execute_at']}")
    return 1 if snap["blocked_new_signals"] else 0


def cmd_scenarios(args) -> int:
    cfg = load_config()
    panel = _require_panel(cfg, fetch=args.fetch)
    result = run_backtest(panel, cfg)
    rep = scenario_report(result, panel, cfg)
    path = ARTIFACTS / "scenarios.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(json.dumps(rep, ensure_ascii=False, indent=2))
    _print(f"已写入 {path}")
    return 0


def cmd_stale(args) -> int:
    cfg = load_config()
    panel = _require_panel(cfg, fetch=args.fetch)
    rep = stale_report(panel, cfg)
    _print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 1 if any(v.get("stale") for v in rep.values()) else 0


def cmd_perps(args) -> int:
    from trading_lab.perps.research import fetch, backtest
    if args.action == 'fetch':
        _print(json.dumps(fetch(args.days), ensure_ascii=False, indent=2))
    else:
        backtest()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trading-lab",
        description="BTC-USD / GLD / QQQ 研究回测与最新信号回放（非实盘）。",
    )
    p.add_argument("--version", action="version", version=f"trading-lab {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_fetch_flag(sp):
        sp.add_argument("--fetch", action="store_true", help="先强制/补齐拉取公开行情")
        return sp

    f = sub.add_parser("fetch", help="拉取并缓存公开日线")
    f.add_argument("--force", action="store_true", help="忽略本地 CSV 优先，重新下载")
    f.set_defaults(func=cmd_fetch)

    add_fetch_flag(sub.add_parser("backtest", help="运行研究回测")).set_defaults(func=cmd_backtest)
    add_fetch_flag(sub.add_parser("paper", help="最新信号回放（非常驻模拟盘）")).set_defaults(func=cmd_paper)
    add_fetch_flag(sub.add_parser("scenarios", help="集中度/杠杆0/同时下跌等情景")).set_defaults(func=cmd_scenarios)
    add_fetch_flag(sub.add_parser("stale", help="检查行情是否过期")).set_defaults(func=cmd_stale)
    perps = sub.add_parser('perps', help='独立 Bitget 多空合约研究，非旧 ETF 回测、非真实下单')
    perps.add_argument('action', choices=['fetch', 'backtest'])
    perps.add_argument('--days', type=int, default=60)
    perps.set_defaults(func=cmd_perps)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as e:  # noqa: BLE001
        _print(f"错误: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
