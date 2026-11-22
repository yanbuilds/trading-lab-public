from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from trading_lab.baselines import buy_and_hold
from trading_lab.config import load_config
from trading_lab.data import fetch_all, load_panel_from_cache, stale_report
from trading_lab.engine import run_backtest
from trading_lab.metrics import performance_to_dict, summarize
from trading_lab.paper import latest_signal_replay, write_snapshot
from trading_lab.scenarios import scenario_report
from trading_lab.timeutil import format_shanghai, now_shanghai

st.set_page_config(page_title="trading-lab 研究台", layout="wide")


def _cfg():
    return load_config()


@st.cache_data(show_spinner=False)
def _cached_panel_stamp():
    # 只作手动刷新的钥匙，不缓存行情本身以免掩盖过期。
    return format_shanghai(now_shanghai())


def load_or_fetch(force: bool):
    cfg = _cfg()
    if force:
        return fetch_all(cfg, force=True), cfg
    panel = load_panel_from_cache(cfg)
    if any(k not in panel.frames for k in cfg.instruments):
        return fetch_all(cfg, force=False), cfg
    return panel, cfg


st.title("trading-lab：BTC / 黄金ETF / 纳指100ETF 研究与信号回放")
st.caption(
    f"本地研究工具，不是实盘。页面时钟 {format_shanghai(now_shanghai())}。"
    "QQQ 是纳斯达克100的可交易 ETF 代理，不是指数本身；GLD 是黄金 ETF 代理，不是现货 XAU 或期货。"
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    do_fetch = st.button("拉取/刷新公开行情", type="primary")
with col_b:
    run_bt = st.button("运行回测")
with col_c:
    run_paper = st.button("回放最新信号")

error_box = st.empty()

try:
    panel, cfg = load_or_fetch(force=do_fetch)
except Exception as e:
    st.error(f"行情不可用：{type(e).__name__}: {e}")
    st.info("可以把真实 CSV 放到 data/csv/，文件名为 BTC-USD.csv、GLD.csv、QQQ.csv，列包含 Date,Open,High,Low,Close,Volume。不会用合成数据顶替。")
    st.stop()

if panel.warnings:
    st.warning("\n\n".join(panel.warnings))

st.subheader("数据状态")
st.caption("时间均为已完成 K 线或拉取时刻，已换算标注 Asia/Shanghai。未完成的当日 K 线会被丢弃。")
rows = []
stale = stale_report(panel, cfg)
for k, inst in cfg.instruments.items():
    meta = panel.metas.get(k)
    s = stale.get(k, {})
    rows.append(
        {
            "标的": k,
            "代理含义": inst.proxy_for,
            "来源": None if meta is None else meta.source,
            "可用": panel.availability.get(k, "missing"),
            "首根": None if meta is None else str(meta.first_bar),
            "最近已完成": None if meta is None else str(meta.last_complete_bar),
            "行数": None if meta is None else meta.rows,
            "是否过期": "是" if s.get("stale") else "否",
            "说明": s.get("reason", ""),
            "拉取时间": None if meta is None else meta.to_dict().get("fetched_at_shanghai"),
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch")

st.subheader("锁定的默认规则（示例，不是建议）")
st.markdown(
    f"""
- 策略：已完成收盘价高于 {cfg.sma_window} 日均线则在**下一交易时段开盘**做多，否则开盘卖到现金。
- 示例资金：{cfg.example_capital_usd:,.0f} USD（不是你的真实资金）。
- 单标的上限 {cfg.max_weight:.0%}，现金下限 {cfg.cash_floor:.0%}，回撤 {cfg.max_drawdown_halt:.0%} 起停止新开仓。
- 只做多、不融资、不做空，总多头不超过 100%。
- 训练段截止 {cfg.train_end}，测试段从 {cfg.test_start} 起。窗口与阈值在测试段不再调参。
- 大模型不参与信号和风控。
"""
)

if any(s.get("stale") for s in stale.values()):
    st.error("存在过期行情：不会发出新的交易信号。请刷新数据或检查数据源。")

if run_bt or "bt_result" not in st.session_state:
    if run_bt or st.session_state.get("auto_once") is None:
        try:
            with st.spinner("正在回测…"):
                result = run_backtest(panel, cfg)
                st.session_state["bt_result"] = result
                st.session_state["auto_once"] = True
        except Exception as e:
            st.error(f"回测失败：{type(e).__name__}: {e}")
            result = None
    else:
        result = st.session_state.get("bt_result")
else:
    result = st.session_state.get("bt_result")

# Always show last result if we have one
result = st.session_state.get("bt_result")
if result is not None and not result.nav.empty:
    st.subheader("回测净值（开发检查，非已审计绩效）")
    st.warning("已修正为UTC开收盘事件排序；ETF卖出款到下一美股交易时段开盘才可用，不融资。仍是假设开盘全量成交的研究回测，非独立审计绩效；收盘权重可漂移。")
    st.line_chart(result.nav.rename("策略净值"))
    train = result.slice_dates(cfg.history_start, cfg.train_end)
    test = result.slice_dates(cfg.test_start)
    table = []
    for label, sl in (("全样本", result), ("训练段", train), ("测试段（样本外）", test)):
        if sl.nav.empty:
            continue
        table.append(performance_to_dict(summarize(sl.nav, sl.trades, label=label, holdings=sl.holdings)))
    try:
        bh_nav, bh_trades, bh_perf = buy_and_hold(panel, cfg, start=cfg.test_start)
        table.append(performance_to_dict(bh_perf))
        st.caption("买住基准是测试段一次性等权买入并持有，目标接近满仓，和带现金下限的策略风险不同。")
        chart = pd.DataFrame({"策略（测试段）": test.nav, "买住基准": bh_nav}).dropna(how="all")
        st.line_chart(chart)
    except Exception as e:
        st.warning(f"买住基准无法计算：{e}")
    st.dataframe(pd.DataFrame(table), width="stretch")

    if not result.trades.empty:
        st.subheader("成交明细（回测）")
        st.dataframe(result.trades.tail(50), width="stretch")

    st.subheader("情景（不是预测）")
    sc = scenario_report(result, panel, cfg)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("毛敞口", f"{sc['gross_exposure']:.1%}")
    c2.metric("杠杆", "0")
    c3.metric("最大单标的权重", f"{sc['concentration']['max_name_weight']:.1%}")
    c4.metric("回测最大回撤", sc["max_drawdown_in_backtest"])
    st.write(sc["leverage_note"])
    st.write(sc["liquidity_note"])
    st.write(
        "同时下跌日数（各标的当日都有 K 线且收跌）:",
        sc["simultaneous_declines"]["common_session_days_all_present_down"],
        "最近一次:",
        sc["simultaneous_declines"]["last_such_day"],
    )
    st.write(
        "若最近标记价同时下跌 10%（不假设能先卖出），净值约变为",
        sc["hypothetical_same_mark_shock"]["nav_after"],
        f"（相对最近净值 {sc['hypothetical_same_mark_shock']['loss_fraction_vs_last_nav']}）。",
    )

st.subheader("最新信号回放（不是常驻纸上交易）")
if run_paper or True:
    snap = latest_signal_replay(panel, cfg)
    if run_paper:
        path, unchanged = write_snapshot(snap)
        st.info(f"已写入快照 {path}（指纹相同={unchanged}）")
    st.write("状态：", snap["status"], " 生成时间：", snap["generated_at_shanghai"])
    sig_rows = []
    for k in cfg.instruments:
        sig_rows.append(
            {
                "标的": k,
                "最近已完成K线": snap["last_completed_bars"].get(k),
                "信号": snap["signals"].get(k),
                "目标权重": snap["target_weights"].get(k),
                "标记价": snap["marks"].get(k),
            }
        )
    st.dataframe(pd.DataFrame(sig_rows), width="stretch")
    if snap["blocked_new_signals"]:
        st.error("过期数据已阻止新信号。")
    elif not snap["intended_orders"]:
        st.write("当前没有再平衡意向单。")
    else:
        st.write("意向单将在各标的**下一根开盘**才能成交，不会用最新收盘价虚构成交。")
        st.dataframe(pd.DataFrame(snap["intended_orders"]), width="stretch")
    with st.expander("说明与限制"):
        for n in snap["notes"]:
            st.write("- " + n)
        st.write(snap["disclaimer"])

st.divider()
st.caption(
    "本工具不保证盈利。vectorbt 社区版（Apache 2.0 + Commons Clause）已安装，"
    "但多资产混日历组合记账使用本仓库的显式引擎，差异见 README。"
)
