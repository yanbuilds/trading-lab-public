"""Traceable conversational research workbench bound strictly to a JudgmentPacket snapshot."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from piia.intelligence.agent_v2.pillars import SYMBOLS
from piia.intelligence.judgment.evidence import load_judgment_config
from piia.intelligence.judgment.synthesizer import BIAS_CN, HORIZONS, _synthesize_one

SYMBOL_LABELS = {
    "XAUUSDT": "黄金 (XAUUSDT)",
    "BTCUSDT": "比特币 (BTCUSDT)",
    "NDX100USDT": "纳指100 (NDX100USDT)",
}
ORDERED_SYMBOLS = ("XAUUSDT", "BTCUSDT", "NDX100USDT")
_BJ_TZ = timezone(timedelta(hours=8))
_DIR_CN = {
    "bullish": "偏多",
    "bearish": "偏空",
    "two_sided_risk": "双边风险",
    "neutral": "中性",
    "unknown": "未知",
}
_ROLE_CN = {
    "changed_direction": " · 改判驱动",
    "confirmed_direction": " · 顺向确认",
    "reduced_conviction": " · 风险反证",
    "triggered_veto": " · 触发否决",
    "context_only": "",
}
_ATOM_LINE_RE = re.compile(
    r"^\[(?P<eid>[^\]]+)\]\s*(?P<name>.+)\s+\((?P<src>[^,]+),\s*(?P<ts>\d{4}-\d{2}-\d{2}T[^)]+)\):\s*dir=(?P<dir>[a-z_]+),\s*role=(?P<role>[a-z_]+)$"
)
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)")
_LONG_DEC_RE = re.compile(r"\b(\d+\.\d{3,})\b")


def _extract_symbols(question: str) -> list[str]:
    q = question.lower()
    picked = []
    if any(k in q for k in ("黄金", "gold", "xau")):
        picked.append("XAUUSDT")
    if any(k in q for k in ("btc", "比特币", "bitcoin")):
        picked.append("BTCUSDT")
    if any(k in q for k in ("纳指", "纳斯达克", "nasdaq", "ndx")):
        picked.append("NDX100USDT")
    return picked or list(ORDERED_SYMBOLS)


def classify_chat_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if any(k in q for k in ("相比上次", "上次", "变化", "哪条证据变", "diff", "changed")):
        return "diff_since_last"
    if any(k in q for k in ("如果", "假如", "要是", "若", "what if", "would flip")):
        return "what_if_sensitivity"
    if any(k in q for k in ("为什么", "为何", "why", "不做空", "不做多", "不看空", "不看多", "观望")):
        return "why_not_direction"
    return "overview"


def build_summary_table_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sym in ORDERED_SYMBOLS:
        j = packet["judgments"][sym]
        i_d = j["intraday"]
        s_d = j["swing_1_3d"]
        pe = j["paper_eligibility"]
        tech_str = f"{i_d['technical_only']['label_cn']} / {s_d['technical_only']['label_cn']}"
        changed_tags = []
        if i_d["changed_vs_technical"]:
            changed_tags.append("日内改判")
        if s_d["changed_vs_technical"]:
            changed_tags.append("1-3日改判")
        if pe.get("status") == "pending_review" or pe.get("review_required"):
            pe_str = f"待复核 ({', '.join(pe.get('review_triggers') or pe.get('blocking_reasons') or [])})"
        elif pe.get("eligible"):
            pe_str = "是"
        else:
            pe_str = f"否 ({', '.join(pe.get('blocking_reasons') or ['stand_aside'])})"
        rows.append(
            {
                "symbol": sym,
                "label": SYMBOL_LABELS[sym],
                "last_close": j.get("last_close"),
                "technical_only": tech_str,
                "intraday": i_d["final_label_cn"],
                "swing_1_3d": s_d["final_label_cn"],
                "changed_vs_technical": "；".join(changed_tags) if changed_tags else "与纯技术面同向",
                "official_v2_2_1_signal": j["official_v2_2_1_signal"],
                "paper_eligible": pe_str,
                "paper_status": pe.get("status") or ("eligible" if pe.get("eligible") else "blocked"),
                "top_data_gap": (i_d["data_gaps"] or s_d["data_gaps"] or ["无"])[0],
            }
        )
    return rows


def _parse_dt(ts_val: Any) -> datetime | None:
    if not ts_val:
        return None
    try:
        s = str(ts_val).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _fmt_utc_short(ts_val: Any, *, include_sec: bool = False) -> str:
    dt = _parse_dt(ts_val)
    if dt is None:
        return "无" if ts_val is None else str(ts_val)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC" if include_sec else "%Y-%m-%d %H:%M UTC")


def _fmt_bj_short(ts_val: Any) -> str:
    dt = _parse_dt(ts_val)
    if dt is None:
        return str(ts_val or "未知")
    return dt.astimezone(_BJ_TZ).strftime("%Y-%m-%d %H:%M 北京时间")


def _fmt_num(val: Any, digits: int = 2, suffix: str = "") -> str:
    if val is None or isinstance(val, bool):
        return "缺失"
    try:
        return f"{float(val):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _bias_badge(label_cn: str) -> str:
    if label_cn == "偏多":
        return "🟢 偏多"
    if label_cn == "偏空":
        return "🔴 偏空"
    return f"⚪ {label_cn}"


def _humanize_gap(gap: str) -> str:
    parts = str(gap or "").split(":", 2)
    return parts[2].strip() if len(parts) == 3 else str(gap or "").strip()


def _round_long_decimals(text: str) -> str:
    return _LONG_DEC_RE.sub(lambda m: f"{float(m.group(1)):.2f}", text)


_SETUP_CN = {
    "range_reversion": "震荡极值均值回归",
    "trend_breakout": "趋势突破",
    "trend_pullback": "趋势回踩",
    "none": "无触发形态",
}
_FACTOR_PREFIX_LABELS = (
    ("macro:rate_expectations:DGS2_minus_DFF", "2Y-DFF 利率预期"),
    ("macro:nominal_rates:DGS10_DGS2", "10Y/2Y 美债收益率"),
    ("macro:real_yield_daily:DFII10", "10Y 实际利率"),
    ("macro:fed_net_liquidity", "美联储净流动性"),
    ("macro:broad_dollar_proxy:DTWEXBGS", "广义美元"),
    ("macro:intraday_dollar_proxy:DX_Y_NYB", "美元盘中代理"),
    ("macro:intraday_rates_tips_proxy:TNX_TIP", "盘中美债/TIPS"),
    ("macro:ice_dxy", "ICE DXY"),
    ("macro:real_yield_intraday", "盘中 TIPS 实际利率"),
    ("sentiment:btc:spot_etf_flows", "BTC 现货 ETF 净流"),
    ("sentiment:btc:onchain_exchange_netflow", "BTC 交易所链上净流"),
    ("sentiment:btc:onchain_network_activity", "BTC mempool 拥堵"),
    ("sentiment:btc:fear_greed", "恐慌贪婪指数"),
    ("sentiment:ndx:tech_relative_strength", "QQEW/QQQ 科技广度"),
    ("sentiment:ndx:vix_volatility_backdrop", "VIX 高波动"),
    ("sentiment:xau:cftc_cot_lagged", "黄金 COT 持仓"),
    ("sentiment:xau:gold_etf_flows", "黄金 ETF 净流"),
    ("sentiment:polymarket:filtered_events", "Polymarket 赔率"),
    ("tide:macro_backdrop", "宏观潮汐大环境"),
)


def _short_factor_label(raw: Any, *, with_dir: bool = False) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    dir_cn = ""
    eid = ""
    m = _ATOM_LINE_RE.match(s)
    if m:
        eid = m.group("eid")
        dir_cn = _DIR_CN.get(m.group("dir"), "")
    elif s.startswith("["):
        eid = s[1:].split("]", 1)[0]
    for prefix, label in _FACTOR_PREFIX_LABELS:
        if eid.startswith(prefix) or prefix in s:
            return f"{label}{dir_cn}" if (with_dir and dir_cn) else label
    if eid.endswith(":funding_and_oi") or "funding_and_oi" in s:
        return "资金费率/OI"
    if eid.endswith(":chronos_neural") or "chronos_neural" in s:
        return f"Chronos 预测{dir_cn}" if (with_dir and dir_cn) else "Chronos 预测"
    if eid.endswith(":statistical_drift_168h") or "statistical_drift_168h" in s:
        return f"168h 统计漂移{dir_cn}" if (with_dir and dir_cn) else "168h 统计漂移"
    if eid.startswith("news:") or "[原始]" in s:
        body = re.sub(r"^\[[^\]]+\]\s*(\[原始\]\s*)?", "", s)
        head = body.split("(", 1)[0].strip()
        return (head[:26] + "…") if len(head) > 26 else head
    if "向上突破反转候选" in s:
        return "1H 上破 48h 高点"
    if "向下破位反转候选" in s:
        return "1H 下破 48h 低点"
    if "潮汐大环境" in s:
        return "宏观潮汐大环境"
    return _humanize_evidence_item(s)


def _humanize_evidence_item(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    m = _ATOM_LINE_RE.match(s)
    if m:
        d_cn = _DIR_CN.get(m.group("dir"), m.group("dir"))
        r_cn = _ROLE_CN.get(m.group("role"), "")
        ts_clean = _fmt_utc_short(m.group("ts"))
        name_clean = re.sub(r"^\[原始\]\s*", "", m.group("name").strip())
        return f"**{d_cn}{r_cn}** — {name_clean}（{m.group('src').strip()}，{ts_clean}）"
    if s.startswith("[tide:macro_backdrop:"):
        body = re.sub(r"^\[tide:macro_backdrop:[^\]]+\]\s*", "", s)
        body = re.sub(r"\s*->\s*[A-Z0-9]+\s+tailwind$", "（顺风支撑）", body)
        body = re.sub(r"\s*->\s*[A-Z0-9]+\s+headwind$", "（逆风压制）", body)
        return body
    s = re.sub(r"^\[[^\]]+\]\s*", "", s)
    s = _ISO_TS_RE.sub(lambda tm: _fmt_utc_short(tm.group(0)), s)
    return _round_long_decimals(s)


def _dedup_list(items: list[Any], limit: int, *, humanize: bool = False) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = _humanize_evidence_item(item) if humanize else str(item or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def _concise_rationale(sym: str, j: dict[str, Any], pos: dict[str, Any]) -> str:
    """Build a crisp 1-line trader explanation of why intraday & 1-3d are long/short/neutral."""
    i_d, s_d = j["intraday"], j["swing_1_3d"]
    i_lbl, s_lbl = i_d["final_label_cn"], s_d["final_label_cn"]
    pstate = pos.get("position_state") or ""
    adx = (pos.get("metrics") or {}).get("adx14")
    adx_txt = f" (ADX {float(adx):.1f})" if adx is not None else ""

    if i_d["hard_gates_triggered"] or s_d["hard_gates_triggered"]:
        gates = i_d["hard_gates_triggered"] or s_d["hard_gates_triggered"]
        return f"触发风控门控（{gates[0]}）强制观望"

    if pstate == "range_consolidation":
        struct_txt = f"1H 震荡{adx_txt}"
    elif pstate == "potential_bullish_reversal":
        struct_txt = "1H 上破 48h 高点"
    elif pstate == "potential_bearish_reversal":
        struct_txt = "1H 下破 48h 低点"
    elif pstate in {"downtrend_aligned", "bearish_trend"} or "顺势空头" in str(pos.get("structure_kind_cn") or ""):
        struct_txt = "1D/4H/1H 空头共振"
    elif pstate in {"uptrend_aligned", "bullish_trend"} or "顺势多头" in str(pos.get("structure_kind_cn") or ""):
        struct_txt = "1D/4H/1H 多头共振"
    else:
        struct_txt = str(pos.get("structure_kind_cn") or "多周期分化")

    drv_labels = _dedup_list(
        [
            _short_factor_label(d.get("summary") or d.get("group"))
            for d in (*(i_d.get("active_causal_drivers") or []), *(s_d.get("active_causal_drivers") or []))
            if d.get("vote")
        ],
        3,
    )
    drv_txt = " + ".join(drv_labels)

    if i_d["changed_vs_technical"] or s_d["changed_vs_technical"]:
        verb = f"改判为{s_lbl}"
        base = f"{struct_txt}，叠加{drv_txt}共振{verb}" if drv_txt else f"{struct_txt}，多维因子共振{verb}"
    elif i_lbl == "观望" and s_lbl != "观望":
        base = f"日内{struct_txt}观望；1–3日随{pos.get('daily_dir') == 'down' and '日线空头' or '日线多头'}"
        if drv_txt:
            base += f"（{drv_txt}）{s_lbl}"
        else:
            base += s_lbl
    elif i_lbl == "观望" and s_lbl == "观望":
        base = f"{struct_txt}，外部因子分歧或不足以破局，继续观望"
    else:
        base = f"{struct_txt}，与外部驱动（{drv_txt}）同向确认{s_lbl}" if drv_txt else f"{struct_txt}确认{s_lbl}"

    pe = j.get("paper_eligibility") or {}
    if pe.get("eligible"):
        side_cn = "做空" if pe.get("side") == "short" else "做多"
        setup_cn = _SETUP_CN.get(str(j.get("official_v2_2_1_setup_type") or ""), str(j.get("official_v2_2_1_setup_type") or ""))
        base += f"；短线另触发 {setup_cn}{side_cn}"
    elif pe.get("status") == "pending_review":
        base += "；短线信号触发异常待复核"
    return base


def _render_trade_plan_lines(sym: str, j: dict[str, Any], pos: dict[str, Any], earliest_exec_utc: Any) -> list[str]:
    pe = j["paper_eligibility"]
    setup = str(j.get("official_v2_2_1_setup_type") or "none")
    setup_cn = _SETUP_CN.get(setup, setup)
    if pe.get("status") == "pending_review":
        trigs = "；".join(t.get("detail") or t.get("trigger") for t in (pe.get("review_trigger_details") or [])) or ", ".join(pe.get("review_triggers") or [])
        return [
            f"- **⚠️ {SYMBOL_LABELS[sym]}｜`v2.2.1` {j['official_v2_2_1_signal']} ({setup_cn}) — 待人工复核（暂拦截开仓）**",
            f"  - 拦截原因：{trigs}",
        ]
    if not pe.get("eligible"):
        return []

    ep = pe.get("entry_plan") or {}
    xp = pe.get("exit_plan") or {}
    ps = pe.get("position_sizing") or {}
    cr = pe.get("cost_and_risk") or {}
    lvl = pos.get("invalidation_levels") or {}

    side_cn = "做空" if pe.get("side") == "short" else "做多"
    ref_px = ep.get("reference_price") or j.get("last_close")
    fill_px = ep.get("assumed_fill_with_slippage") or ref_px
    sl_px = xp.get("initial_stop_loss")
    sl_fill = xp.get("assumed_stop_fill")
    be_trig = xp.get("breakeven_ratchet_trigger_price")
    be_stop = xp.get("fee_adjusted_breakeven_stop")
    max_h = xp.get("max_hold_hours")
    rule = str(xp.get("trailing_or_target_rule") or "")

    sl_pct_txt = f"（距入场 `{abs(float(sl_px) - float(fill_px)) / float(fill_px) * 100.0:.2f}%`）" if (sl_px and fill_px) else ""
    bb_mid = lvl.get("bb_mid")
    if "bb_sma20_mid" in rule and bb_mid and fill_px:
        tp_pct = ((float(fill_px) - float(bb_mid)) / float(fill_px) * 100.0) if pe.get("side") == "short" else ((float(bb_mid) - float(fill_px)) / float(fill_px) * 100.0)
        tp_txt = f"**{_fmt_num(bb_mid)}**（当前 1H 布林中轨 SMA20，空间 `{tp_pct:+.2f}%`，随每根 1H 收盘动态调整）"
    else:
        tp_txt = "Chandelier 22h 2.5×ATR 移动止盈（或 1H+4H 均线同向反转平仓）"

    be_txt = f"价格到达 `{_fmt_num(be_trig)}` 时止损上/下推至保本价 `{_fmt_num(be_stop)}`" if (be_trig and be_stop) else "无"
    hold_txt = f"**{max_h} 小时**（超时无论盈亏平仓；最早 `{_fmt_bj_short(earliest_exec_utc)}` / `{_fmt_utc_short(earliest_exec_utc)}` 起可执行）" if max_h else "按规则离场"

    return [
        f"##### ⚡ {SYMBOL_LABELS[sym]}｜`{side_cn}` · {setup_cn}（`v2.2.1` 纸面可执行）",
        f"- **计划入场**：参考价 `{_fmt_num(ref_px)}`（含滑点模拟成交约 `{_fmt_num(fill_px)}`，以下一根 1H 开盘起按规则挂单/市价）",
        f"- **初始止损**：**`{_fmt_num(sl_px)}`**{sl_pct_txt}（含滑点止损 `{_fmt_num(sl_fill)}`｜预计止损净亏 `{_fmt_num(cr.get('estimated_normal_stop_loss_usdt'))}` USDT / `{_fmt_num(cr.get('estimated_normal_stop_loss_pct_equity'))}%` 净值）",
        f"- **止盈与保本**：目标 {tp_txt}；**移动保本**：{be_txt}",
        f"- **最长持仓**：{hold_txt}",
        f"- **参考仓位与风控**：`{ps.get('qty_contracts')}` 合约（名义约 `{_fmt_num(ps.get('notional_usdt'))}` USDT，`{_fmt_num(ps.get('leverage'), 1)}x` 逐仓，保证金 `{_fmt_num(ps.get('initial_margin_usdt'))}` USDT，剩余现金 `{_fmt_num(ps.get('free_cash_after_entry_usdt'))}` USDT｜1% 风险预算 `{_fmt_num(cr.get('risk_budget_usdt'))}` USDT，估算强平 `{_fmt_num(cr.get('estimated_isolated_liquidation'))}`）",
    ]


def _render_overview(packet: dict[str, Any]) -> dict[str, Any]:
    table_rows = build_summary_table_rows(packet)
    lc = packet.get("lifecycle_timestamps") or {}
    cash_ref = packet.get("starting_cash_usdt")
    tide = packet["tide"]
    macro = packet["macro_summary"]
    sent_sum = packet.get("sentiment_summary") or {}
    vix = sent_sum.get("vix") or {}
    earliest_exec = lc.get("earliest_executable_at_utc") or packet["cutoff_utc"]

    lines = [
        f"### 三合约研判简报（{_fmt_bj_short(packet['cutoff_utc'])} / {_fmt_utc_short(packet['cutoff_utc'])}）",
        "",
        f"> 快照 `{packet['judgment_hash'][:12]}`｜参考资金 `{cash_ref}` USDT｜仅作研究参考，无实盘下单权限｜冻结 `v2.2.1` 信号与多维研判严格分栏",
        "",
        "#### 一、今日结论总览",
        "| 品种 | 最新收盘 | 日内 | 1–3日 | `v2.2.1` 信号 | 一句话逻辑 |",
        "| --- | ---: | :---: | :---: | :---: | --- |",
    ]
    for r in table_rows:
        sym = r["symbol"]
        j = packet["judgments"][sym]
        pos = tide["instrument_positions"][sym]
        sig_txt = f"⚡ `{r['official_v2_2_1_signal']}` ({r['paper_eligible']})" if r["official_v2_2_1_signal"] in {"long", "short"} else "`stand_aside`"
        rationale = _concise_rationale(sym, j, pos)
        lines.append(
            f"| **{r['label']}** | `{_fmt_num(r['last_close'])}` | **{_bias_badge(r['intraday'])}** | **{_bias_badge(r['swing_1_3d'])}** | {sig_txt} | {rationale} |"
        )

    plan_blocks: list[str] = []
    for sym in ORDERED_SYMBOLS:
        plan_blocks.extend(_render_trade_plan_lines(sym, packet["judgments"][sym], tide["instrument_positions"][sym], earliest_exec))

    lines.extend(["", "#### 二、可执行纸面交易计划（`v2.2.1` 正式信号）"])
    if plan_blocks:
        lines.extend(plan_blocks)
    else:
        lines.append("- 当前三合约均未触发 `v2.2.1` 可开仓信号，纸面保持空仓观望（等待触发条件见下）。")

    lines.extend(["", "#### 三、分品种研判与等待/失效位"])
    for sym in ORDERED_SYMBOLS:
        j = packet["judgments"][sym]
        pos = tide["instrument_positions"][sym]
        i_d, s_d = j["intraday"], j["swing_1_3d"]
        lvl = pos.get("invalidation_levels") or {}
        sb = j.get("setup_bridge") or {}
        va = j.get("venue_microstructure_audit") or {}
        supp = j.get("agent_or_human_supplement") or packet.get("agent_or_human_supplement") or {}

        raw_ctr = _dedup_list(list(i_d["strongest_counterevidence"]) + list(s_d["strongest_counterevidence"]), 3)
        ctr_short = _dedup_list([_short_factor_label(x, with_dir=True) for x in raw_ctr], 3)
        inv_line = f"4H EMA50 `{_fmt_num(lvl.get('h4_ema50'))}` / 48h 通道 `{_fmt_num(lvl.get('donchian_48h_low'))} ~ {_fmt_num(lvl.get('donchian_48h_high'))}` 收盘突破或宏观跨过阈值"

        lines.extend(
            [
                "",
                f"##### {SYMBOL_LABELS[sym]}｜`{pos['structure_kind_cn']}`",
                f"- **研判逻辑**：{_concise_rationale(sym, j, pos)}",
                f"- **主要反证与风险**：{('、'.join(ctr_short)) if ctr_short else '暂无明显强反证'}",
                f"- **失效参考线**：{inv_line}",
                f"- **研究→开仓等待条件 (`v2.2.1`)**：{sb.get('summary_cn') or '无'}",
                f"- **交易所微观结构盘口/资金费率**：{va.get('summary_cn') or '未采集'}",
            ]
        )
        if supp.get("provided"):
            lines.append(f"- **人工/Agent 补充判断**：{supp.get('actor')}: {supp.get('note')}（{_fmt_utc_short(supp.get('recorded_at_utc'))}，不可绕过风控）")

    fc_by_sym = (packet.get("forecast_and_models") or {}).get("by_symbol") or {}
    ai_prov = (packet.get("forecast_and_models") or {}).get("ai_provenance") or {}
    dx_y_txt = f"`{_fmt_num(macro.get('intraday_dollar_dx_y_nyb'))}` ({_fmt_num(macro.get('intraday_dollar_dx_y_nyb_pct_24h'), 2, '%')})" if macro.get("intraday_dollar_dx_y_nyb") is not None else "缺失"
    tnx_txt = f"`{_fmt_num(macro.get('intraday_tnx_pct'), 3, '%')}` ({_fmt_num(macro.get('intraday_tnx_delta_24h_bps'), 1, ' bps')})" if macro.get("intraday_tnx_pct") is not None else "缺失"
    dfii_txt = f"`{_fmt_num(macro.get('real_yield_dfii10_pct'), 2, '%')}` ({_fmt_num(macro.get('real_yield_delta_5obs_bps'), 1, ' bps')})" if macro.get("real_yield_dfii10_pct") is not None else "缺失"

    lines.extend(
        [
            "",
            "#### 四、宏观环境、Chronos 预测与审计速查",
            f"- **宏观与周期**：{tide['macro_backdrop']['state_cn']}｜周期：`{tide['cycle_stage']['stage_cn']}`",
            f"- **核心宏观代理**：美元 `DX-Y.NYB` {dx_y_txt} / 广义 `{_fmt_num(macro.get('broad_dollar_dtwexbgs'))}`｜利率 `^TNX` {tnx_txt} / 10Y实际利率 {dfii_txt}｜`VIX` `{_fmt_num(vix.get('value'))}`",
            "",
            "| 品种 | 现价 | Chronos 6h | Chronos 24h (对比现价) | 168h 统计漂移 | 状态 |",
            "| --- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for sym in ORDERED_SYMBOLS:
        fc = fc_by_sym.get(sym) or {}
        cn = fc.get("chronos_neural") or {}
        bs = fc.get("baselines") or {}
        last_c = fc.get("last_close") or packet["judgments"][sym].get("last_close")
        q50_24 = cn.get("q50_24h")
        if last_c and q50_24:
            pct_diff = ((float(q50_24) / float(last_c)) - 1.0) * 100.0
            c24_txt = f"`{_fmt_num(q50_24)}` ({pct_diff:+.2f}%)"
        else:
            c24_txt = f"`{_fmt_num(q50_24)}`"
        c6_txt = f"`{_fmt_num(cn.get('q50_6h'))}`" if cn.get("q50_6h") is not None else "—"
        st_txt = f"`{cn.get('status')}`" if cn.get("status") else "未运行"
        lines.append(
            f"| {SYMBOL_LABELS[sym]} | `{_fmt_num(last_c)}` | {c6_txt} | {c24_txt} | `{_fmt_num(bs.get('statistical_drift_168h_24h_q50'))}` | {st_txt} |"
        )

    ledger = packet.get("factor_usage_ledger") or {}
    used_cnt = sum(1 for entries in ledger.values() for e in entries if e.get("participated_in_judgment"))
    chg_cnt = sum(1 for entries in ledger.values() for e in entries if e.get("participation_status") == "changed_judgment")
    raw_missing = list((packet.get("audit_coverage") or {}).get("factors_missing") or [])
    proxied_labels: list[str] = []
    if "macro:ice_dxy" in raw_missing:
        proxied_labels.append("ICE DXY(用DX-Y.NYB代理)")
    if "macro:real_yield_intraday" in raw_missing:
        proxied_labels.append("盘中TIPS(用^TNX/TIP代理)")
    true_missing_ids = [m for m in raw_missing if m not in {"macro:ice_dxy", "macro:real_yield_intraday"}]
    true_missing_labels = _dedup_list([_short_factor_label(m) for m in true_missing_ids], 6)
    proxied_s = "、".join(proxied_labels) if proxied_labels else "无"
    missing_s = "、".join(true_missing_labels) if true_missing_labels else "无"
    lines.extend(
        [
            "",
            f"- **因子审计与缺口**：7 域共介入 `{used_cnt}` 项（改判驱动 `{chg_cnt}` 项），外部 AI=`{(ai_prov.get('llm_research_opinion') or {}).get('status')}` / 独立反方=`{(ai_prov.get('independent_ai_critic') or {}).get('status')}`｜已用代理替代(无付费直连)：{proxied_s}｜真缺失：{missing_s}（完整留痕见 JSON `factor_usage_ledger`）。",
            f"- **时点链路**：K线截止 `{_fmt_utc_short(lc.get('bar_cutoff_utc', packet['cutoff_utc']))}` → 最新公告 `{_fmt_utc_short(lc.get('latest_announcement_published_at_utc'))}` → 判断生成 `{_fmt_utc_short(lc.get('judgment_generated_at_utc'), include_sec=True)}` → 最早可成交 `{_fmt_utc_short(earliest_exec)}`。",
        ]
    )

    return {
        "intent": "overview",
        "cutoff_utc": packet["cutoff_utc"],
        "starting_cash_usdt": cash_ref,
        "lifecycle_timestamps": lc,
        "judgment_hash": packet["judgment_hash"],
        "counterfactual_simulation": False,
        "summary_table": table_rows,
        "judgments": packet["judgments"],
        "factor_usage_ledger": ledger,
        "audit_coverage": packet.get("audit_coverage"),
        "agent_or_human_supplement": packet.get("agent_or_human_supplement"),
        "markdown": "\n".join(lines),
    }


def _render_why_not_direction(question: str, packet: dict[str, Any]) -> dict[str, Any]:
    q = question.lower()
    targets = _extract_symbols(question)
    asked_dir = "short" if any(k in q for k in ("做空", "看空", "short", "bear")) else (
        "long" if any(k in q for k in ("做多", "看多", "long", "bull")) else "short_or_long"
    )
    dir_cn = "做空" if asked_dir == "short" else ("做多" if asked_dir == "long" else "单边开仓")

    details: dict[str, Any] = {}
    lines = [f"### 追问解答：为什么不{dir_cn}？（绑定快照 `{_fmt_utc_short(packet['cutoff_utc'])}` | `{packet['judgment_hash'][:12]}`）", ""]
    for sym in targets:
        j = packet["judgments"][sym]
        pos = packet["tide"]["instrument_positions"][sym]
        i_d, s_d = j["intraday"], j["swing_1_3d"]
        reasons: list[str] = []
        for h_label, hd in (("日内", i_d), ("1-3日", s_d)):
            if hd["hard_gates_triggered"]:
                reasons.append(f"{h_label}触发硬门槛：{', '.join(hd['hard_gates_triggered'])}")
            reasons.append(f"{h_label}综合为 **{_bias_badge(hd['final_label_cn'])}**（技术面：{hd['technical_only']['label_cn']}；原因：{_round_long_decimals(hd['why_changed_or_unchanged_vs_technical'])}）")
            if hd["strongest_counterevidence"]:
                reasons.append(f"{h_label}反证/阻力：{_humanize_evidence_item(hd['strongest_counterevidence'][0])}")
        reasons.append(f"标的多周期位置：{pos['structure_kind_cn']}（1D={pos['daily_dir']}, 4H={pos['h4_dir']}, 1H={pos['h1_dir']}）")
        reasons.append(f"v2.2.1 正式策略信号：`{j['official_v2_2_1_signal']}`（reason={j['official_v2_2_1_reason']}），纸面开仓条件：{'满足' if j['paper_eligibility']['eligible'] else '不满足'}")
        sb = j.get("setup_bridge") or {}
        if sb.get("summary_cn"):
            reasons.append(f"距离 v2.2.1 正式触发条件：{sb['summary_cn']}")
        details[sym] = {
            "intraday_bias": i_d["final_bias"],
            "swing_bias": s_d["final_bias"],
            "reasons": reasons,
            "flip_conditions": [_humanize_evidence_item(x) for x in (s_d["flip_conditions"] + s_d["invalidation_conditions"][:1])],
        }
        lines.append(f"#### {SYMBOL_LABELS[sym]}")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append(f"- **若要转向{dir_cn}所需条件**：{'；'.join(details[sym]['flip_conditions'])}")
        lines.append("")

    return {
        "intent": "why_not_direction",
        "asked_direction": asked_dir,
        "cutoff_utc": packet["cutoff_utc"],
        "judgment_hash": packet["judgment_hash"],
        "counterfactual_simulation": False,
        "details_by_symbol": details,
        "markdown": "\n".join(lines),
    }


def _simulate_what_if(question: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Perturb queried macro/volatility driver in memory and re-run deterministic synthesis."""
    cfg = load_judgment_config()
    q = question.lower()
    atoms_copy = deepcopy(packet["evidence_atoms"])
    macro_copy = deepcopy(packet["macro_summary"])
    tide_copy = deepcopy(packet["tide"])
    vix_copy = deepcopy((packet.get("sentiment_summary") or {}).get("vix") or {"value": None, "elevated_volatility_regime": False})

    scenario_desc = []
    macro_copy["real_yield_delta_5obs_bps"] = macro_copy.get("real_yield_delta_5obs_bps") or 0.0
    macro_copy["dollar_pct_5obs"] = macro_copy.get("dollar_pct_5obs") or 0.0

    dollar_up = any(k in q for k in ("美元上", "美元走强", "美元涨", "dxy上", "dollar up", "stronger dollar", "dollar rises"))
    dollar_down = any(k in q for k in ("美元下", "美元走弱", "美元跌", "dollar down", "weaker dollar"))
    rates_up = any(k in q for k in ("利率上", "利率走高", "利率升", "美债收益", "加息", "鹰派", "yield up", "rate hike"))
    rates_down = any(k in q for k in ("降息", "鸽派", "利率下", "利率走低", "利率回落", "利率降", "rate cut", "yield down"))
    vix_spike = "vix" in q

    mb = tide_copy["macro_backdrop"]
    if dollar_up or rates_up:
        if dollar_up:
            macro_copy["dollar_pct_5obs"] = 1.25
            mb["pillars"]["dollar"]["status"] = "tightening"
            scenario_desc.append("美元代理 5 观测假设升至 +1.25%（超出 +0.80% 紧缩阈值）")
        if rates_up or dollar_up:
            # Check tightening: if only dollar asked, simulate dollar tightening + check current real yield or stress tightened dollar+rates
            pass
        if rates_up:
            macro_copy["real_yield_delta_5obs_bps"] = 18.0
            mb["pillars"]["real_yield"]["status"] = "tightening"
            scenario_desc.append("10Y 实际利率假设上行 +18 bps（超出 +12 bps 阈值）")
        tight = sum(1 for k in ("dollar", "real_yield", "fed_net_liquidity") if mb["pillars"][k]["status"] == "tightening")
        if tight >= 2 or (dollar_up and mb["pillars"]["real_yield"]["status"] != "easing"):
            mb["state"] = "headwind"
            mb["state_cn"] = "[反事实推演] 美元与利率走紧构成宏观逆风"
            mb["backdrop_by_symbol"] = {s: "headwind" for s in SYMBOLS}
        # Also inject/update counterfactual dollar & rate atoms cleanly
        for a in atoms_copy:
            if a["evidence_id"] in {"macro:broad_dollar_proxy:DTWEXBGS", "macro:ice_dxy"} and dollar_up:
                a["freshness_status"] = "fresh" if a["evidence_id"] == "macro:ice_dxy" else "lagged_usable"
                a["direction_by_symbol"] = {s: "bearish" for s in SYMBOLS}
            if a["evidence_id"] == "macro:real_yield_daily:DFII10" and (rates_up or dollar_up):
                a["direction_by_symbol"] = {s: "bearish" if rates_up else a["direction_by_symbol"].get(s, "neutral") for s in SYMBOLS}
    elif dollar_down or rates_down:
        macro_copy["dollar_pct_5obs"] = -1.25 if dollar_down else macro_copy["dollar_pct_5obs"]
        macro_copy["real_yield_delta_5obs_bps"] = -18.0 if rates_down else macro_copy["real_yield_delta_5obs_bps"]
        if dollar_down:
            mb["pillars"]["dollar"]["status"] = "easing"
            scenario_desc.append("美元代理假设回落 -1.25%（宽松顺风阈值）")
        if rates_down:
            mb["pillars"]["real_yield"]["status"] = "easing"
            scenario_desc.append("实际利率假设下行 -18 bps（宽松顺风阈值）")
        mb["state"] = "tailwind"
        mb["state_cn"] = "[反事实推演] 美元/利率走弱转为宏观顺风"
        mb["backdrop_by_symbol"] = {s: "tailwind" for s in SYMBOLS}
        for a in atoms_copy:
            if a["evidence_id"] in {"macro:broad_dollar_proxy:DTWEXBGS", "macro:ice_dxy"} and dollar_down:
                a["freshness_status"] = "fresh" if a["evidence_id"] == "macro:ice_dxy" else "lagged_usable"
                a["direction_by_symbol"] = {s: "bullish" for s in SYMBOLS}
    elif vix_spike:
        vix_copy = {"status": "ok", "value": 28.5, "pct_change_1_obs": 22.0, "elevated_volatility_regime": True}
        mb["pillars"]["risk_appetite"]["status"] = "risk_off"
        scenario_desc.append("VIX 假设跃升至 28.5（+22%，进入高波动避险状态；注意 VIX 不单独触发纳指方向，仅降把握或配合黄金避险）")
    else:
        # Generic dollar-up sensitivity check by default if phrasing ambiguous
        macro_copy["dollar_pct_5obs"] = 1.25
        mb["pillars"]["dollar"]["status"] = "tightening"
        mb["state"] = "headwind"
        mb["state_cn"] = "[反事实推演] 美元走强条件假设"
        mb["backdrop_by_symbol"] = {s: "headwind" for s in SYMBOLS}
        scenario_desc.append("美元走强假设 (+1.25%)")

    sw = dict(packet["ablation_switches"])
    diffs = []
    lines = [
        f"### 反事实条件敏感性推演（依据快照 `{packet['cutoff_utc']}` / `{packet['judgment_hash'][:12]}`）",
        f"- **推演假设（非既定事实）**：{'；'.join(scenario_desc)}",
        "",
        "| 品种 | 期限 | 当前判断 | 假设情景下判断 | 是否改判 | 关键变化说明 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for sym in ORDERED_SYMBOLS:
        pos = tide_copy["instrument_positions"][sym]
        for h in HORIZONS:
            h_cn = "日内" if h == "intraday" else "1-3日"
            cur = packet["judgments"][sym][h]
            sim = _synthesize_one(
                sym, h, pos=pos, tide=tide_copy, macro_summary=macro_copy, vix_summary=vix_copy,
                atoms=atoms_copy, ablation=sw, cfg=cfg,
            )
            flipped = (cur["final_bias"] != sim["final_bias"]) or (cur["conviction_note"] != sim["conviction_note"])
            diffs.append(
                {
                    "symbol": sym,
                    "horizon": h,
                    "before_bias": cur["final_bias"],
                    "after_bias": sim["final_bias"],
                    "before_cn": cur["final_label_cn"],
                    "after_cn": sim["final_label_cn"],
                    "direction_changed": cur["final_bias"] != sim["final_bias"],
                    "conviction_changed": cur["conviction_note"] != sim["conviction_note"],
                    "reason": sim["why_changed_or_unchanged_vs_technical"],
                }
            )
            chg_txt = "是（方向改判）" if cur["final_bias"] != sim["final_bias"] else ("把握降级" if cur["conviction_note"] != sim["conviction_note"] else "不变")
            lines.append(f"| {SYMBOL_LABELS[sym]} | {h_cn} | {cur['final_label_cn']} | **{sim['final_label_cn']}** | {chg_txt} | {sim['why_changed_or_unchanged_vs_technical']} |")

    return {
        "intent": "what_if_sensitivity",
        "cutoff_utc": packet["cutoff_utc"],
        "judgment_hash": packet["judgment_hash"],
        "counterfactual_simulation": True,
        "scenario": scenario_desc,
        "diffs": diffs,
        "markdown": "\n".join(lines),
    }


def _render_diff_since_last(packet: dict[str, Any], previous_packet: dict[str, Any] | None) -> dict[str, Any]:
    if previous_packet is None:
        return {
            "intent": "diff_since_last",
            "cutoff_utc": packet["cutoff_utc"],
            "judgment_hash": packet["judgment_hash"],
            "counterfactual_simulation": False,
            "previous_cutoff_utc": None,
            "evidence_changes": [],
            "judgment_changes": [],
            "markdown": f"### 快照证据对比（当前 `{packet['cutoff_utc']}`）\n\n当前账本尚未找到更早一次的已存快照记录，因此无法进行跨周期对比；请积累下一小时/下一周期快照后再追问。",
        }

    prev_map = {a["evidence_id"]: a for a in previous_packet.get("evidence_atoms") or []}
    cur_map = {a["evidence_id"]: a for a in packet.get("evidence_atoms") or []}
    ev_changes = []
    for eid in sorted(set(prev_map) | set(cur_map)):
        pa, ca = prev_map.get(eid), cur_map.get(eid)
        if pa is None:
            ev_changes.append({"evidence_id": eid, "change_type": "added", "after": {"freshness": ca["freshness_status"], "value": ca["value_summary"]}})
        elif ca is None:
            ev_changes.append({"evidence_id": eid, "change_type": "removed", "before": {"freshness": pa["freshness_status"]}})
        elif pa["freshness_status"] != ca["freshness_status"] or pa["direction_by_symbol"] != ca["direction_by_symbol"] or pa["value_summary"] != ca["value_summary"]:
            ev_changes.append(
                {
                    "evidence_id": eid,
                    "factor_name": ca["factor_name"],
                    "change_type": "updated",
                    "before": {"freshness": pa["freshness_status"], "directions": pa["direction_by_symbol"], "value": pa["value_summary"]},
                    "after": {"freshness": ca["freshness_status"], "directions": ca["direction_by_symbol"], "value": ca["value_summary"]},
                }
            )

    j_changes = []
    for sym in ORDERED_SYMBOLS:
        for h in HORIZONS:
            pj = previous_packet["judgments"][sym][h]
            cj = packet["judgments"][sym][h]
            if pj["final_bias"] != cj["final_bias"] or pj["conviction_note"] != cj["conviction_note"]:
                j_changes.append(
                    {
                        "symbol": sym,
                        "horizon": h,
                        "before": pj["final_label_cn"],
                        "after": cj["final_label_cn"],
                        "before_conviction": pj["conviction_note"],
                        "after_conviction": cj["conviction_note"],
                    }
                )

    lines = [
        f"### 相比上次快照的证据与结论变化（`{previous_packet['cutoff_utc']}` → `{packet['cutoff_utc']}`）",
        "",
        f"- **发生变化的证据项（共 {len(ev_changes)} 项）**：",
    ]
    for ch in ev_changes[:12]:
        if ch["change_type"] == "updated":
            lines.append(f"  - `{ch['evidence_id']}` ({ch.get('factor_name', '')}): {ch['before']['freshness']} -> {ch['after']['freshness']} | value {ch['before']['value']} -> {ch['after']['value']}")
        else:
            lines.append(f"  - `{ch['evidence_id']}`: {ch['change_type']}")
    if not ev_changes:
        lines.append("  - 各项外部证据新鲜度、方向与数值摘要均无变化。")

    lines.append(f"- **结论变化（共 {len(j_changes)} 项）**：")
    for jc in j_changes:
        lines.append(f"  - {jc['symbol']} ({jc['horizon']}): {jc['before']} ({jc['before_conviction']}) → **{jc['after']}** ({jc['after_conviction']})")
    if not j_changes:
        lines.append("  - 三个品种在两个期限上的综合判断方向与把握级别未发生变化。")

    return {
        "intent": "diff_since_last",
        "cutoff_utc": packet["cutoff_utc"],
        "judgment_hash": packet["judgment_hash"],
        "previous_cutoff_utc": previous_packet["cutoff_utc"],
        "previous_hash": previous_packet["judgment_hash"],
        "counterfactual_simulation": False,
        "evidence_changes": ev_changes,
        "judgment_changes": j_changes,
        "markdown": "\n".join(lines),
    }


def answer_judgment_query(
    question: str,
    *,
    packet: dict[str, Any],
    previous_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer overview or traceable follow-up questions strictly bound to a captured JudgmentPacket."""
    intent = classify_chat_intent(question)
    if intent == "diff_since_last":
        out = _render_diff_since_last(packet, previous_packet)
    elif intent == "what_if_sensitivity":
        out = _simulate_what_if(question, packet)
    elif intent == "why_not_direction":
        out = _render_why_not_direction(question, packet)
    else:
        out = _render_overview(packet)
    out["question"] = question
    out["order_permissions"] = False
    return out
