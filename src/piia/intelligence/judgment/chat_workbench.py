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
        body = re.sub(r"\s*->\s*[A-Z0-9]+\s+tailwind$", "（对该标的构成宏观顺风）", body)
        body = re.sub(r"\s*->\s*[A-Z0-9]+\s+headwind$", "（对该标的构成宏观逆风）", body)
        return body
    if s.startswith("[sentiment:"):
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


def _render_overview(packet: dict[str, Any]) -> dict[str, Any]:
    table_rows = build_summary_table_rows(packet)
    lc = packet.get("lifecycle_timestamps") or {}
    cash_ref = packet.get("starting_cash_usdt")
    tide = packet["tide"]
    macro = packet["macro_summary"]
    sent_sum = packet.get("sentiment_summary") or {}
    vix = (sent_sum.get("vix") or {})

    lines = [
        f"### 三合约多维研判报告（{_fmt_bj_short(packet['cutoff_utc'])} / {_fmt_utc_short(packet['cutoff_utc'])}）",
        "",
        f"> **研究声明**：快照 `{packet['judgment_hash'][:12]}` | 影子参考资金 `{cash_ref}` USDT | 无自动/实盘下单权限 | 冻结 `v2.2.1` 正式信号分栏展示。",
        "",
        "#### 一句话结论速览",
    ]
    for sym in ORDERED_SYMBOLS:
        j = packet["judgments"][sym]
        i_d, s_d = j["intraday"], j["swing_1_3d"]
        pe = j["paper_eligibility"]
        pe_note = f"⚡ v2.2.1 触发 `{j['official_v2_2_1_signal']}` ({j.get('official_v2_2_1_setup_type')})，纸面开仓={pe.get('label_cn')}" if j["official_v2_2_1_signal"] in {"long", "short"} else "v2.2.1 暂观望"
        chg_note = "（**相对技术面已改判**）" if (i_d["changed_vs_technical"] or s_d["changed_vs_technical"]) else ""
        lines.append(
            f"- **{SYMBOL_LABELS[sym]}**（现价 `{_fmt_num(j.get('last_close'))}`）：日内 **{_bias_badge(i_d['final_label_cn'])}** / 1–3 日 **{_bias_badge(s_d['final_label_cn'])}**{chg_note} — {pe_note}"
        )

    lines.extend(
        [
            "",
            "#### 核心判断总览",
            "| 品种 | 最新收盘 | 纯技术面 (日内 / 1-3日) | 综合：日内 | 综合：1-3日 | 相对技术面 | `v2.2.1` 信号 | 纸面开仓 | 首要缺口 |",
            "| --- | ---: | :---: | :---: | :---: | --- | :---: | :---: | --- |",
        ]
    )
    for r in table_rows:
        lines.append(
            f"| {r['label']} | `{_fmt_num(r['last_close'])}` | {r['technical_only']} | **{_bias_badge(r['intraday'])}** | **{_bias_badge(r['swing_1_3d'])}** | {r['changed_vs_technical']} | `{r['official_v2_2_1_signal']}` | {r['paper_eligible']} | {_humanize_gap(r['top_data_gap'])} |"
        )

    ice_dxy_txt = "实时直连可用" if macro.get("ice_dxy_available") else "缺失（未接入付费源）"
    dx_y_txt = (
        f"`{_fmt_num(macro.get('intraday_dollar_dx_y_nyb'))}` (24h `{_fmt_num(macro.get('intraday_dollar_dx_y_nyb_pct_24h'), 2, '%')}`)"
        if macro.get("intraday_dollar_dx_y_nyb") is not None
        else "缺失"
    )
    tnx_txt = (
        f"`{_fmt_num(macro.get('intraday_tnx_pct'), 3, '%')}` (24h `{_fmt_num(macro.get('intraday_tnx_delta_24h_bps'), 1, ' bps')}`)"
        if macro.get("intraday_tnx_pct") is not None
        else "缺失"
    )
    dfii_txt = (
        f"`{_fmt_num(macro.get('real_yield_dfii10_pct'), 2, '%')}` (5观测 `{_fmt_num(macro.get('real_yield_delta_5obs_bps'), 1, ' bps')}`)"
        if macro.get("real_yield_dfii10_pct") is not None
        else "缺失"
    )
    lines.extend(
        [
            "",
            "#### 宏观大环境与市场潮汐",
            f"- **大环境判定**：{tide['macro_backdrop']['state_cn']}",
            f"- **市场周期阶段**：{tide['cycle_stage']['stage_cn']}（依据：`{tide['cycle_stage']['definition_used']}`）",
            "",
            "| 宏观因子 | 实时直连 | 公开延迟 / 日频代理 | 最新读数与变动 |",
            "| --- | --- | --- | --- |",
            f"| 美元强弱 | ICE DXY：{ice_dxy_txt} | Yahoo `DX-Y.NYB` 1H / ALFRED `DTWEXBGS` | {dx_y_txt} \\| 广义美元 `{_fmt_num(macro.get('broad_dollar_dtwexbgs'))}` (5观测 `{_fmt_num(macro.get('dollar_pct_5obs'), 2, '%')}`) |",
            f"| 美国利率 | 盘中实时 TIPS：缺失 | Yahoo `^TNX` 1H / ALFRED `DFII10` | `^TNX` {tnx_txt} \\| 10Y实际利率 {dfii_txt} |",
            f"| 利率与期限结构 | — | ALFRED `DGS10` / `DGS2` / `DFF` | 10Y `{_fmt_num(macro.get('dgs10_pct'), 2, '%')}` · 2Y `{_fmt_num(macro.get('dgs2_pct'), 2, '%')}` · 联邦基金 `{_fmt_num(macro.get('dff_pct'), 2, '%')}` |",
            f"| 波动率背景 | — | ALFRED / CBOE `VIX` ({vix.get('observation_date') or '未知'}) | `{_fmt_num(vix.get('value'))}`（{'高波动风险区' if vix.get('elevated_volatility_regime') else '常态区间'}，仅限风险降权） |",
            "",
            f"- **黄金 vs 纳指传导机制**：{macro['gold_vs_ndx_divergence_note']}",
            "",
            "#### 分品种深度明细",
        ]
    )

    for sym in ORDERED_SYMBOLS:
        j = packet["judgments"][sym]
        pos = tide["instrument_positions"][sym]
        i_d, s_d = j["intraday"], j["swing_1_3d"]
        chg_evs = j.get("evidence_that_changed_conclusion") or []
        chg_lines = _dedup_list([f"[{c.get('horizon')}] {_humanize_evidence_item(c.get('detail'))}" for c in chg_evs], 4)
        sup_lines = _dedup_list(list(i_d["supporting_evidence"]) + list(s_d["supporting_evidence"]), 4, humanize=True)
        ctr_lines = _dedup_list(list(i_d["strongest_counterevidence"]) + list(s_d["strongest_counterevidence"]), 3, humanize=True)
        inv_lines = _dedup_list(list(i_d["invalidation_conditions"]) + list(s_d["invalidation_conditions"]), 3, humanize=True)
        gaps_list = [_humanize_gap(g) for g in sorted(set(i_d["data_gaps"] + s_d["data_gaps"]))]
        supp = j.get("agent_or_human_supplement") or packet.get("agent_or_human_supplement") or {}
        supp_txt = f"{supp.get('actor')}: {supp.get('note')} ({_fmt_utc_short(supp.get('recorded_at_utc'))}, 不可绕过风控)" if supp.get("provided") else "无"
        sb = j.get("setup_bridge") or {}
        va = j.get("venue_microstructure_audit") or {}

        lines.extend(
            [
                "",
                f"##### {SYMBOL_LABELS[sym]}｜结构：`{pos['structure_kind_cn']}`",
                f"- **日内（{_bias_badge(i_d['final_label_cn'])}）**：纯技术 `{i_d['technical_only']['label_cn']}` —— {_round_long_decimals(_ISO_TS_RE.sub(lambda tm: _fmt_utc_short(tm.group(0)), i_d['why_changed_or_unchanged_vs_technical']))}",
                f"- **1–3 日（{_bias_badge(s_d['final_label_cn'])}）**：纯技术 `{s_d['technical_only']['label_cn']}` —— {_round_long_decimals(_ISO_TS_RE.sub(lambda tm: _fmt_utc_short(tm.group(0)), s_d['why_changed_or_unchanged_vs_technical']))}",
                f"- **改判证据**：{('；'.join(chg_lines)) if chg_lines else '无外部因子推翻纯技术面'}",
                "- **关键依据**：",
                *(f"  - {x}" for x in (sup_lines or ["无"])),
                "- **最强反证与风险提示**：",
                *(f"  - {x}" for x in (ctr_lines or ["无"])),
                "- **失效与改判条件**：",
                *(f"  - {x}" for x in (inv_lines or ["无"])),
                f"- **研究→开仓等待条件 (`v2.2.1`)**：{sb.get('summary_cn') or '无'}",
                f"- **交易所微观结构盘口/资金费率**：{va.get('summary_cn') or '未采集'}",
                f"- **数据缺口**：{('、'.join(gaps_list)) if gaps_list else '无关键缺失'}",
                f"- **人工/Agent 补充判断**：{supp_txt}",
            ]
        )

    fc_by_sym = (packet.get("forecast_and_models") or {}).get("by_symbol") or {}
    ai_prov = (packet.get("forecast_and_models") or {}).get("ai_provenance") or {}
    lines.extend(
        [
            "",
            "#### 模型预测对照（非校准交易概率）",
            f"- **外部 AI 状态**：大模型研究=`{(ai_prov.get('llm_research_opinion') or {}).get('status')}`｜独立 AI 反方=`{(ai_prov.get('independent_ai_critic') or {}).get('status')}`（内置反方门控为确定性规则）",
            "",
            "| 品种 | 最新收盘 | Chronos 6h 中位 | Chronos 24h 中位 (对比现价) | 168h 统计漂移 (24h) | 模型运行状态 |",
            "| --- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for sym in ORDERED_SYMBOLS:
        fc = fc_by_sym.get(sym) or {}
        cn = fc.get("chronos_neural") or {}
        bs = fc.get("baselines") or {}
        last_c = fc.get("last_close") or (packet["judgments"][sym].get("last_close"))
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
    if ledger:
        lines.extend(
            [
                "",
                "#### 七大因子域审计摘要（API 成功 ≠ 参与判断；完整溯源见 JSON `factor_usage_ledger`）",
                "| 因子域 | 触发改判/否决 | 有效背景参考 | 过期 / 缺失 / 未运行 |",
                "| --- | --- | --- | --- |",
            ]
        )
        cat_labels = {
            "macro": "宏观",
            "tide": "潮汐",
            "sentiment_positioning": "情绪与仓位",
            "etf_and_onchain": "ETF 与链上",
            "polymarket": "Polymarket",
            "news": "新闻与 IR",
            "forecast_models": "预测模型与 AI",
        }
        for ckey, clabel in cat_labels.items():
            entries = ledger.get(ckey) or []
            chg = [e["factor_id"].split(":")[-1] for e in entries if e["participation_status"] == "changed_judgment"]
            bg = [e["factor_id"].split(":")[-1] for e in entries if e["participation_status"] == "background_only"]
            off = [
                f"{e['factor_id'].split(':')[-1]}({e['participation_status_cn']})"
                for e in entries
                if e["participation_status"] in {"expired", "missing", "not_run"}
            ]
            chg_s = ", ".join(chg[:3]) + (f" (+{len(chg) - 3})" if len(chg) > 3 else "") if chg else "—"
            bg_s = ", ".join(bg[:4]) + (f" (+{len(bg) - 4})" if len(bg) > 4 else "") if bg else "—"
            off_s = ", ".join(off[:3]) + (f" 等共 {len(off)} 项" if len(off) > 3 else "") if off else "—"
            lines.append(f"| **{clabel}** | {chg_s} | {bg_s} | {off_s} |")

    lines.extend(
        [
            "",
            f"> **完整时点链路**：K线截止 `{_fmt_utc_short(lc.get('bar_cutoff_utc', packet['cutoff_utc']))}`｜最新公告 `{_fmt_utc_short(lc.get('latest_announcement_published_at_utc'))}`｜首次抓取 `{_fmt_utc_short(lc.get('first_retrieved_at_utc'), include_sec=True)}`｜判断生成 `{_fmt_utc_short(lc.get('judgment_generated_at_utc'), include_sec=True)}`｜最早可成交 `{_fmt_utc_short(lc.get('earliest_executable_at_utc'))}`",
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
