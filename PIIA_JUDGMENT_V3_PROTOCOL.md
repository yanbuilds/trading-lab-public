# PIIA Judgment v3 — 可追溯多因子跨周期交易判断、对话工作台与对比验证协议

> **状态：研究与影子纸面验证模块（不改冻结基线，不自动下单）**。本协议以 `main` 上已合并的 PIIA v2.2.1（`74ce2eeb0c9ae759001030272fec91ec3992a415`）为对照组，完整保留 `v2.2.1` 与 [FROZEN_POLICY_V1.md](FROZEN_POLICY_V1.md) 的源码、配置和账本不受任何修改。Judgment v3 使用独立配置文件 [config/piia_judgment_v3.yaml](config/piia_judgment_v3.yaml) 与独立影子账本 `data/piia/piia_judgment_v3_shadow.sqlite`（演示用 `data/piia/piia_judgment_v3_demo.sqlite`）。

## 1. 回答粒度与状态边界

针对 **黄金 (`XAUUSDT`)**、**比特币 (`BTCUSDT`)** 与 **纳指 100 (`NDX100USDT`)** 三个 Bitget USDT 永续合约，系统按快照统一输出：
1. **双期限综合判断**：每个品种分别给出 **日内（4–12h）** 与 **未来 1–3 个交易日（24–72h）** 的研究偏向（`偏多 bullish_bias` / `偏空 bearish_bias` / `观望 neutral_stand_aside`），并明示 **核心依据、最强反证、价格与宏观失效条件、以及数据缺口**。
2. **三栏明确区分**：同一张输出表清晰区分：
   - **纯技术面对照方向（`technical_only`）** 与 **综合研究判断**（说明哪些外部因素改变或确认了技术面判断）；
   - **`v2.2.1` 正式冻结策略信号**（原冻结 1h 机械+风控规则输出，与 v3 综合判断并行展示，不互相篡改）；
   - **当前是否满足纸面开仓条件**（由 `ProgrammaticRiskGate` 按 `500 USDT`、`2x` 逐仓、`1%` 单笔预算与成本/停机约束程序核定，绝不自动实盘）。

## 2. 五大外部证据领域与时间/级别纪律

所有证据均以标准化 `EvidenceAtom` 存储，强制记录 `source_name`、`source_url`、`raw_sha256`、`observed_at_utc`、`published_at_utc`、`available_at_utc`、`retrieved_at_utc`、`data_tier`、`freshness_status`、`causal_driver_group` 与本次作用 `role_in_judgment`（`changed_direction` / `reduced_conviction` / `triggered_veto` / `confirmed_direction` / `context_only` / `excluded_stale_or_future` / `missing_gap`）。

- **严禁时间混用与前视**：凡 `observed_at_utc > cutoff_utc` 或 `published_at_utc > cutoff_utc` 的记录一律标记 `future_excluded`；超出各自新鲜度上限标记 `stale_excluded`；三合约 1h 最新闭合收盘时间不一致时拒绝混合评估。
- **数据分级（不以滞后/代理序列冒充实时原值）**：
  - `realtime_verified`：Bitget 1h/4h/1d 闭合 K 线与 5 分钟内报价/盘口/费率/OI、已核验 `mempool.space` 链上块高与拥堵费、Polymarket 5 分钟内双边盘口。
  - `intraday_delayed_public`：带明确时间标记的公开延迟期货/指数序列（如 `DX.F` 若可取得）。
  - `proxy_series`：ALFRED 美联储广义贸易加权美元指数 `DTWEXBGS`（**绝不冒充 ICE DXY**；未取得可核验 ICE DXY 时单独显式标注 `ICE DXY: 缺失/unavailable`）。
  - `latest_daily_vintage`：ALFRED 日频保守版本（`DFII10` 10Y 实际利率、`DGS10`/`DGS2` 名义利率、`DFF`、净流动性代理、`VIXCLS`、`NASDAQ100`/`SP500`）——**日频实际利率绝不作为盘中实时值**。
  - `lagged_official_report`：CFTC COMEX 黄金期货周度 COT（滞后期货持仓，**不代表黄金 ETF 份额申赎或实物 demand**）、US Spot BTC ETF 完整披露日净流（**横线 `-` 或缺报基金保持 `partial_unreported_not_zero` / 未知，绝不填零**）。
  - `scheduled_only`：Fed / BLS / BEA 官方日程；明确区分 **官方日程（`schedule`）**、**市场预期（`consensus`，无公开数据源时保持不可用）** 与 **已核验实际公布（`actual`，按截止前公布时间戳匹配）**。

## 3. “潮汐”三层可复算定义

1. **大环境（`macro_backdrop`）**：综合美元、10Y 实际利率、美联储净流动性（`WALCL - WDTGAL - RRP`）及全市场风险偏好（`VIXCLS` / 跨市场表现）。至少需要 2 项宏观/流动性支柱有效，**绝不从单条均线推断市场潮汐**，缺口过多则报 `unknown`。同时逐品种输出顺逆风映射，解释同一宏观变化对黄金与纳指 100 的传导差异。
2. **当前阶段（`cycle_stage`）**：按联合条件判定 `expansion`（扩张）、`ebbing`（退潮）、`deleveraging`（去杠杆）、`repair`（修复）或 `insufficient_evidence`（证据不足，关联既有 `liquidity_positioning` 规则）。
3. **标的位置（`instrument_positions`）**：结合已收盘 `1d UTC`（`EMA20/50/SMA200`）、`4h`（`EMA20/50`、`Donchian48h`）与 `1h`（`EMA20/50`、`ADX14`、`RSI14`），显式区分顺势（`aligned_uptrend` / `aligned_downtrend`）、趋势回踩（`pullback_in_uptrend`）、逆势反弹（`bounce_in_downtrend`）、趋势反转候选（`potential_*_reversal`）与区间震荡（`range_consolidation`）。

## 4. 新闻核验、情绪规则与模型边界

- **新闻核验**：仅采信带合法 URL 与带时区发布时间的允许来源；无 URL 或无发布时间的未核实消息标为 `unverified_unknown`（仅披露、零权重）。严格区分 **原始消息（`primary_release`）**、**后续解读（`secondary_interpretation`）** 与 **重复转载（`duplicate_repost`，按归一化文本 Jaccard >= 0.70 去重并归零权重）**，逐条绑定影响标的、方向偏向与价格传导渠道。
- **VIX 与 Polymarket 纪律**：`VIX` 仅作股票波动率背景（`max_directional_vote = 0`），**绝不单独决定纳指方向**；Polymarket 仅筛选直接相关且点差 `<= 0.05`、盘口 5 分钟内有效的市场，标明其为市场定价而非既定事实。
- **BTC 链上区分**：`mempool.space` 仅反映网络活跃度/拥堵费率；公开免费源缺失交易所地址级净流时，**`onchain_exchange_netflow` 明确标为 `unavailable`**，绝不以 mempool 数据冒充交易所净流入/流出。
- **模型预测与 AI 诚实性**：将实际运行的 `Chronos-t5-tiny`（需与当前 `cutoff_utc` 一致且在对应小时内运行）与因果简单基准（零变化、`24h` 动量、168h 统计漂移）并列；Chronos 未运行、超时或截止时间落后时明确标为 `not_run` / `expired_excluded`，**不借用较早一次预测**。同步记录外部大模型研究意见与独立反方审查是否实际运行，**绝不将确定性规则伪装成独立 AI 审查**。

## 5. 因果去重与冲突仲裁规则

- 多条证据归属同一 `causal_driver_group`（例如 `DFII10`、`DGS10` 与 `DTWEXBGS` 同属 `usd_real_rate_tightening`）时只计为 1 个独立驱动因子，避免同一原因被重复加权。
- 仲裁优先级：**硬门槛（数据时效/完整性、日内极端波动、距重大一级发布 <= 2h、去杠杆做多限制、同向极强一手重磅新闻冲突） > 潮汐三层结构 + 宏观驱动 > 经核验一级新闻与完整机构流（ETF） > 经核验当前截止的模型预测 > 滞后情绪/COT/Polymarket 背景**。

## 6. AI 对照与五域消融检验协议

- **四臂同快照对比**：`A0_technical_only`（纯技术面）、`A1_direct_ai_candles_only`（仅相同前缀闭合行情+任务说明，不给外部数据包与未来数据）、`A2_ai_freeform_same_packet`（与规则引擎相同 `EvidenceAtom` 数据包但由 AI 自由综合）、`A3_project_full_judgment_v3`（完整规则综合），以此区分“信息覆盖”与“规则合成”的各自贡献。未配置官方模型 API 时，AI 臂如实记为 `NOT_RUN_NEEDS_MODEL_CONFIG`。
- **五域逐一剔除消融**：`Full`、`No_Macro`、`No_Tide`、`No_Sentiment_Flow`、`No_News`、`No_Forecast`、`Technical_Only`，统计方向分布、覆盖率、观望率、决策改变明细，以及在成熟窗口后的命中表现与影子纸面账户（`500 USDT`、`1x` 与 `2x` 滑点、严格资金费审计）表现；在连续 168 小时前瞻积累与结果成熟前，验证状态保持 **待验证**。
