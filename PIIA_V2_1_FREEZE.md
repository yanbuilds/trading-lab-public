# PIIA v2.1 前瞻验证冻结规范（审计修复后重置前瞻起算点）

> **状态**：**前瞻验证待完成（PENDING）**。自本冻结版本提交并推送后，方可按整点关闭 K 线开启新的前瞻纸面记录；在此之前的任何历史回放（含 `2026-11-16` 至 `2026-11-22`）均显式标注为历史回放对照，不宣称为经独立事前时间戳公证的样本外表现。

---

## 1. 冻结版本标识与不可变策略承诺

- **审查基准提交**：`a3c42e3ffbca76f12cfad86a51e9fe3bbb6cc1e0`（`a3c42e3`）
- **当前冻结代码提交（Commit 1）**：`7a2b518eb310769948419669d577079d391d7e67`（`7a2b518`）
- **当前规则版本**：`PIIA_PERSONAL_INVESTMENT_INTELLIGENCE_AGENT_V2` (`version = "2.1.0"`)
- **配置文件**：[config/piia_agent_v2.yaml](config/piia_agent_v2.yaml)（LF SHA-256 = `967ac9efa76bc74be0865fc2f68857557f3b0e814e6ce928c6401357a438e25f`；实际交易/风控参数与 `a3c42e3` 相比 **零数值调整**，绝不为提高回测收益调参）
- **受控 7 文件清单哈希（Manifest）**：`d9032a8d7ca6690f72f971da9de3475acf35ee79043f52287255e5b3e5da3d80`
- **冻结提交哈希与配置哈希验证方式**：
  - 在工作区运行 `python -m piia agent-verify-freeze` 可即时核验全部 7 个受控核心文件与 [config/piia_agent_v2.yaml](config/piia_agent_v2.yaml) 的 LF SHA-256，但**同分支提交中的哈希仅证明文件一致，不构成第三方独立可信时间戳**。
  - **重要边界**：原 `v2.0.0`（`a3c42e3`）的 `2026-11-16 06:00 UTC` 至 `2026-11-22 23:00 UTC` 回放因当时同提交写入协议与报告，现统一标为**“历史回放，事前冻结未获独立验证”**；本 `v2.1.0` 冻结于 `7a2b518` 提交并推送之后方开启新的持续前瞻纸面记录，目前前瞻观察期处于 **待完成（`PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`）** 状态。

---

## 2. 成本、资金费、风控与停机冻结口径

| 维度 | 冻结口径（与回放、持续纸面完全一致） |
| --- | --- |
| 覆盖标的与合约模式 | Bitget `USDT-FUTURES`：`BTCUSDT`、`XAUUSDT`、`NDX100USDT`，逐仓 `2.0x` 杠杆，全球同刻最多 `1` 仓 |
| 止损风险与保证金约束 | 单笔最大止损预算 `1.0%` 权益（含进出场 Taker 手续费 + 滑点），现金底线 `20%`，最大保证金占比 `50%`，强平价格缓冲 `>= 1.5%` |
| 手续费（Fallback）与滑点 | Taker `0.06%` / 边；基础滑点 `0.05%` / 边（`1x`），并行对照 `2x` 双倍滑点压力情景（`0.10%` / 边） |
| 资金费结算规则 | 仅记入通过 Bitget 公开 `history-fund-rate` 与 `current-fund-rate` 周期核验的已结算时点，按 `多头付正/收负、空头收正/付负 × 张数 × contract_size × 结算小时开盘价代理` 计入现金；**任一结算时点缺失或周期未核验时写为 `funding_unknown`，严禁补零，且自动封锁该笔交易与账本的净收益宣称（`net_return_claim_allowed = false`）** |
| 停止条件（跨重启持久化） | 触发任一即永久锁定停机并持久化至 `ledger_events (event_kind='loss_halt')`：① UTC 当日盯市权益较当日开盘净值回撤 `>= 3.0%`；② 盯市权益较初始本金回撤 `>= 10.0%`。停机后重启仍阻断一切新开仓（Shadow 与人工账本同效） |
| 人工审批再校验 | 审批前重验：15 分钟有效期、实时或 `<5 分钟` 新鲜报价、不利滑移未超 `max(0.5*ATR14, 0.5%)` 且未破初始止损、无已有持仓、按最新余额重过确定性风控、未停机；任一失效即拒绝并要求新周期提案 |
| 执行权限 | `live_order_permissions = false`、`exchange_demo_order_permissions = false`，永不保存或读取交易所账户密钥 |

---

## 3. 前瞻对比基准与模块消融口径

- **持续前瞻纸面同步对照账本（含 `1x` 与 `2x` 滑点）**：
  1. `B0_cash`（零敞口现金）
  2. `B1_simple_ema_1h`（1 小时 EMA20/50 交叉）
  3. `B2_aligned_ema_4h`（1h+4h 对齐 EMA）
  4. `B3_donchian_breakout_4h`（48h 唐奇安突破 + 4h 趋势 + 吊灯跟踪）
  5. `M4_full_piia_v2`（完整 PIIA v2：确定性研究规则 + 成本/波动/资金费反方门控 + 滞后宏观 + 168h 统计漂移 + 可选本地 Chronos）
  6. `human_approved_paper`（仅当人类主动在 CLI/API/UI 审批且通过实时再校验后开仓）
- **模块真实执行诚实区分**：
  - 确定性研究规则、确定性反方成本门控、滞后 ALFRED 宏观过滤、168h 统计漂移均会在 Trace `module_execution` 字段中逐一标注；
  - `Chronos` 神经网络仅在本地存在且权重通过哈希校验时标注 `executed`，否则明确标注 `not_executed_weights_absent`；
  - `official_llm_review` 始终显式标注为 `NOT_WIRED`（不得把同模型多角色调用或确定性规则写成独立大模型审查）。

---

## 4. 前瞻评估与停止判定标准（预注册）

- **最小观察窗口**：累计 **至少 168 小时连续/近似连续（>= 7 个完整自然日）** 冻结后真实采集周期，或至少 **30 笔完整平仓样本** 前，一律保持 `validation_status = PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`（**数据还不足以判断**）。
- **证伪与淘汰标准**：若积累完整资金费覆盖的足够前瞻样本后，`M4_full_piia_v2` 在 `1x` 或 `2x` 滑点下仍不能在净收益/最大回撤/卡玛比维度整体优于 `B0_cash`、`B1_simple_ema_1h`、`B2_aligned_ema_4h`、`B3_donchian_breakout_4h`（含高换手磨损或过度拒单踏空），应如实记录为**系统尚未证明净决策增益**，不得就当前账本参数回调拟合。
