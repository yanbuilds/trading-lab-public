# PIIA v2.2 前瞻验证冻结规范（账本隔离 + 三合约对齐 + 连续小时计数）

> **状态：待验证（`PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`）**。v2.2 只计入本冻结生效后、由当前冻结代码在真实账本上按时记录的真实公开数据观测。历史回放、演示、调用者注入样本以及 v2.0 / v2.1 账本中的任何记录**都不计入**。

## 1. 版本与证据边界

| 项目 | 内容 |
| --- | --- |
| 审查基准提交 | `7e4cc510e2ea42b9691aa4360b17f87599007271`（PR #1 上的 v2.1 文档提交） |
| 规则版本 | `PIIA_PERSONAL_INVESTMENT_INTELLIGENCE_AGENT_V2`，`version = "2.2.0"` |
| 冻结代码提交 | `3b975fd72f984ead0a5a71a702b99bce2b60e793`（`feat(piia-v2.2)`） |
| 配置文件 LF SHA-256 | `008155724c4df21f1777897450d190630a623ca1a61718d237fe69720131f5d9`（[config/piia_agent_v2.yaml](config/piia_agent_v2.yaml)） |
| 受控 7 文件清单 `manifest_sha256` | `65c7a66b3f6a99b584b1ffddcae025caa2856d79148dab7f82dede63ad4ea49f` |
| 前瞻起算时点 | `2026-11-23T08:00:00+00:00`（`prospective_freeze_started_utc`，晚于冻结提交推送时间） |
| 策略/成本/风险参数 | 与 v2.0.0 / v2.1.0 **完全相同**：2x 逐仓、单笔 1% 止损预算、20% 现金底线、50% 最大保证金、Taker 0.06%、滑点 0.05%/边（另并行 2x 滑点）、3% 单日 / 10% 总回撤停机 |
| 受控文件 | `PIIA_V2_PROTOCOL.md`、`config/piia_agent_v2.yaml`、`src/piia/intelligence/agent_v2/{pillars,roles,risk_gate,ablation,continuous_paper}.py`（`python -m piia agent-verify-freeze` 校验） |

旧版本证据边界保持不变：
- v2.0.0（`a3c42e3`）`2026-11-16..11-22` 结果仍为「历史回放，事前冻结未获独立验证」；
- v2.1.0（`7a2b518`）冻结期真实前瞻周期数为 0，已被本版本取代；
- 旧账本 `data/piia/piia_v2_continuous.sqlite`（v2.0，无证据类型标签）与 `data/piia/piia_v2_1_continuous.sqlite`（若存在）**原样保留、只读审计、永不写入、永不计数**。

## 2. 本版修复（不改变任何策略阈值）

1. **账本按证据类型隔离**：真实采集只写 `data/piia/piia_v2_2_live_prospective.sqlite`（标签 `live_prospective`），演示只写 `data/piia/piia_v2_2_demo_synthetic.sqlite`（标签 `demo_synthetic`），调用者注入样本只能写入显式指定的 `test_or_injected` 账本。标签写入不可修改的 `ledger_meta` 表，底层写入函数按标签拒绝跨类型写入（不依赖界面传参）；无标签的旧账本一律拒绝写入。真实账本上的人工审批拒绝注入报价、K 线、资金费或时钟。
2. **三合约收盘对齐门**：真实采集要求 BTCUSDT、XAUUSDT、NDX100USDT 最新完整 1h K 线收盘时间一致，且等于采集时刻所在小时的整点（即刚收盘的小时）；缺失、断档、滞后、未来 K 线或采集跨小时均返回明确原因，**当次不打开账本、不写周期、不生成提案、不推进任何影子账本的仓位、资金费或 K 线游标**。通过后按共同截止时点显式选取 `cutoff-1h` 信号 K 线。
3. **前瞻计数口径**：只计入同时满足以下条件的观测 —— 真实模式；账本标签为 `live_prospective` 且未检出混合数据；`manifest_sha256` 等于当前冻结清单；`policy_version` 等于当前版本；截止时点不早于前瞻起算时点且为整点；采集开始与写入时间都落在该截止小时内；同一截止小时只计一次。

## 3. 评估规则（预注册）

- 状态页分别报告：计入的真实观测数、最长连续小时数（及起止）、观测区间内缺失小时数、自起算以来缺失小时数、各类排除原因计数。
- **样本是否足够只看最长连续窗口**：最长连续真实观测 `>= 168` 小时前一律为 `PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`（待验证）；零散记录总数再多也不能替代连续窗口。
- 达到 168 连续小时后状态仅变为 `PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED`（待审计），仍须人工审计资金费完整性、停机、对照账本后才能讨论结论；淘汰/证伪标准沿用 v2.1：若 `M4_full_piia_v2` 在 1x 或 2x 滑点下不能整体优于 `B0_cash` / `B1` / `B2` / `B3`，如实记录为尚未证明净决策增益，不得回调参数。

## 4. 旧账本处理方式

- 运行 `python -m piia agent-inspect-ledger` 可只读审计真实、演示与旧账本：报告标签、周期模式分布、`init_book` 模式分布、是否混合、是否可用作真实前瞻账本。
- 混合或无标签账本：**保留原文件不动**（测试校验字节级不变），不迁移、不在统计中“剔除演示行后继续使用”，因为模拟余额、仓位、停机与游标可能已被演示/注入周期影响。v2.2 从全新的真实账本重新开始。

## 5. 登记

- 冻结代码提交 `3b975fd72f984ead0a5a71a702b99bce2b60e793`；配置 SHA-256 `008155724c4df21f1777897450d190630a623ca1a61718d237fe69720131f5d9`；清单 `65c7a66b3f6a99b584b1ffddcae025caa2856d79148dab7f82dede63ad4ea49f`。哈希只证明仓内文件一致，不是第三方可信时间戳。
- 登记时（`2026-11-23` 约 06:00 UTC）真实前瞻账本 `data/piia/piia_v2_2_live_prospective.sqlite` 尚不存在：**计入观测 0 小时，最长连续 0 小时，状态待验证**。
- 本地旧账本 `data/piia/piia_v2_continuous.sqlite`：无标签（v2.0），1 个真实周期（v2.0 清单 `17b31673…`），未检出演示行，但因无标签且清单不同，按规则只读保留、不计数。
