# PIIA v2.2.1 前瞻验证冻结规范（账本隔离 + 三合约对齐 + 冻结前阻断 + 连续小时计数）

> **状态：待验证（`PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`）**。v2.2.1 只计入本冻结起算点（`2026-11-23T09:00:00+00:00`）之后、由当前冻结代码在干净真实账本上按时记录的真实公开数据观测。任何早于起算点的周期或已含冻结前记录的账本、历史回放、演示、调用者注入样本及 v2.0 / v2.1 账本**一律拒绝写入后续前瞻并均不计入**。

## 1. 版本与证据边界

| 项目 | 内容 |
| --- | --- |
| 审查基准提交 | `7d46223d33d0d8d2aaba952e064cacdb2ab4ec52`（PR #1 上的 v2.2.0 文档提交；前序审查基准为 `7e4cc510e2ea42b9691aa4360b17f87599007271`） |
| 规则版本 | `PIIA_PERSONAL_INVESTMENT_INTELLIGENCE_AGENT_V2`，`version = "2.2.1"` |
| 冻结代码提交 | `378b093c33fe879e1cc73bf276a869dc652a4cc4`（`fix(piia-v2.2.1)`） |
| 配置文件 LF SHA-256 | `40a8e6aa6c9782f58c355f85af9a9339b4d939d4c52f6a5154f597e7f3d512ba`（[config/piia_agent_v2.yaml](config/piia_agent_v2.yaml)） |
| 受控 7 文件清单 `manifest_sha256` | `0c8358c94e3cf12beb0a719fef4c7227516bc77bce2a7528c0e3343cbdc47578` |
| 前瞻起算时点 | `2026-11-23T09:00:00+00:00`（`prospective_freeze_started_utc`，严格晚于修复提交推送时间） |
| 策略/成本/风险参数 | 与 v2.0.0 / v2.1.0 / v2.2.0 **完全相同**：2x 逐仓、单笔 1% 止损预算、20% 现金底线、50% 最大保证金、Taker 0.06%、滑点 0.05%/边（另并行 2x 滑点）、3% 单日 / 10% 总回撤停机 |
| 受控文件 | `PIIA_V2_PROTOCOL.md`、`config/piia_agent_v2.yaml`、`src/piia/intelligence/agent_v2/{pillars,roles,risk_gate,ablation,continuous_paper}.py`（`python -m piia agent-verify-freeze` 校验） |

旧版本证据边界保持不变：
- v2.0.0（`a3c42e3`）`2026-11-16..11-22` 结果仍为「历史回放，事前冻结未获独立验证」；
- v2.1.0（`7a2b518`）与 v2.2.0（`3b975fd`）冻结期真实前瞻周期数均为 0，已被本版本取代；
- 旧账本 `data/piia/piia_v2_continuous.sqlite`（v2.0，无证据类型标签）与 `data/piia/piia_v2_1_continuous.sqlite`（若存在）**原样保留、只读审计、永不写入、永不计数**。

## 2. 本版修复（不改变任何策略阈值）

1. **账本按证据类型隔离**：真实采集只写 `data/piia/piia_v2_2_live_prospective.sqlite`（标签 `live_prospective`），演示只写 `data/piia/piia_v2_2_demo_synthetic.sqlite`（标签 `demo_synthetic`），调用者注入样本只能写入显式指定的 `test_or_injected` 账本。标签写入不可修改的 `ledger_meta` 表，底层写入函数按标签拒绝跨类型写入（不依赖界面传参）；无标签的旧账本一律拒绝写入。真实账本上的人工审批拒绝注入报价、K 线、资金费或时钟。
2. **冻结起算前阻断与预污染账本拒写（v2.2.1 新增）**：
   - 在 UI、CLI 与 FastAPI 共用的底层真实采集入口 `run_prospective_cycle`，先按 `prospective_freeze_started_utc` 校验本次预期整点；早于起算点时**在抓数和打开/创建 SQLite 之前**直接返回 `rejected_before_freeze_start`，不创建账本、不初始化本金账户、不写事件、不生成提案、不推进仓位。
   - 若既有的 `live_prospective` 账本已包含早于 `prospective_freeze_started_utc` 的周期、账户事件或提案，`inspect_ledger` 将其标为 `contaminated_pre_freeze_cycles`（`usable_as_live_prospective_ledger = false`），状态校验返回 `INVALID_PRE_FREEZE_CONTAMINATED_LEDGER`（计入数为 0），`_preflight_existing_db_for_write` 在只读阶段拒绝后续写入并保持原文件字节级不变。
3. **三合约收盘对齐门**：真实采集要求 BTCUSDT、XAUUSDT、NDX100USDT 最新完整 1h K 线收盘时间一致，且等于采集时刻所在小时的整点（即刚收盘的小时）；缺失、断档、滞后、未来 K 线或采集跨小时均返回明确原因，**当次不打开账本、不写周期、不生成提案、不推进任何影子账本的仓位、资金费或 K 线游标**。通过后按共同截止时点显式选取 `cutoff-1h` 信号 K 线。
4. **前瞻计数口径**：只计入同时满足以下条件的观测 —— 真实模式；账本标签为 `live_prospective` 且未检出混合数据或冻结前记录；`manifest_sha256` 等于当前冻结清单；`policy_version` 等于当前版本；截止时点不早于前瞻起算时点且为整点；采集开始与写入时间都落在该截止小时内；同一截止小时只计一次。

## 3. 评估规则（预注册）

- 状态页分别报告：计入的真实观测数、最长连续小时数（及起止）、观测区间内缺失小时数、自起算以来缺失小时数、各类排除原因计数，以及账本是否可用作前瞻验证（`usable_for_prospective_validation`）。
- **样本是否足够只看最长连续窗口**：最长连续真实观测 `>= 168` 小时前一律为 `PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE`（待验证）；零散记录总数再多也不能替代连续窗口。
- 达到 168 连续小时后状态仅变为 `PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED`（待审计），仍须人工审计资金费完整性、停机、对照账本后才能讨论结论；淘汰/证伪标准沿用 v2.1：若 `M4_full_piia_v2` 在 1x 或 2x 滑点下不能整体优于 `B0_cash` / `B1` / `B2` / `B3`，如实记录为尚未证明净决策增益，不得回调参数。

## 4. 旧账本与冻结前账本处理方式

- 运行 `python -m piia agent-inspect-ledger` 可只读审计真实、演示与旧账本：报告标签、周期模式分布、`init_book` 模式分布、是否混合、是否含冻结前记录，以及是否可用作真实前瞻账本。
- 混合、无标签或已含冻结前周期/事件的账本：**保留原文件不动**（测试校验字节级不变），不迁移、不“仅剔除旧行后继续使用”，因为模拟余额（如 100U vs 500U）、仓位、停机与游标可能已被影响。v2.2.1 从全新的干净真实账本重新开始。

## 5. 登记

- 冻结代码提交 `378b093c33fe879e1cc73bf276a869dc652a4cc4`；配置 SHA-256 `40a8e6aa6c9782f58c355f85af9a9339b4d939d4c52f6a5154f597e7f3d512ba`；清单 `0c8358c94e3cf12beb0a719fef4c7227516bc77bce2a7528c0e3343cbdc47578`。哈希只证明仓内文件一致，不是第三方可信时间戳。
- 登记时（`2026-11-23` 约 06:42 UTC）真实前瞻账本 `data/piia/piia_v2_2_live_prospective.sqlite` 尚不存在：**计入观测 0 小时，最长连续 0 小时，状态待验证**（前瞻起算点设为 `2026-11-23T09:00:00+00:00`）。
- 本地旧账本 `data/piia/piia_v2_continuous.sqlite`：无标签（v2.0），1 个真实周期（v2.0 清单 `17b31673…`），未检出演示行，但因无标签、早于冻结且清单不同，按规则只读保留、不计数。
