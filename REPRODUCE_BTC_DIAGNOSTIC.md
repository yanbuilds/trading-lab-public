# BTC 历史诊断复算

这是一份**已查看过的 BTC 永续历史价格诊断**，不是前瞻实测、留出样本或净收益。窗口为 2026-01-01 00:00 UTC 至 2026-11-16 06:00 UTC（终点不含），不是完整年度。冻结候选和 EMA 均使用每小时可见的最近 300 根收盘线、独立 100 USDT 起始资金、2 倍示例杠杆和相同的离线费用及风控；两者持仓暴露、退出规则不同，不是风险匹配对照。现金参考为 100 USDT。

## 输入与版本

下列本地 CSV 不在仓库中，须自行合法取得**相同文件**；文件哈希只证明字节一致，不证明来源或行情真实。2023、2024 文件供第一步的已有探索窗口使用，2026 对照使用 `BTCUSDT-year.csv`。

| 文件 | SHA-256 |
| --- | --- |
| BTCUSDT-2023.csv | `d3bf262d70e12e72f93567c880123ea75030e8142b905130304bbcdbf361f63f` |
| BTCUSDT-2024.csv | `3fe805935bb747201a20bded58215600da53d313833254ac3209a0ac19b9088e` |
| BTCUSDT-year.csv | `8abd6991573104be3785dea2658204e886fc38afc56793428cc445a248a2caf4` |

前三步复算使用 Git 提交 `294bce9` 的代码、Python 3.13 和仓库 `requirements-perps.lock`；该提交包含前三步的脚本。一年区间入口在其后加入，复算时应另外记录 `git rev-parse HEAD`。依赖环境单独创建，不能用于实盘。冻结政策及代码指纹由脚本检查，来源文件或政策不匹配时停止。

## Windows PowerShell

在仓库根目录执行，将数据目录换成包含上述三个文件的目录。运行前用 `Get-FileHash -Algorithm SHA256` 对照上表。下面的环境变量是本次任务专用变量；报告只写入本地 `artifacts`，不要提交行情、报告或虚拟环境。

```powershell
py -3.13 -m venv ..\btc-repro-env
..\btc-repro-env\Scripts\python.exe -m pip install -r requirements-perps.lock
..\btc-repro-env\Scripts\python.exe -m pip install -e . --no-deps
$env:PYTHONPATH = 'src'
$marketDataDir = '..\https-github-com-yanbuilds-trading-lab\data\perps-extended'
..\btc-repro-env\Scripts\python.exe -m experiments.btc_frozen_policy_history_check --data-dir $marketDataDir --out artifacts\btc_history_verify_chain.json
..\btc-repro-env\Scripts\python.exe -m experiments.btc_rolling_window_replay --data-dir $marketDataDir --reference artifacts\btc_history_verify_chain.json --out artifacts\btc_rolling_candidate_verification.json
..\btc-repro-env\Scripts\python.exe -m experiments.btc_rolling_ema_comparison --data-dir $marketDataDir --candidate-reference artifacts\btc_rolling_candidate_verification.json --out artifacts\btc_comparison_reproduced.json
```

三个命令必须依次成功；第二、三步会检查前一步报告和输入的哈希。不要把缺失文件、计算失败或不同文件当作相同结果。

## 核对值及边界

本地此次运行得到：冻结候选期末约 96.7912 USDT、最大回撤 3.2088%、11 笔平仓；EMA 期末约 113.9480 USDT、最大回撤 6.3728%、8 笔平仓。两者均触发停机。浮点结果可能随运行环境变化；差异须检查依赖、代码和输入，不得忽略。

该模拟包含假设手续费及滑点，但**资金费未知且排除**；历史合约规格、盘口成交、实际强平与期末平仓成本未独立核验。报告不包含黄金和纳指，也不验证 AI 带来的增益。Git、本地时钟和哈希不能证明数据在历史决策时已知，结果不能用于宣称真实收益、稳定优势或开仓许可。

## 固定的一年重叠区间

在同样环境和 `BTCUSDT-year.csv` 上，可单独复算 2025-11-16 06:00 UTC 至 2026-11-16 06:00 UTC（终点不含），包含此前 300 小时预热：

```powershell
..\btc-repro-env\Scripts\python.exe -m experiments.btc_one_year_price_diagnostic --bars "$marketDataDir\BTCUSDT-year.csv" --out artifacts\btc_one_year_price_diagnostic.json
```

本地一次回放结果为候选 89.6081 USDT（52 笔平仓）和 EMA 89.7517 USDT（19 笔平仓），均从 100 USDT 开始并触发停机。报告包含模拟逐笔交易、输入和源码哈希。**该窗口在代码入口创建前已被查看且与其他研究重叠，入口并非事前注册；它不是留出测试或经验证净收益。** 资金费、历史规格、真实成交和强平仍未知；两账户的暴露及退出规则不同。

如需重新核对该 BTC 文件的公开行情，可手动运行下列只读命令。它请求 Bitget 当前公开历史数据，并将每页原始响应、哈希和缺口或差异写到本地；不会发出交易订单。请勿提交生成的文件。

```powershell
..\btc-repro-env\Scripts\python.exe -m experiments.audit_btc_public_overlap --bars "$marketDataDir\BTCUSDT-year.csv" --out artifacts\btc_public_overlap_check.json
```

核对范围包含上述一年及 300 小时预热期。历史公开数据可能修订；匹配只能证明本次获取的 OHLCV 与文件在该范围内一致，不能证明文件最初来源、当时可用时间、资金费或交易收益。数值比较使用报告内列明的容差，同时记录最大绝对差。

## 历史资金费来源核查（2026-11-19）

Bitget 当日公开历史资金费接口实际返回的记录从 2026-08-21 起，未覆盖一年诊断中的 2025 年交易。另一来源 [CryptoHFTData 的 Bitget 资金费数据](https://www.cryptohftdata.com/datasets/bitget-funding-rate-data) 宣称从 2025-07-15 开始提供数据，但其[已知数据缺口](https://www.cryptohftdata.com/docs/data-gaps-and-corrections)明确列出 Bitget 合约 2025-09-23 07:00 至 2025-12-26 20:00 UTC 的小时文件缺失，覆盖了本次两组模拟全部交易的日期。2026-11-19 手动请求 BTCUSDT 标记价格文件时，2025-11-18 的 00、07、15 时以及 2025-12-01、2025-12-22 的 07 时均返回 HTTP 404；2026-01-01、2026-08-22、2026-11-18 的 07 时请求返回 HTTP 200。单小时抽查不能证明其他时间或字段完整。

该第三方文件是交易所推送的标记价格和当时的资金费快照，不应未经核对便当成实际结算费率。因为缺口覆盖所有模拟交易，本次仍不能计算可核验的资金费后收益；其他交易所费率不能替代 Bitget 费率。

另一候选是 [Petrishko 的 2026 年论文及附件](https://dspace.vut.cz/items/0c45b1d0-a8d3-4a43-9941-fc71ef4f84fd)，论文称其缓存包含 Bitget 历史资金费。学校附件中的 `fetch_thesis_data.py` 指向外部数据镜像并声明 SHA-256；2026-11-19 对该镜像发送 HEAD 请求返回 HTTP 530，尚未取得缓存。附件本身不含 `funding.parquet`。在拿到数据、核实具体合约、结算时间、来源与是否填补之前，不能将论文的数据描述当作可复算资金费。

### 停机与交易明细解读

上面的一年是账户观察窗口，不代表策略全年持续有仓位。该次回放中：

| 项目 | 冻结候选 | EMA 对照 |
| --- | ---: | ---: |
| 最后一笔平仓（UTC） | 2025-12-22 10:00 | 2025-12-26 08:00 |
| 平仓后模拟账户 | 89.6081 USDT | 89.7517 USDT |
| 盈 / 亏 / 近似持平 | 14 / 37 / 1 | 2 / 17 / 0 |
| 模拟手续费合计 | 3.6388 USDT | 1.2264 USDT |
| 有仓小时比例的上界 | 1.26% | 8.31% |
| 平仓原因：反向信号 / ATR 止损 / 停机 | 43 / 8 / 1 | 8 / 10 / 1 |

两组都因损失保护停止，之后没有重新开仓。离线引擎在小时开盘或收盘检查权益是否低于当日初始权益的 97%，或低于起始资金的 90%；触发后停机并按引擎规则平仓。成交费用和滑点可以使最后金额低于阈值。现有报告只记录 `loss_halt`，不区分是哪一项阈值首先触发，不应自行推断；完整交易清单在本地 JSON 的 `accounts.*.diagnostic_trades`。有仓小时比例把整根小时线计为有仓，是时间暴露上界，并非风险匹配或名义价值加权暴露。

### 独立账本算术检查

复算后可运行：

```powershell
..\btc-repro-env\Scripts\python.exe -m experiments.audit_btc_diagnostic_ledger artifacts\btc_one_year_price_diagnostic.json
```

检查器独立核对交易时间及重叠、平仓数量、方向、盈亏和退出原因统计、已报告的逐笔盈亏之和与期末权益，并要求资金费保持未知。2026-11-19 对本地报告检查通过：候选逐笔盈亏合计 -10.3918872 USDT，EMA 合计 -10.2482724 USDT。测试使用明确标记的合成数据。

**此检查只证明报告内部算术一致。** 目前每笔交易缺少成交价格、数量和单笔费用，无法单独重算各笔盈亏；汇总费用也不能证明成交符合真实市场。它不验证行情真实性、信号时点、资金费、强平或策略优势，也不能检测所有可能的篡改。引擎已冻结，此次未更改策略或账本生成逻辑。

### 模拟成交逐笔重算

原诊断 JSON 本身仍缺少逐笔价格、数量和费用。可在不改动冻结引擎源码的情况下，另行重跑并捕捉该引擎的模拟平仓局部变量：

```powershell
..\btc-repro-env\Scripts\python.exe -m experiments.btc_fill_trace --bars "$marketDataDir\BTCUSDT-year.csv" --reference artifacts\btc_one_year_price_diagnostic.json --out artifacts\btc_simulated_fills.json
```

2026-11-19 本地重跑提取候选 52 笔、EMA 19 笔模拟成交。每笔记录数量、含假设滑点的进出价、假设费率、两端费用和毛盈亏；独立公式重算后与引擎交易盈亏一致，且重跑的汇总及逐笔结果与原报告一致。输出及重跑报告只保存在本地，不提交。该追踪依赖固定版本引擎的 Python 局部变量，形状变化或记录不完整时应失败；合成多空测试检查开启追踪前后结果相同，并检查价格篡改被拒绝。

这些是引擎内部的**模拟**成交，不是交易所成交，重算使用与引擎相同的模拟价格与规格，不能独立验证这些输入的真实性。资金费仍未知，仍不是净收益或留出测试；成交记录的捕捉不验证信号因果性、真实手续费、盘口滑点或强平。

### Tardis 月首 BTC 快照的有限核对

[Tardis Bitget Futures 文档](https://docs.tardis.dev/historical-data-details/bitget-futures)说明每月第一日的 BTCUSDT 衍生品 ticker 样本可免密钥获取。2026-11-19 分别取得 [2025-12-01 文件](https://datasets.tardis.dev/v1/bitget-futures/derivative_ticker/2025/12/01/BTCUSDT.csv.gz)（SHA-256 `cb4148090f0a622289d764c717c8c986281be66b20c2cb01daf3da9f61d04a6b`）和 [2026-11-01 文件](https://datasets.tardis.dev/v1/bitget-futures/derivative_ticker/2026/11/01/BTCUSDT.csv.gz)（SHA-256 `c701f0727a25355c24aeca0b418fa12472fd84f0e0549c500027f8f0a8409975`）；文件仅本地保存。以下命令挑选交易所事件时间和采集接收时间均早于对应结算时刻、并且指定的下一资金费时刻相同的最近接收快照：

```powershell
..\btc-repro-env\Scripts\python.exe -m experiments.audit_tardis_funding_sample artifacts\tardis_bitget_btc_2025_12_01.csv.gz 2025-12-01T08:00:00+00:00 2025-12-01T16:00:00+00:00
..\btc-repro-env\Scripts\python.exe -m experiments.audit_tardis_funding_sample artifacts\tardis_bitget_btc_2026_11_01.csv.gz 2026-11-01T08:00:00+00:00 2026-11-01T16:00:00+00:00
```

旧样本两次快照费率分别为 `0.0001` 和 `0.000033`，采集接收时间距结算分别约 0.096 秒和 0.670 秒；当时 EMA 模拟持仓跨过这两个时刻。新样本对应费率为 `0.000055` 和 `0.000083`，与本次取得的 Bitget 公开历史接口同一结算时刻记录一致。该接口旧记录已不可取得，因此这两个较新的相符点不能证明 2025 年快照就是最终结算值，也不能补齐其他日期。快照不用于修改前述收益，仍未验证历史资金费后收益。

进一步核对了 2026-09-01 与 2026-10-01 的 08:00、16:00 UTC，加上上述 11 月的两个时点，合计六个快照与当次归档的 Bitget 官方历史费率一致。9 月样本 SHA-256 为 `b0ef8cc39cc8f1424f7490cfcf47683dd19e237e160d950ad7b30370b1dd029c`，10 月为 `934c405c137c91ea05b3fdd1ab92b36892351bbb70123121d264fb54dedc3f86`。本地官方响应归档 `artifacts\btc_funding_public_pages_20261119.json` 的 SHA-256 为 `01e107a09c6928873d10d113694147aab6a47f7e2905eeb5994dd14177a01658`，原始响应不提交。

可在前述核查命令后追加 `--official-pages artifacts\btc_funding_public_pages_20261119.json`，逐点比较快照和归档接口记录，缺失的官方记录显示 `null`，不会自动填零。旧官方响应可能已滚出接口；这些文件及其哈希只能由已保存相同归档的复核者原样复算。六个近期匹配样本不能确定历史日、其他交易时刻或缺失期间的结算费率。
