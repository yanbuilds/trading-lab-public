# Research integration and license matrix

This describes what this increment reuses, what is actually wired, and what stays
unconfigured. Star counts and third-party marketing claims are not evidence of a
trading edge. The project still has no root license; third-party terms stay with
those components.

## License and reuse matrix

| Component | License (as checked 2026-11-17) | How we use it | Vendored into core? |
|---|---|---|---|
| Existing PIIA researcher / critic / synthesizer and LangGraph workflow | project code; LangGraph MIT | Conceptual debate workflow already in the repo | No new copy |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | Apache-2.0 | Conceptual reference only: independent research roles, retain disagreement, no auto-trade | No. Do not copy the repository. Apache-2.0 would allow reuse with attribution, but we did not import its source. |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | GPL-3.0 | Existing separate environment and candidate strategy only | No. Keep it isolated from the core library. |
| CCXT | MIT | Existing public market access | No source copy |
| [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) | Apache-2.0 | Optional local probe already in `experiments/`; not a default dependency | No weights or source vendored |
| Kronos (`shiyu-coder/Kronos`) | check upstream MIT/terms before any copy | Optional existing probe only | No weights or source vendored |
| FRED / ALFRED official vintages | public data via the existing adapter | Wired for current conservative date-vintage snapshots | No |
| [OpenBB](https://docs.openbb.co/odp/v4/python/faqs/license) platform | AGPL-3.0 (commercial license optional) | Not imported, not vendored. An optional future external tool only if the user accepts AGPL or a commercial license. | **No** |
| vectorbt | Apache-2.0 + Commons Clause | Legacy ETF research only | Already an isolated dependency |

## Wired and runnable in this increment

- Deterministic technical regime labels (`trend` / `range` / `extreme_vol` / `uncertain` / `insufficient_data`) and one frozen range-gated Bollinger mean-reversion candidate, compared with the existing 1h EMA trend book at 100 USDT (optional separate 30 USDT book).
- Research evidence bundle: independently tagged technical regime, macro snapshot labels, statistical forecast benchmark, neural-forecast library status, and AI adapter status. Contradictions are retained. No vote and no orders.
- Current ALFRED adapter for four official series, with missing or stale series left visible. Macro-enhanced historical alpha is blocked because this adapter is not a historical point-in-time archive.
- Official model request schemas for OpenAI, xAI, and Anthropic, unit-tested without sending paid calls. Real model research still requires user credentials and stays `NOT_WIRED` for independent review.
- Random-walk-with-drift statistical forecast, labelled uncalibrated, used when Chronos/Kronos are not installed.

## Unconfigured or not validated

- Chronos and Kronos weights are not downloaded here. If the libraries or pinned files are absent, neural forecast status is `unavailable`.
- Official LLM calls are not made unless the user later sets allowlisted environment variables. No secrets are requested in this increment.
- Historical macro vintages as-of each past decision time are not available in the current adapter, so no historical backtest may claim a macro-enhanced edge.
- Open interest remains the existing optional public snapshot field; it is not a directional signal.
- Liquidation, historical funding-schedule changes, and live or demo exchange execution remain unverified and disabled.
- OpenBB is intentionally not integrated.

## Commands

```bash
# Frozen mean-reversion vs 1h EMA trend, 100 USDT primary, complete funding required
../freqtrade-env/bin/python experiments/regime_mean_reversion.py \
  --data-dir /absolute/path/to/local/perps-extended --include-optional-30u

# Research bundle from closed prices (no orders, no paid model)
../piia-env/bin/python -m piia.intelligence.research_fusion \
  --closes-json artifacts/sample_closes.json --symbol BTCUSDT
```

## Chronos-t5-tiny（2026-11-17）

官方模型 `amazon/chronos-t5-tiny`，修订 `29d808298f1a62493e7b9a5e08529d0d930fa189`，Apache-2.0，权重 `model.safetensors` SHA-256 `eed458758bd8165d80f496a90bcd2cfed9f1bf7d7b08677acd64e0f7d72bdcf2`。通过独立 `forecast-env` 和 `tools/run_chronos_tiny.py` 在 CPU 上调用 `chronos-forecasting` 的标准 `ChronosPipeline`，`trust_remote_code=False`。这不是随机游走替身，也不是已校准或已验证的交易优势；训练数据泄漏情况未知。Kronos 仍未接入。

## Public Bitcoin on-chain and US spot BTC ETF flows (2026-11-17)

No dedicated Glassnode, CoinMetrics, Farside, or SoSoValue connector was installed. Public HTTP is used instead. These adapters collect **current** research evidence only; they do not backfill historical decision points and are not an alpha claim.

- Bitcoin on-chain: documented [mempool.space REST API](https://mempool.space/docs/api/rest). Required set: tip height, tip hash, latest confirmed block time, mempool count/vsize/total fee, recommended fees, difficulty adjustment, and current hashrate/difficulty from `/api/v1/mining/hashrate/3d`. Hashrate is stored as hashes per second; EH/s is a labeled derived field. One field is never treated as a complete on-chain picture. A tip older than three hours is stale. This is not exchange-flow, UTXO-age, or realized-cap data. The project uses the public HTTP API as a client and does not vendor mempool software.
- US spot Bitcoin ETF net fund flows: public [Farside Bitcoin ETF Flow](https://farside.co.uk/btc/) table, units US$ millions, measure = net fund flows, not AUM or price. Expected tickers: IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC. A dash is unreported, not zero. A later incomplete day does not replace the last complete fund-level day. A complete day older than seven calendar days is stale. Farside is a compiler of issuer disclosures and disclaims errors; it is not a primary SEC daily filing. The full table is stored only in gitignored local snapshots and is not republished in this repository.
- Applicability: these datasets are hard requirements for a BTC trading recommendation only. They do not automatically fail XAU or Nasdaq-100. Single-stock fundamentals are not a universal hard gate for gold or an index perpetual. Official free gold inventory/physical-demand and Nasdaq-100 aggregate valuation/breadth feeds were not wired (issuer or index sites often require signup); those remain labeled research gaps, not fake completeness.
- Forecast calibration, hosted official model autonomy, and exact liquidation remain unverified. Official U.S. calendars and a sampled Polymarket public-evidence adapter are wired separately below; they are not an alpha claim.

## Official U.S. calendars and partial gold positioning (2026-11-17)

No dedicated Fed, BLS, BEA, CFTC, Nasdaq index, or gold-ETF connector was installed. Public official HTTP is used. These adapters collect **current** published schedules or reports only; they do not backfill historical decision points.

- Event calendar: official [Federal Reserve Board calendar JSON](https://www.federalreserve.gov/json/calendar.json) (required subset is `FOMC` and `Beige` events; speeches are omitted), official [BLS monthly selected-releases schedule](https://www.bls.gov/schedule/news_release/) (`/schedule/{year}/{mm}_sched.htm`), and official [BEA release ICS](https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics). Times on Fed and BLS pages are published clock times assumed to be America/New_York. This is a schedule, not a verified exact release tick, and it does not cover non-U.S. central banks or earnings. Completeness requires all three official sources.
- Gold positioning: official [CFTC COMEX futures-only Commitments of Traders](https://www.cftc.gov/dea/futures/deacmxsf.htm), contracts of 100 troy ounces, CFTC code 088691. This is futures positioning, not gold ETF holdings/flows, warehouse inventory, or physical/central-bank demand. The ETF/physical-demand research gap stays open. U.S. government works are used as public official data; the project does not vendor CFTC software.
- Nasdaq-100 official aggregate valuation and breadth were not wired. Attempts: `https://www.nasdaq.com/market-activity/index/ndx` failed or timed out from this environment; `https://indexes.nasdaqomx.com/` and `/Index/Overview/NDX` returned HTML with a subscribe/sign-in wall and no parseable P/E, market-cap, or breadth figures in the unauthenticated page. A Nasdaq official license or user-approved signup would be required; that was not requested.
- Official issuer gold ETF holdings pages (including guessed SPDR GLD CSV paths and the JS-heavy `spdrgoldshares.com` product page) did not expose a stable unauthenticated holdings/flow file in this check.

## Polymarket public prediction-market evidence (2026-11-17)

No dedicated Polymarket connector was installed. Official documented public HTTP is used, read-only: no bets, no orders, no wallet, no API credentials, and no Relayer. Today's odds are not backfilled into prior frozen cycles.

- Discovery: official [Gamma API](https://docs.polymarket.com/market-data/discover-markets) `https://gamma-api.polymarket.com/public-search` and `https://gamma-api.polymarket.com/events/{id}`.
- Books and quoted prices: official [CLOB public reads](https://docs.polymarket.com/market-data/prices-order-books) `https://clob.polymarket.com/book`, `/midpoint`, `/spread`, and `/last-trade-price`. Best bid and best ask are the last entries of the official book arrays (bids ascend, asks descend).
- 24-hour changes: official Data API `https://data-api.polymarket.com/v2/prices-history` for the selected outcome token, measured in percentage points, only when a documented point exists within one hour of the 24-hour mark. Otherwise the change is `not_available`, never zero and never a return.
- Scope: a sample of contemporaneous Fed/rate, inflation, recession, political-macro, Bitcoin, gold, and Nasdaq-related events from public search. A topic missing from the sample is not proof the venue has no such market.
- Interpretation: CLOB mid/bid/ask/last trade and Gamma `outcomePrices` are stored as distinct quoted prices. An implied probability is labeled only under a standard $1 binary-payoff assumption and is not a calibrated probability or a fact. Gamma `liquidity_num` is a platform-reported figure, not executable book depth. Resting size is in outcome tokens visible at read time. Dependent or overlapping markets are never summed, and odds do not force a trade stance.


## Frozen research versus fresh entry revalidation (2026-11-17)

No new connector is used. Official public Bitget market data already used by this project remains the only execution-context source: `https://api.bitget.com/api/v2/mix/market/orderbook`, `/ticker`, and `/contracts`. Requests are unsigned and read-only. Private, simulated-trading, or order endpoints are not called.

- The frozen research cycle (`STATE.json`, evidence, dispatch, inbox, import) is hashed and never rewritten. Revalidation writes only `execution_context/<revalidation_id>/`.
- A fresh ticker or book does not cure stale research: closed-candle age, explicit thesis expiry, flat / stand-aside / need-more-evidence stances, missing hard evidence, or tampered hashes still block a plan. The existing 5-minute quote freshness gate still applies to the new context.
- Visible book costing uses documented side convention (bids descending, asks ascending, best first). Long entry walks asks; short entry walks bids. Conservative exit liquidity, when evaluated, walks the opposite side and uses the worse of the stop and the visible exit VWAP. Quantity is rounded down to `minTradeNum` / `sizeMultiplier` and checked against `minTradeUSDT` and `maxMarketOrderQty`. Insufficient, empty, crossed, stale, or future books are rejected; the unseen tail is never assumed to fill. Mark price is recorded and is not an executable price. A snapshot is not a future-fill guarantee.
- Output is only a structured rejection or an unapproved, non-actionable paper proposal with an explicit expiry. Approvals are never synthesized. The virtual paper sandbox does not assume user holdings. A labelled `FIXTURE_NOT_REAL_MODEL_TRADE` path exists only for tests.


## Public Bitget position tiers and estimated isolated liquidation (2026-11-17)

Official documented public endpoint only: `GET https://api.bitget.com/api/v2/mix/market/query-position-lever?productType=USDT-FUTURES&symbol=...` as listed under Classic Contract Position. Authenticated `/api/v2/mix/position/*` endpoints (UID rate-limited, include live estimated liquidation price) are not called. The UTA v3 public path is a different account class and is not used as authority here.

- Official English field wording is "Minimum/Maximum value"; official Chinese is "价值下限/上限". For USDT-M linear contracts this is treated as a USDT notional bound. The JSON keys remain `startUnit`/`endUnit` without an explicit USDT suffix.
- Boundary inclusivity is not documented. A shared boundary uses the more conservative adjacent tier. Gaps or interior overlaps are rejected.
- Current snapshots are contemporaneous only and are not applied to historical decisions as if they had been known then.
- Paper risk consumes a validated current snapshot for isolated mode only: leverage caps, downward sizing, and an initial-margin + taker fee + one-interval funding reserve constraint. Configured leverage is never raised. Cross, UTA, and multi-asset modes are rejected.
- Isolated classic estimated liquidation follows the official formula in [Liquidation Mechanism](https://www.bitget.com/support/articles/12560603895826) and [Estimated Liquidation Price](https://www.bitget.com/support/articles/12560603808759), with the unpublished maintenance-margin deduction / pre-calculated offset set to zero. Trigger is mark price, not last traded price. The stop must sit strictly before this estimate; a gap through the stop remains a residual risk. This is algebraic use of the published isolated formula, not exchange-production liquidation validation. `exact_liquidation_verified` stays false.

## Chronos environment

Chronos-t5-tiny inference stays in the isolated `../forecast-env` with `requirements-forecast.txt` (`chronos-forecasting==2.3.2`) and `../hf-cache`. It is intentionally not installed into `piia-env`. Tests that import `chronos` should be run with `../forecast-env/bin/python -m pytest tests/test_chronos_weekly_probe.py`. Production research-cycle inference already uses `tools/run_chronos_tiny.py` via that environment.
