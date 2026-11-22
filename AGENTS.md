# Contributor / Codex guide

Legacy baseline (unchanged): this is a research-only Python project: BTC-USD spot, GLD gold ETF proxy, QQQ Nasdaq-100 ETF proxy. QQQ is not the literal index. No leverage, short selling, broker credentials, or execution integrations.

## Setup and checks

Python 3.13 is the tested runtime. Create a virtual environment, install `requirements.lock`, then `pip install -e . --no-deps`. Run `python -m pytest`. Use `python -m trading_lab fetch`, `backtest`, `paper`, `scenarios`; `python -m trading_lab.directions` outputs conditional trend research. Streamlit: `streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8787`.

Never commit .env, credentials, user holdings, downloaded market data, generated artifacts, or venvs. Synthetic fixtures must remain explicitly labelled. Data access failures must remain visible; never substitute fixture data for real data.

## Engineering priorities before performance claims

1. Independently review the corrected timestamp-ordered UTC event engine. BTC daily open occurs before US ETF opens. Later ETF sales must never fund earlier BTC purchases. ETF closes must not enter earlier BTC sizing or NAV. Use the official market calendar including holidays/early closes.
2. Preserve the future-cash and T+1 settlement regressions in tests/test_utc_execution.py; expand coverage. Preserve positive cash, no shorts, cost-aware fills and causal next-session signals. Current results remain a research prototype, not independently audited.
3. Add a fair benchmark with identical starting capital/time and exposure/cash assumptions, then walk-forward tests. Do not tune on held-out periods. Price targets or confidence percentages require held-out calibration.
4. Improve missing-bar validation and input checks; verify ETF adjusted OHLC conventions.
5. Keep LLM commentary separate from deterministic signal and risk modules. Bearish research classification is NOT permission to short. Paper is a one-shot replay, not an always-on broker simulator.

Dependencies include vectorbt under Apache-2.0 + Commons Clause. Do not label the entire project as unrestricted Apache/MIT or remove third-party notices. No root project license is granted yet; obtain a separate license decision before redistribution/commercial use.

## Bitget perps v1 — separate subsystem

- User now explicitly requests BTC/XAU/Nasdaq100 leveraged contract research with a final total 30-USDT execution test budget. Do not silently use GLD/QQQ/tokenized gold or carry over legacy returns.
- `freqtrade/` is a native Freqtrade candidate, `src/trading_lab/perps/` is explicitly an additional offline stress harness, NOT the native engine or exchange demo. Environment `../freqtrade-env` is separate from `.venv`. See README exact commands.
- Config must remain dry_run=true, isolated, one open position, no credentials; native initialization/start guards and callback fail-closed checks are mandatory. No live API actions or credentials requested in this iteration.
- Example leverage 2x, risk 1%, quantity rounded DOWN; minimum-order budget failures must be reported, never upsize. Stop gaps can exceed budget. No auto margin topup, no cross collateral.
- Preserve old tests and add contract regressions for long/short, fees/funding notional, candle boundary pagination, intrabar/gap stops, risk budgets, minimum quantity, no future cash and live guards.
- Remaining before demo/live: independently validate mark price/tier liquidation and fees, funding schedules changing over time, Classic/UTA adapter support, hosted stop trigger type and reduce-only semantics, idempotency/reconciliation after restart and exchange demo symbol coverage. Current offline stress model is not exact liquidation simulation.
- Native Freqtrade backtest uses separate recent timerange because its downloader supplied only last100 funding records; custom fetcher uses documented pageNo pagination. Do not set missing funding=0 and then claim net funding performance.
- Never commit provider data, generated reports, credentials or environments. Avoid modifying old ETF code except the explicit perps CLI dispatch. No hyperparameter optimization or extrapolated accuracy claims.


## PIIA Phase 1 — separate intelligence module

- Package: `src/piia/intelligence/`. Isolated env `../piia-env`. Do not break `.venv` or `../freqtrade-env`. SQLite is the local default; PostgreSQL is schema-compatible but **not tested**.
- Invariants: append-only thesis revisions and evidence; no update/delete API; SQLite triggers enforce immutability (an administrator can still edit the DB file; hashes are not a truth guarantee). Subjective confidence is not calibrated accuracy. Prospective timestamps only; do not import retrospective validated track records. Missing data stays missing.
- Risk is a deterministic function, not an LLM. Futures/perp/leveraged executable paper plans must fail closed while the exact margin and liquidation model is unimplemented (`supported=false`). Schema already includes instrument_type, venue, contract_symbol, settlement currency, contract_size, side, leverage, margin_mode, nullable stop/exit.
- Multi-agent roles (researcher/critic/synthesizer) share typed evidence I/O and have **no order permissions**. They are designed, not wired. Do not label a deterministic demo countercase as independent LLM review. No trade/executor routes. No live keys. Existing Freqtrade lab stays untouched until a later approved integration.
- Bind 127.0.0.1 only. There is no authentication; document that fact. Approval is an explicit API action, never assumed.
- Tests: `../piia-env/bin/python -m pytest tests/test_piia.py tests/test_piia_api.py tests/test_piia_ui.py`. Keep the original 45 strategy-lab tests green in `../freqtrade-env`.
- Next Codex priorities: (1) FRED/ALFRED point-in-time macro evidence, (2) optional LangGraph researcher/critic/synthesizer on the existing analysis table still without order permissions, (3) horizon-dated outcome metrics that do not treat confidence as accuracy, (4) paper integration with Freqtrade only after an exact margin/liquidation model, (5) never add live credentials or trade routes without a new explicit user approval.


## Integrated v1 — research workflow + local paper

- LangGraph researcher→critic→synthesizer runs in labelled fixture mode. Official API adapter exists but must not be called without `PIIA_MODEL_PROVIDER`, allowlisted `PIIA_MODEL_BASE_URL`, `PIIA_MODEL_NAME`, and `PIIA_MODEL_API_KEY`. If unset, return `needs_model_configuration` and keep `independent_llm_review=NOT_WIRED`. Never log secret values.
- Do not flip `margin_model_supported` or enable live/demo exchange orders. Local paper uses `trading_lab.perps.risk` (`size_order`, `stop_fill`, `exit_pnl`) with owner file `config/piia_paper_risk.yaml`. LLM cannot edit that file or the kill switch.
- Tests to keep green: `tests/test_piia*.py` in `../piia-env`, plus the original 45 tests in `../freqtrade-env`.
- Next Codex backlog: (1) user-configured official model research with cost logging and no fabricated tokens, (2) exact venue liquidation/mark-tier verification before any demo/live discussion, (3) Freqtrade paper adapter still dry_run-only, (4) no scheduled watchers, (5) never add live keys or trade routes without a new explicit approval.
