"""Public Bitget USDT-perpetual specs captured for local paper sizing.

These are not an exact liquidation model. Maintenance tiers and mark prices
were read from documented public endpoints; venue liquidation, ADL, and
account-class (Classic/UTA) rules remain unverified. Do not enable live orders
from this module.
"""
from __future__ import annotations

CONTRACTS = {
    "BTCUSDT": {
        "instrument_id": "btcusdt-bitget-perp",
        "ccxt_symbol": "BTC/USDT:USDT",
        "venue": "bitget",
        "instrument_type": "perpetual",
        "settlement_currency": "USDT",
        "contract_size": 1.0,
        "min_qty": 0.0001,
        "qty_step": 0.0001,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "min_leverage": 1,
        "max_leverage_venue": 150,
        "tier1_keep_margin_rate": 0.0040,
        "tier1_end_unit": 200000,
    },
    "XAUUSDT": {
        "instrument_id": "xauusdt-bitget-perp",
        "ccxt_symbol": "XAU/USDT:USDT",
        "venue": "bitget",
        "instrument_type": "perpetual",
        "settlement_currency": "USDT",
        "contract_size": 1.0,
        "min_qty": 0.01,
        "qty_step": 0.01,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "min_leverage": 1,
        "max_leverage_venue": 100,
        "tier1_keep_margin_rate": 0.0050,
        "tier1_end_unit": 50000,
    },
    "NDX100USDT": {
        "instrument_id": "ndx100usdt-bitget-perp",
        "ccxt_symbol": "NDX100/USDT:USDT",
        "venue": "bitget",
        "instrument_type": "perpetual",
        "settlement_currency": "USDT",
        "contract_size": 1.0,
        "min_qty": 0.0001,
        "qty_step": 0.0001,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "min_leverage": 1,
        "max_leverage_venue": 50,
        "tier1_keep_margin_rate": 0.0100,
        "tier1_end_unit": 10000,
    },
}

PROVENANCE = {
    "contracts_url": "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES",
    "tiers_url": "https://api.bitget.com/api/v2/mix/market/query-position-lever?productType=USDT-FUTURES&symbol={symbol}",
    "ticker_url": "https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol={symbol}",
    "retrieved_at_utc": "2026-11-16T07:25:25.211000+00:00",
    "retrieved_at_asia_taipei": "2026-11-16T15:25:25.211000+08:00",
    "exact_liquidation_verified": False,
    "note": "Public snapshot used for local_simplified paper only. Mark/tier data can change; re-fetch before relying on size limits.",
}

# Last public marks observed in the same request window. Informational only.
SAMPLE_MARKS = {
    "BTCUSDT": {"last": 64607.6, "mark": 64612.1, "index": 64640.601304347826087, "funding_rate": -0.000025, "observed_at_utc": "2026-11-16T07:25:25.211000+00:00"},
    "XAUUSDT": {"last": 3857.09, "mark": 3857.01, "index": 3855.343674391962247, "funding_rate": 0.0, "observed_at_utc": "2026-11-16T07:25:25.461000+00:00", "funding_rate_provenance": "public_ticker_field_zero_may_be_missing"},
    "NDX100USDT": {"last": 27686.0, "mark": 27686.0, "index": 27674.4457405381445894, "funding_rate": 0.0, "observed_at_utc": "2026-11-16T07:25:25.733000+00:00", "funding_rate_provenance": "public_ticker_field_zero_may_be_missing"},
}
