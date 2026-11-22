"""Current public 100-USDT contract-size feasibility, not a trade or account check."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import json
import math

import httpx
import numpy as np
import talib

from piia.intelligence.public_data import CANDLES, _closed_candles, candle_quality
from piia.intelligence.public_orderbook import fetch_execution_books
from trading_lab.perps.risk import Spec, size_order, worst_stop_loss_per_contract

SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")


def minimum_feasibility(symbol: str, price: float, atr: float, contract: dict) -> dict:
    """Compare minimum-order stop cost with a hypothetical 100-USDT risk budget."""
    if contract.get("symbol") != symbol or contract.get("status") != "normal" or contract.get("instrument_type") != "perpetual":
        raise ValueError("contract mismatch or unavailable")
    spec = Spec(symbol, *(float(contract[k]) for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")))
    if not all(math.isfinite(v) and v > 0 for v in (price, atr, spec.min_qty, spec.qty_step, spec.contract_size)):
        raise ValueError("invalid price, ATR or contract spec")
    required = max(Decimal(str(spec.min_qty)), Decimal(str(spec.min_notional)) / (Decimal(str(price)) * Decimal(str(spec.contract_size))))
    step = Decimal(str(spec.qty_step))
    minimum = float((required / step).to_integral_value(rounding=ROUND_CEILING) * step)
    worst_loss = minimum * worst_stop_loss_per_contract(price, 2 * atr, spec, slippage=.0005)
    quantity, reason = size_order(100, 100, price, 2 * atr, spec, leverage=2, slippage=.0005)
    return {"symbol": symbol, "minimum_qty": minimum, "risk_budget_usdt": 1.0,
            "estimated_minimum_normal_stop_loss_usdt": worst_loss,
            "hypothetical_qty_rounded_down": quantity, "reason": reason,
            "mark_is_not_executable": True, "exact_liquidation_verified": False,
            "funding_and_gap_loss": "unknown", "order_permissions": False}


def run() -> dict:
    now = datetime.now(timezone.utc)
    books = fetch_execution_books()
    if not books.get("ok"):
        raise ValueError("public contract or quote snapshot unavailable")
    result, retained = {}, {}
    with httpx.Client(timeout=20., follow_redirects=False) as client:
        for symbol in SYMBOLS:
            response = client.get(CANDLES.format(symbol=symbol, limit=300))
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
                raise ValueError(f"{symbol} candle response unavailable")
            received = datetime.now(timezone.utc)
            server = datetime.fromtimestamp(int(payload["requestTime"]) / 1000, timezone.utc)
            if abs((received - server).total_seconds()) > 120:
                raise ValueError("candle provider clock disagreement")
            bars = _closed_candles(payload["data"], now)
            if candle_quality(bars, received):
                raise ValueError(f"{symbol} candle data unavailable or stale")
            retained[symbol] = bars
            high, low, close = (np.array([c[k] for c in bars], dtype=float) for k in ("high", "low", "close"))
            atr = float(talib.ATR(high, low, close, timeperiod=14)[-1])
            item = books["instruments"][symbol]
            result[symbol] = minimum_feasibility(symbol, float(item["ticker"]["mark"]), atr, item["contract"])
            result[symbol]["ticker_observed_at_utc"] = item["ticker"]["ticker_observed_at_utc"]
            result[symbol]["last_closed_candle_utc"] = bars[-1]["close_at_utc"]
    completed = datetime.now(timezone.utc)
    for symbol in SYMBOLS:
        quote = datetime.fromisoformat(result[symbol]["ticker_observed_at_utc"])
        if not -timedelta(seconds=5) <= completed - quote <= timedelta(minutes=5) or candle_quality(retained[symbol], completed):
            raise ValueError(f"{symbol} quote or candles became stale")
    return {"kind": "public_snapshot_virtual_feasibility_not_trade", "started_at_utc": now.isoformat(),
            "completed_at_utc": completed.isoformat(), "virtual_equity_usdt": 100, "assumed_leverage": 2,
            "note": "Public fees/mark/ATR example only; no actual account or executable price. Gaps can exceed stop budget.",
            "sources": {"contracts": "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES",
                        "ticker": "https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol={symbol}",
                        "candles": CANDLES}, "markets": result, "order_permissions": False}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2))
    except (ValueError, KeyError, TypeError, OSError, httpx.HTTPError) as exc:
        raise SystemExit(f"Feasibility unavailable: {type(exc).__name__}: {exc}") from exc
