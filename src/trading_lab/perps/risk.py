"""Deterministic quantity/risk checks shared with the Freqtrade candidate."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import math

@dataclass(frozen=True)
class Spec:
    symbol: str
    min_qty: float
    qty_step: float
    min_notional: float
    taker: float = .0006
    contract_size: float = 1.

def floor_step(qty, step):
    if not math.isfinite(qty) or not math.isfinite(step) or qty < 0 or step <= 0:
        raise ValueError('Invalid quantity/step')
    q, s = Decimal(str(qty)), Decimal(str(step))
    return float((q/s).to_integral_value(rounding=ROUND_DOWN)*s)


def worst_stop_loss_per_contract(price, stop_distance, spec, *, slippage):
    """Conservative normal-stop loss for either side, including both taker fees.

    Uses the short-side stop above reference price, which has the larger exit
    notional and adverse slippage. Gaps and funding are not bounded here.
    """
    vals = (price, stop_distance, spec.contract_size, spec.taker, slippage)
    if (not all(math.isfinite(x) for x in vals) or price <= 0 or stop_distance <= 0
            or spec.contract_size <= 0 or not 0 <= spec.taker < 1 or not 0 <= slippage < 1):
        raise ValueError('Invalid stop-cost input')
    entry = price * (1 - slippage)
    exit_price = (price + stop_distance) * (1 + slippage)
    loss = spec.contract_size * (exit_price - entry + spec.taker * (entry + exit_price))
    if not math.isfinite(loss) or loss <= 0:
        raise ValueError('Invalid stop-cost result')
    return loss

def size_order(equity, free_cash, price, stop_distance, spec, *, leverage=2., risk_fraction=.01,
               cash_floor=.20, max_margin_fraction=.50, slippage=.0005):
    """Contracts, NOT base units. Never round up to meet a minimum.

    Stop budget includes both fees and assumed entry/exit slippage. Example leverage
    range 1..3; 5% maintenance stress assumption is NOT venue liquidation modelling.
    """
    vals=[equity,free_cash,price,stop_distance,leverage,risk_fraction, cash_floor,
          max_margin_fraction,slippage,spec.min_qty,spec.qty_step,spec.min_notional,
          spec.taker,spec.contract_size]
    if (not all(math.isfinite(x) for x in vals)
            or min(equity,free_cash,price,stop_distance,leverage,risk_fraction,
                   spec.min_qty,spec.qty_step,spec.contract_size)<=0 or spec.min_notional<0):
        return 0., 'invalid_input'
    if (not 1<=leverage<=3 or risk_fraction>.02 or not 0<=cash_floor<1 or not 0<max_margin_fraction<=.5
            or not 0<=slippage<1 or not 0<=spec.taker<1):
        return 0., 'risk_bounds'
    try:
        per_contract_loss = worst_stop_loss_per_contract(price, stop_distance, spec, slippage=slippage)
    except ValueError:
        return 0., 'invalid_input'
    if per_contract_loss / (price * spec.contract_size) >= 1/leverage-.05:
        return 0., 'stop_outside_stress_margin'
    unit=spec.contract_size
    qty_risk=equity*risk_fraction/per_contract_loss
    available=min(equity*max_margin_fraction,max(0.,free_cash-equity*cash_floor))
    cash_cost=max(1/leverage+spec.taker+slippage, (1+slippage)*(1/leverage+spec.taker))
    qty_cash=available/(price*unit*cash_cost)
    qty=floor_step(min(qty_risk,qty_cash),spec.qty_step)
    if qty<spec.min_qty or qty*price*unit<spec.min_notional:
        return 0., 'minimum_order_exceeds_budget'
    return qty, 'ok'

def exit_pnl(side, qty, entry, exit_price, spec):
    if side not in (-1,1): raise ValueError('side must be -1/+1')
    return side*qty*spec.contract_size*(exit_price-entry)

def stop_fill(side, stop, open_price, low, high):
    """Stops precede signal exits; gaps get the less favourable open price."""
    if side==1 and low<=stop: return min(stop,open_price)
    if side==-1 and high>=stop: return max(stop,open_price)
    return None

def assert_simulation(config):
    if config.get('dry_run') is not True:
        raise ValueError('LIVE DISABLED: dry_run must be true')
    exchange=config.get('exchange',{})
    if any(exchange.get(k) for k in ('key','secret','password','api_key')):
        raise ValueError('Credentials are forbidden in this simulation integration')
    if config.get('margin_mode')!='isolated' or config.get('max_open_trades')!=1:
        raise ValueError('Only isolated / one global open position supported')
