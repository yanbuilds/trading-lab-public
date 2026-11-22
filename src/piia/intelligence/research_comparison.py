"""Fail-closed checks for proposed research comparisons, not a backtester.

Timestamps are recorded assertions, not independent proof of publication.
The result must never be interpreted as an audited or profitable strategy.
"""
from datetime import datetime
import math


def _utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('timezone required')
    return stamp


def audit_evidence_at_decision(items, decision_at):
    """Require timestamped first observation and source publication for each input."""
    decision = _utc(decision_at)
    issues = []
    if not items:
        issues.append('no_evidence_supplied')
    for item in items:
        name = str(item.get('id', 'unknown'))
        try:
            if not item.get('source_url') or not item.get('version_id'):
                raise ValueError('source and version required')
            published = _utc(item['published_at'])
            seen = _utc(item['first_seen_at'])
            if seen < published:
                issues.append(f'{name}:first_seen_before_publication')
            if published > decision or seen > decision:
                issues.append(f'{name}:not_yet_available')
            if item.get('historical_replay_supported') is not True:
                issues.append(f'{name}:historical_replay_not_supported')
        except (KeyError, TypeError, ValueError, AttributeError):
            issues.append(f'{name}:missing_or_invalid_provenance')
    return {'ok': not issues, 'issues': issues,
            'limitation': 'Timestamps and version identifiers do not independently prove source truth or availability.'}


COMPARABLE_FIELDS = ('start', 'end', 'starting_cash', 'market_data_hash', 'cost_model_id',
                     'risk_budget_id', 'execution_model_id', 'funding_model_id')


def audit_comparison(arms):
    """Refuse incomparable/missing arms; cash may have different exposure by design."""
    if len(arms) < 2:
        return {'ok': False, 'issues': ['at_least_two_arms_required']}
    issues = []
    baseline = arms[0]
    if len({arm.get('name') for arm in arms}) != len(arms) or any(not a.get('name') for a in arms):
        issues.append('arm_names_missing_or_duplicate')
    for field in COMPARABLE_FIELDS:
        if any(arm.get(field) is None or arm.get(field) == '' for arm in arms):
            issues.append(f'{field}:missing')
        elif any(arm[field] != baseline[field] for arm in arms[1:]):
            issues.append(f'{field}:mismatch')
    try:
        start, end = _utc(baseline['start']), _utc(baseline['end'])
        if end <= start or not math.isfinite(float(baseline['starting_cash'])) or float(baseline['starting_cash']) <= 0:
            issues.append('invalid_period_or_capital')
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.append('invalid_period_or_capital')
    for arm in arms:
        if arm.get('rules_frozen_before_test') is not True:
            issues.append(f"{arm.get('name', 'unknown')}:rules_not_frozen")
        if arm.get('held_out') is not True:
            issues.append(f"{arm.get('name', 'unknown')}:not_held_out")
    return {'ok': not issues, 'issues': issues,
            'limitation': 'Matching metadata does not independently prove causal data, exposure equivalence, or trading performance.'}


def assess_portfolio(*, equity, positions, max_gross_leverage, max_symbol_fraction, max_stress_loss):
    """Conservative gross notional and supplied per-position stress losses; local research only.

    The caller must supply nonnegative liquidity-inclusive stress loss for each position.
    Correlation is not inferred from a small or missing sample: simultaneous losses are summed.
    """
    def number(x, *, positive=False):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or (x <= 0 if positive else x < 0):
            raise ValueError('invalid risk input')
        return float(x)
    try:
        equity = number(equity, positive=True)
        max_gross_leverage = number(max_gross_leverage, positive=True)
        max_symbol_fraction = number(max_symbol_fraction, positive=True)
        max_stress_loss = number(max_stress_loss, positive=True)
        notionals, stress = {}, 0.0
        for p in positions:
            symbol = p['symbol']
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError('symbol missing')
            n = number(p['notional_usdt'])
            stress += number(p['liquidity_stress_loss_usdt'])
            notionals[symbol] = notionals.get(symbol, 0) + n
        gross = sum(notionals.values()) / equity
        concentration = max(notionals.values(), default=0) / equity
        failures = []
        if gross > max_gross_leverage: failures.append('gross_leverage')
        if concentration > max_symbol_fraction: failures.append('symbol_concentration')
        if stress > max_stress_loss: failures.append('joint_stress_loss')
        return {'ok': not failures, 'issues': failures, 'gross_leverage': gross,
                'max_symbol_notional_to_equity': concentration, 'joint_stress_loss_usdt': stress,
                'limitations': ['Conservative simultaneous losses, not a correlation estimate.',
                                'Stress losses are inputs, not independently verified fills or liquidation.']}
    except (KeyError, TypeError, ValueError):
        return {'ok': False, 'issues': ['missing_or_invalid_portfolio_risk_inputs']}
