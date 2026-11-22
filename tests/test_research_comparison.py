from piia.intelligence.research_comparison import audit_comparison, audit_evidence_at_decision, assess_portfolio


def test_future_and_unverifiable_evidence_are_rejected():
    item = {'id': 'macro', 'source_url': 'https://example.com', 'version_id': 'v1',
            'published_at': '2026-01-01T10:00:00Z', 'first_seen_at': '2026-01-01T11:00:00Z',
            'historical_replay_supported': True}
    assert not audit_evidence_at_decision([item], '2026-01-01T10:30:00Z')['ok']
    assert audit_evidence_at_decision([item], '2026-01-01T11:00:00Z')['ok']
    item['historical_replay_supported'] = False
    assert not audit_evidence_at_decision([item], '2026-01-01T12:00:00Z')['ok']
    item['historical_replay_supported'] = True
    item['first_seen_at'] = '2026-01-01T09:00:00Z'
    assert 'macro:first_seen_before_publication' in audit_evidence_at_decision([item], '2026-01-01T12:00:00Z')['issues']


def test_comparison_requires_equal_costs_and_held_out():
    arm = dict(name='base', start='2026-01-01T00:00:00Z', end='2026-02-01T00:00:00Z',
               starting_cash=100, market_data_hash='a', cost_model_id='fees', risk_budget_id='risk',
               execution_model_id='next', funding_model_id='funding', rules_frozen_before_test=True, held_out=True)
    alt = dict(arm, name='model')
    assert audit_comparison([arm, alt])['ok']
    alt['cost_model_id'] = 'different'
    assert 'cost_model_id:mismatch' in audit_comparison([arm, alt])['issues']
    alt['held_out'] = False
    assert 'model:not_held_out' in audit_comparison([arm, alt])['issues']


def test_portfolio_sums_losses_and_refuses_missing():
    params = dict(equity=100, max_gross_leverage=2, max_symbol_fraction=1, max_stress_loss=5)
    positions = [dict(symbol='BTC', notional_usdt=80, liquidity_stress_loss_usdt=3),
                 dict(symbol='XAU', notional_usdt=80, liquidity_stress_loss_usdt=3)]
    assert 'joint_stress_loss' in assess_portfolio(positions=positions, **params)['issues']
    del positions[1]['liquidity_stress_loss_usdt']
    assert not assess_portfolio(positions=positions, **params)['ok']
