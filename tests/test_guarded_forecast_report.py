"""Synthetic reports: guard failure hides model performance, never invents it."""
import copy
import pytest

from experiments.guarded_forecast_report import apply_guard


def fixtures():
    identity = {'first_cutoff_utc': '2026-11-18T11:00:00+00:00',
                'last_cutoff_utc': '2026-11-18T13:00:00+00:00', 'observations': 3}
    report = {'status': 'all_segments_no_aggregate', 'aggregate_return': None, 'order_permissions': False,
              'segments': [{**identity, 'result': {'candidate': {'initial': 100},
                             'chronos2_shadow': {'status': 'complete', 'result': {'final': 110}}}}]}
    check = {'status': 'post_hoc_local_claim_audit_only', 'audited_at_utc': '2026-11-18T14:00:00+00:00',
             'segments': [{**identity, 'arms': {'chronos2_shadow': {'ok': False}}}]}
    return report, check


def test_failed_or_missing_claim_removes_only_chronos2_result():
    report, check = fixtures()
    result = apply_guard(report, check)
    section = result['segments'][0]
    assert section['result']['chronos2_shadow']['result'] is None
    assert section['result']['candidate'] == {'initial': 100}
    assert section['chronos2_post_hoc_claim_audit'] == 'failed'
    report, check = fixtures()
    check['segments'][0]['arms'].clear()
    assert apply_guard(report, check)['segments'][0]['result']['chronos2_shadow']['result'] is None


def test_pass_retains_result_and_segment_mismatch_fails_closed():
    report, check = fixtures()
    check['segments'][0]['arms']['chronos2_shadow']['ok'] = True
    assert apply_guard(copy.deepcopy(report), check)['segments'][0]['result']['chronos2_shadow']['result']['final'] == 110
    check['segments'][0]['observations'] = 4
    with pytest.raises(ValueError):
        apply_guard(report, check)
