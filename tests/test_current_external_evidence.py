"""Synthetic provider responses, never market fixtures for live research."""
from datetime import datetime, timedelta, timezone
import httpx
import pytest
from piia.intelligence.etf_flows import EXPECTED_FUNDS
from piia.intelligence.etf_tftc import fetch_tftc_etf
from piia.intelligence.fear_greed import fetch_fear_greed

NOW = datetime(2026, 11, 17, 5, tzinfo=timezone.utc)


def client(payload):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload, request=req)))


def etf():
    funds = {x: 0 for x in EXPECTED_FUNDS}
    funds['IBIT'] = -1000000
    return {'updatedThrough': '2026-11-13', 'dateModified': '2026-11-13T22:00:00Z',
            'latestSession': {'date': '2026-11-13', 'netFlowUsd': -1000000, 'perEtfUsd': funds}}


def test_tftc_complete_and_reconciled(tmp_path):
    import hashlib
    out = fetch_tftc_etf(now=NOW, client=client(etf()), snapshot_dir=tmp_path)
    assert out['usable_for_btc_research']
    assert out['latest_complete_row']['table_total_millions_usd'] == -1
    assert out['latest_complete_age_calendar_days'] == 4
    assert out['latest_exchange_session_verified'] is False
    raw = (tmp_path / out['raw_snapshot_file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == out['raw_sha256']


@pytest.mark.parametrize('case', ['missing', 'null', 'mismatch', 'future', 'preclose'])
def test_tftc_refuses_bad_or_incomplete(case):
    p = etf()
    if case == 'missing': del p['latestSession']['perEtfUsd']['IBIT']
    if case == 'null': p['latestSession']['perEtfUsd']['IBIT'] = None
    if case == 'mismatch': p['latestSession']['netFlowUsd'] = 0
    if case == 'future': p['dateModified'] = '2026-11-18T22:00:00Z'
    if case == 'preclose': p['dateModified'] = '2026-11-13T19:00:00Z'
    assert not fetch_tftc_etf(now=NOW, client=client(p))['usable_for_btc_research']


def test_sentiment_rejects_future_and_stale():
    row = {'value': '12', 'value_classification': 'Extreme Fear', 'timestamp': str(int(NOW.timestamp()))}
    assert fetch_fear_greed(now=NOW, client=client({'data': [row]}))['usable_for_research']
    row['timestamp'] = str(int(NOW.timestamp()) + 1)
    assert not fetch_fear_greed(now=NOW, client=client({'data': [row]}))['usable_for_research']


def test_sentiment_snapshot_reproduces_and_rejects_tampering_and_staleness(tmp_path):
    from copy import deepcopy
    from piia.intelligence.fear_greed import verify_fear_greed_snapshot
    row = {'value': '12', 'value_classification': 'Extreme Fear', 'timestamp': str(int(NOW.timestamp()))}
    out = fetch_fear_greed(now=NOW, client=client({'data': [row]}), snapshot_dir=tmp_path)
    assert verify_fear_greed_snapshot(out, tmp_path, NOW)['ok']
    altered = deepcopy(out)
    altered['observations'][0]['value'] = 50
    with pytest.raises(ValueError):
        verify_fear_greed_snapshot(altered, tmp_path, NOW)
    with pytest.raises(ValueError, match='stale'):
        verify_fear_greed_snapshot(out, tmp_path, NOW + timedelta(days=3))
    (tmp_path / 'fear_greed.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='hash'):
        verify_fear_greed_snapshot(out, tmp_path, NOW)


def test_sentiment_rejects_duplicate_or_unsorted_observations():
    row = {'value': '12', 'value_classification': 'Extreme Fear', 'timestamp': str(int(NOW.timestamp()))}
    previous = {**row, 'timestamp': str(int(NOW.timestamp()) - 86400)}
    assert not fetch_fear_greed(now=NOW, client=client({'data': [row, row]}))['usable_for_research']
    assert not fetch_fear_greed(now=NOW, client=client({'data': [previous, row]}))['usable_for_research']
    row['timestamp'] = str(int(NOW.timestamp()) - 3*86400)
    assert not fetch_fear_greed(now=NOW, client=client({'data': [row]}))['usable_for_research']


def test_sentiment_records_completion_after_response(monkeypatch):
    from piia.intelligence import fear_greed
    class FakeClock(datetime):
        calls = 0
        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return NOW + timedelta(seconds=cls.calls - 1)
    monkeypatch.setattr(fear_greed, "datetime", FakeClock)
    row = {'value': '12', 'value_classification': 'Extreme Fear', 'timestamp': str(int(NOW.timestamp()))}
    out = fear_greed.fetch_fear_greed(client=client({'data': [row]}))
    assert out['retrieved_at_utc'] == NOW.isoformat()
    assert out['completed_at_utc'] == (NOW + timedelta(seconds=1)).isoformat()
