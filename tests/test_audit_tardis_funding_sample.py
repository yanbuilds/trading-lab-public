"""Synthetic ticker records do not represent settlement data."""
import csv
import gzip
import hashlib
import json
import pytest

from experiments.audit_tardis_funding_sample import compare_official, inspect


def test_snapshot_must_be_received_before_its_settlement(tmp_path):
    path = tmp_path / 'synthetic.csv.gz'
    fields = ['exchange', 'symbol', 'timestamp', 'local_timestamp', 'funding_timestamp', 'funding_rate']
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event, received, rate in [(80, 90, '0.001'), (99, 101, '0.002'), (95, 99, '0.003')]:
            writer.writerow(dict(zip(fields, ['bitget-futures', 'BTCUSDT', event, received, 100, rate], strict=True)))
    result = inspect(path, [100, 200])
    assert result[100]['funding_rate_snapshot'] == '0.003'
    assert result[100]['age_at_settlement_us'] == 1
    assert result[200] is None
    with pytest.raises(ValueError):
        inspect(path, [100, 100])


def test_official_comparison_preserves_missing_and_rejects_tampering(tmp_path):
    raw = json.dumps({'code': '00000', 'data': [{'symbol': 'BTCUSDT', 'fundingTime': '100', 'fundingRate': '0.001'}]})
    archive = {'kind': 'readonly_public_btc_funding_page_probe',
               'source_url': 'https://api.bitget.com/api/v2/mix/market/history-fund-rate', 'symbol': 'BTCUSDT',
               'pages': [{'http_status': 200, 'raw_json': raw, 'response_sha256': hashlib.sha256(raw.encode()).hexdigest()}]}
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps(archive), encoding='utf-8')
    result = compare_official({100000: {'funding_rate_snapshot': '0.002'}, 200000: None}, path)
    assert result['100000'] == {'official_rate': '0.001', 'match': False}
    assert result['200000'] == {'official_rate': None, 'match': None}
    archive['pages'][0]['raw_json'] = raw + ' '
    path.write_text(json.dumps(archive), encoding='utf-8')
    with pytest.raises(ValueError):
        compare_official({}, path)
