"""Inspect a third-party ticker sample, not authoritative settled funding."""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path


def inspect(path, settlements_us):
    """Find latest received pre-settlement snapshot for each specified instant."""
    targets = {int(t): None for t in settlements_us}
    if len(targets) != len(settlements_us) or not targets:
        raise ValueError('settlement targets must be nonempty and unique')
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('exchange') != 'bitget-futures' or row.get('symbol') != 'BTCUSDT':
                raise ValueError('unexpected instrument')
            if not row.get('funding_timestamp') or not row.get('funding_rate'):
                continue
            target = int(row['funding_timestamp'])
            if target not in targets:
                continue
            event, received = int(row['timestamp']), int(row['local_timestamp'])
            try:
                rate = Decimal(row['funding_rate'])
            except InvalidOperation as exc:
                raise ValueError('invalid rate') from exc
            if not rate.is_finite():
                raise ValueError('invalid rate')
            if event >= target or received >= target:
                continue
            existing = targets[target]
            if existing is None or (received, event) > (existing['received_us'], existing['event_us']):
                targets[target] = {'funding_rate_snapshot': str(rate), 'event_us': event, 'received_us': received,
                                   'age_at_settlement_us': target - received}
    return targets


def compare_official(snapshots, path):
    """Compare with a previously archived Bitget v2 public response, never fill gaps."""
    archive = json.loads(Path(path).read_text(encoding='utf-8'))
    if (archive.get('kind') != 'readonly_public_btc_funding_page_probe'
            or archive.get('source_url') != 'https://api.bitget.com/api/v2/mix/market/history-fund-rate'
            or archive.get('symbol') != 'BTCUSDT'):
        raise ValueError('unexpected official archive')
    official = {}
    for page in archive['pages']:
        raw = page['raw_json']
        if (page['http_status'] != 200 or hashlib.sha256(raw.encode('utf-8')).hexdigest() != page['response_sha256']):
            raise ValueError('official response integrity failure')
        payload = json.loads(raw)
        if payload.get('code') != '00000' or not isinstance(payload.get('data'), list):
            raise ValueError('official response unavailable')
        for row in payload['data']:
            if row['symbol'] != 'BTCUSDT':
                raise ValueError('unexpected official instrument')
            timestamp, rate = int(row['fundingTime']) * 1000, Decimal(row['fundingRate'])
            if not rate.is_finite() or (timestamp in official and official[timestamp] != rate):
                raise ValueError('invalid or conflicting official rates')
            official[timestamp] = rate
    return {str(t): {'official_rate': str(official[t]) if t in official else None,
                     'match': (Decimal(item['funding_rate_snapshot']) == official[t]
                               if item is not None and t in official else None)}
            for t, item in snapshots.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('settlements', nargs='+', help='UTC settlement timestamps, e.g. 2025-12-01T08:00:00+00:00')
    parser.add_argument('--official-pages', type=Path, help='Local archived Bitget v2 public responses')
    args = parser.parse_args()
    targets = []
    for raw in args.settlements:
        value = datetime.fromisoformat(raw)
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError('UTC required')
        targets.append(int(value.timestamp() * 1_000_000))
    result = {'file_sha256': hashlib.sha256(args.file.read_bytes()).hexdigest(),
              'snapshots_by_settlement_us': inspect(args.file, targets),
              'status': 'third_party_ticker_snapshots_not_verified_settlement',
              'limitations': ['Funding snapshot may differ from the rate actually settled.',
                              'This sample does not establish complete historical coverage.']}
    if args.official_pages:
        result['official_archive_sha256'] = hashlib.sha256(args.official_pages.read_bytes()).hexdigest()
        result['official_comparison'] = compare_official(result['snapshots_by_settlement_us'], args.official_pages)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
