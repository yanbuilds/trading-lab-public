"""Read-only comparison against Bitget public historical candles, not a strategy test."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import httpx
import pandas as pd

URL = 'https://api.bitget.com/api/v2/mix/market/history-candles'
START = datetime(2025, 11, 16, 6, tzinfo=timezone.utc) - timedelta(hours=300)
END = datetime(2026, 11, 16, 6, tzinfo=timezone.utc)
COLS = ('open', 'high', 'low', 'close', 'volume')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


async def fetch_pages(start=START, end=END):
    """Fetch fixed 96-hour pages with bounded concurrency; failures are recorded."""
    windows = []
    while start < end:
        nxt = min(start + timedelta(hours=96), end)
        windows.append((start, nxt))
        start = nxt
    pages = []
    async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
        async def fetch(window):
            first, last = window
            query = {'productType': 'USDT-FUTURES', 'symbol': 'BTCUSDT', 'granularity': '1H', 'limit': '100',
                     'startTime': int(first.timestamp() * 1000), 'endTime': int(last.timestamp() * 1000)}
            page = {'start_utc': first.isoformat(), 'end_exclusive_utc': last.isoformat(), 'query': query}
            try:
                response = await client.get(URL, params=query)
                received = datetime.now(timezone.utc).isoformat()
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get('code') != '00000' or not isinstance(payload.get('data'), list):
                    raise ValueError('provider response invalid')
                page.update(received_at_utc=received, http_status=response.status_code,
                            sha256=sha(response.content), raw_response=response.text)
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                page['error'] = type(exc).__name__
            return page
        for offset in range(0, len(windows), 2):
            pages.extend(await asyncio.gather(*(fetch(w) for w in windows[offset:offset + 2])))
            await asyncio.sleep(.2)
    return pages


def compare_pages(local: pd.DataFrame, pages: list[dict], start=START, end=END) -> dict:
    """Reparse saved responses and compare exact timestamps and numeric OHLCV."""
    if local.index.tz is None or not local.index.is_unique or not local.index.is_monotonic_increasing:
        raise ValueError('local timestamps must be unique, ordered and timezone-aware')
    if not set(COLS).issubset(local.columns):
        raise ValueError('local OHLCV columns missing')
    expected = pd.date_range(start, end, freq='h', inclusive='left')
    seen = set()
    duplicate = out_of_window = missing_local = 0
    mismatch = dict.fromkeys(COLS, 0)
    maxdiff = dict.fromkeys(COLS, 0.0)
    failed = 0
    for page in pages:
        if 'error' in page:
            failed += 1
            continue
        raw = page['raw_response'].encode('utf-8')
        if sha(raw) != page['sha256']:
            raise ValueError('response hash mismatch')
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get('code') != '00000' or not isinstance(payload.get('data'), list):
            raise ValueError('saved response invalid')
        first, last = pd.Timestamp(page['start_utc']), pd.Timestamp(page['end_exclusive_utc'])
        if first.tzinfo is None or last.tzinfo is None or not start <= first < last <= end:
            raise ValueError('page bounds invalid')
        for row in payload['data']:
            stamp = pd.Timestamp(int(row[0]), unit='ms', tz='UTC')
            if not first <= stamp < last:
                out_of_window += 1
                continue
            if stamp in seen:
                duplicate += 1
                continue
            seen.add(stamp)
            if stamp not in local.index:
                missing_local += 1
                continue
            for i, column in enumerate(COLS, 1):
                actual, saved = float(row[i]), float(local.loc[stamp, column])
                if not math.isfinite(actual) or not math.isfinite(saved):
                    raise ValueError('non-finite value')
                diff = abs(actual - saved)
                maxdiff[column] = max(maxdiff[column], diff)
                if not math.isclose(actual, saved, rel_tol=1e-9, abs_tol=1e-9):
                    mismatch[column] += 1
    missing = expected.difference(pd.DatetimeIndex(sorted(seen)))
    ok = not (failed or duplicate or out_of_window or missing_local or len(missing) or any(mismatch.values()))
    return {'status': 'matched' if ok else 'incomplete_or_mismatched', 'expected_hours': len(expected),
            'unique_provider_hours': len(seen), 'missing_provider_hours': len(missing),
            'first_missing_utc': missing[0].isoformat() if len(missing) else None,
            'missing_local_hours': missing_local, 'duplicate_provider_rows': duplicate,
            'out_of_window_rows': out_of_window, 'failed_pages': failed,
            'mismatches': mismatch, 'max_absolute_difference': maxdiff,
            'numeric_tolerance': {'relative': 1e-9, 'absolute': 1e-9}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('artifacts/btc_bitget_overlap_one_year.json'))
    args = parser.parse_args()
    if args.bars.resolve() == args.out.resolve():
        raise ValueError('output cannot overwrite market data')
    before = sha(args.bars.read_bytes())
    local = pd.read_csv(args.bars, dtype={'date': 'string'})
    if 'date' not in local or not local['date'].str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True, na=False).all():
        raise ValueError('local timestamps must include a UTC offset')
    local = local.set_index('date')
    local.index = pd.to_datetime(local.index, utc=True, errors='raise')
    pages = asyncio.run(fetch_pages())
    summary = compare_pages(local, pages)
    if sha(args.bars.read_bytes()) != before:
        raise ValueError('local input changed during audit')
    result = {'kind': 'public_historical_comparison_not_independent_provenance',
              'order_permissions': False, 'source_url': URL, 'input_sha256': before,
              'start_utc': START.isoformat(), 'end_exclusive_utc': END.isoformat(),
              'summary': summary, 'pages': pages,
              'limitations': ['Current public historical data may be revised; historical availability and source are not independently proven.',
                              'Only BTC OHLCV checked; funding, actual fills and liquidation remain unverified.',
                              'Previously viewed data, not a held-out strategy test.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'kind': result['kind'], 'order_permissions': False, 'summary': summary}))


if __name__ == '__main__':
    main()
