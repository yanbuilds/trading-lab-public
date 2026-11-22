"""Synthetic fixture responses; no provider data or profitability claims."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from experiments import audit_btc_public_overlap as audit


def test_saved_pages_detect_gaps_revisions_duplicates_and_corruption():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    index = pd.date_range(start, end, freq='h', inclusive='left')
    local = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.}, index=index)
    rows = [[str(int(t.timestamp() * 1000)), '100', '101', '99', '100', '1'] for t in index]
    def page(data):
        raw = json.dumps({'code': '00000', 'data': data})
        return {'start_utc': start.isoformat(), 'end_exclusive_utc': end.isoformat(),
                'raw_response': raw, 'sha256': audit.sha(raw.encode())}
    good = audit.compare_pages(local, [page(rows)], start, end)
    assert good['status'] == 'matched' and good['expected_hours'] == 2
    missing = audit.compare_pages(local, [page(rows[:1])], start, end)
    assert missing['status'] == 'incomplete_or_mismatched' and missing['missing_provider_hours'] == 1
    failed = audit.compare_pages(local, [{'error': 'ReadTimeout'}], start, end)
    assert failed['status'] == 'incomplete_or_mismatched' and failed['failed_pages'] == 1
    assert failed['missing_provider_hours'] == 2
    revised = [row.copy() for row in rows]
    revised[0][4] = '100.1'
    changed = audit.compare_pages(local, [page(revised)], start, end)
    assert changed['status'] == 'incomplete_or_mismatched' and changed['mismatches']['close'] == 1
    repeated = audit.compare_pages(local, [page(rows + rows[:1])], start, end)
    assert repeated['status'] == 'incomplete_or_mismatched' and repeated['duplicate_provider_rows'] == 1
    bad = page(rows)
    bad['raw_response'] += ' '
    with pytest.raises(ValueError, match='hash mismatch'):
        audit.compare_pages(local, [bad], start, end)
