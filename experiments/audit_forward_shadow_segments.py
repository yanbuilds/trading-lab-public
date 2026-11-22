"""Post-hoc local timing/binding audit; it is not a preregistered return report."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sqlite3
import tempfile

from experiments import frozen_policy_capture as capture, prospective_paper_report, segmented_forward_report_v2
from piia.intelligence.shadow_claim_audit import audit_claims


def audit_snapshot(path, *, now=None):
    now = now or datetime.now(timezone.utc)
    records = prospective_paper_report.read_observations(path)
    segments, excluded = segmented_forward_report_v2.partition(records)
    results = []
    for segment in segments:
        cutoffs = [row['cutoff_utc'] for _, row in segment]
        arms = {arm: audit_claims(path, arm, now=now, cutoffs=cutoffs) for arm in
                ('ai_shadow', 'chronos_shadow', 'chronos2_shadow')}
        results.append({'first_cutoff_utc': cutoffs[0], 'last_cutoff_utc': cutoffs[-1],
                        'observations': len(cutoffs), 'arms': arms})
    return {'status': 'post_hoc_local_claim_audit_only', 'audited_at_utc': now.isoformat(),
            'excluded_pre_activation_observations': excluded, 'segments': results,
            'order_permissions': False,
            'limitations': ['Local clock and database do not independently prove provenance or timestamps.',
                            'This audit is post hoc, not a precommitted model or return evaluation.',
                            'A passing audit does not prove model output, accuracy, or profitability.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=capture.LEDGER)
    args = parser.parse_args()
    if not args.ledger.is_file():
        raise FileNotFoundError('prospective ledger missing')
    with tempfile.TemporaryDirectory(prefix='shadow-audit-') as temp:
        snapshot = Path(temp) / 'ledger.sqlite'
        with closing(sqlite3.connect(f'{args.ledger.resolve().as_uri()}?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        try:
            result = audit_snapshot(snapshot)
        finally:
            gc.collect()  # Release SQLite handles before Windows removes the snapshot.
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
