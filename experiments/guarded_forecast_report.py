"""Post-hoc presentation guard over the unchanged precommitted report."""
import argparse
from contextlib import closing
import gc
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

from experiments import audit_forward_shadow_segments as audit, frozen_policy_capture as capture


def apply_guard(report, audit_result):
    if (report.get('status') != 'all_segments_no_aggregate' or report.get('aggregate_return') is not None
            or report.get('order_permissions') is not False
            or audit_result.get('status') != 'post_hoc_local_claim_audit_only'):
        raise ValueError('unexpected report or audit')
    sections, audited = report['segments'], audit_result['segments']
    if len(sections) != len(audited):
        raise ValueError('segment coverage mismatch')
    for section, check in zip(sections, audited, strict=True):
        for field in ('first_cutoff_utc', 'last_cutoff_utc', 'observations'):
            if section[field] != check[field]:
                raise ValueError('segment identity mismatch')
        value = section.get('result')
        if not isinstance(value, dict) or 'chronos2_shadow' not in value:
            continue
        passed = check.get('arms', {}).get('chronos2_shadow', {}).get('ok') is True
        if not passed:
            value['chronos2_shadow'] = {'status': 'unavailable_post_hoc_claim_audit_failed', 'result': None}
        section['chronos2_post_hoc_claim_audit'] = 'passed' if passed else 'failed'
    report['post_hoc_display_guard'] = {
        'status': 'applied',
        'note': 'Local timing and binding only; not independent provenance, model verification, or accuracy.',
        'audited_at_utc': audit_result['audited_at_utc'],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=capture.LEDGER)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.ledger.is_file() or args.ledger.resolve() == args.out.resolve():
        raise ValueError('ledger missing or output overlaps ledger')
    with tempfile.TemporaryDirectory(prefix='guarded-forward-') as temp:
        snapshot, raw_report = Path(temp) / 'ledger.sqlite', Path(temp) / 'report.json'
        with closing(sqlite3.connect(f'{args.ledger.resolve().as_uri()}?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        try:
            audit_result = audit.audit_snapshot(snapshot)
            subprocess.run([sys.executable, '-m', 'experiments.forecast_baseline_report', '--ledger', str(snapshot),
                            '--out', str(raw_report)], check=True, stdout=subprocess.DEVNULL, timeout=180)
            report = json.loads(raw_report.read_text(encoding='utf-8'))
            guarded = apply_guard(report, audit_result)
        finally:
            gc.collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(guarded, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'post_hoc_display_guard_applied', 'segments': len(guarded['segments'])}))


if __name__ == '__main__':
    main()
