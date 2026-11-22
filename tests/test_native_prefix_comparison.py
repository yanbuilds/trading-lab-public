import pandas as pd
import pytest

from experiments.compare_native_prefix import compare


def test_compares_cross_cutoff_entry_and_excludes_forced_exit():
    trade = {"pair": "BTC/USDT:USDT", "is_short": False, "open_date": "2026-01-01T01:00:00Z",
             "amount": .01, "open_rate": 100., "leverage": 2., "initial_stop_loss_abs": 90.,
             "initial_stop_loss_ratio": -.1, "close_date": "2026-01-03T00:00:00Z",
             "close_rate": 102., "profit_abs": .02, "funding_fees": 0., "exit_reason": "exit_signal"}
    cut = pd.Timestamp("2026-01-02T00:00:00Z")
    forced = {**trade, "close_date": "2026-01-01T23:00:00Z", "exit_reason": "force_exit"}
    report = compare([trade], [forced], cut)
    assert report["matching_entries"] == 1
    assert report["matching_natural_exits"] == 0
    assert report["prefix_forced_exits_excluded"] == 1
    with pytest.raises(ValueError, match="entry mismatch"):
        compare([trade], [{**forced, "amount": .02}], cut)
