import io

import pandas as pd
import pytest
import zstandard

from experiments.funding_archive_audit import decode_archive, last_pre_settlement_quote


def test_only_last_timely_pre_settlement_quote_with_matching_settlement_is_used():
    settlement = pd.Timestamp("2026-09-02T16:00:00Z")
    ns, ms = settlement.value, int(settlement.timestamp() * 1000)
    frame = pd.DataFrame({"received_time": [ns - 300_000_000, ns - 100_000_000, ns + 1, ns - 50_000_000],
                          "funding_rate": ["0.0001", "0.0002", "0.9", "0.8"],
                          "next_funding_time": [ms, ms, ms, ms + 28_800_000]})
    result = last_pre_settlement_quote(frame, settlement)
    assert result["proxy_rate"] == pytest.approx(0.0002)
    assert result["lag_ms"] == pytest.approx(100.0, abs=0.001)
    assert "not_official" in result["provenance"]


def test_stale_or_missing_quote_fails_closed():
    settlement = pd.Timestamp("2026-09-02T16:00:00Z")
    frame = pd.DataFrame({"received_time": [settlement.value - 2_000_000_000],
                          "funding_rate": ["0.1"], "next_funding_time": [int(settlement.timestamp() * 1000)]})
    with pytest.raises(ValueError, match="too old"):
        last_pre_settlement_quote(frame, settlement)
    frame["next_funding_time"] += 28_800_000
    with pytest.raises(ValueError, match="no quote"):
        last_pre_settlement_quote(frame, settlement)


def test_plain_and_zstd_wrapped_parquet_and_corrupt_input():
    f = pd.DataFrame({"received_time": [1], "funding_rate": ["0.1"], "next_funding_time": [2]})
    buffer = io.BytesIO()
    f.to_parquet(buffer, index=False)
    data = buffer.getvalue()
    pd.testing.assert_frame_equal(decode_archive(data), f)
    pd.testing.assert_frame_equal(decode_archive(zstandard.ZstdCompressor().compress(data)), f)
    with pytest.raises(ValueError, match="complete parquet"):
        decode_archive(b"not parquet")
