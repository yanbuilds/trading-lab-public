from datetime import datetime, timezone
import json

import httpx
import pytest

from piia.intelligence import macro_data


NOW = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)


def test_alfred_vintage_skips_missing_preserves_provenance():
    raw = b"observation_date,DGS10_20261114\n2026-11-10,4.9\n2026-11-11,\n2026-11-12,5.1\n"
    result = macro_data.parse_vintage_csv(raw, "DGS10", "2026-11-14", NOW)
    assert result["value"] == 5.1
    assert result["observation_date"] == "2026-11-12"
    assert result["vintage_date"] == "2026-11-14"
    assert result["stale"] is False
    assert len(result["source_csv_sha256"]) == 64
    assert "not for historical replay" in result["time_limit"]


@pytest.mark.parametrize("raw,vintage", [
    (b"observation_date,DGS10_20261115\n2026-11-12,5.1\n", "2026-11-15"),
    (b"observation_date,DGS10_20261114\n2026-11-15,5.1\n", "2026-11-14"),
    (b"observation_date,DGS10_20261114\n2026-11-12,nan\n", "2026-11-14"),
    (b"observation_date,DGS10_20261114\n2026-11-12,5.1\n2026-11-12,5.2\n", "2026-11-14"),
    (b"observation_date,DGS10_20261113\n2026-11-12,5.1\n", "2026-11-14"),
])
def test_alfred_rejects_unsafe_vintage_or_invalid_data(raw, vintage):
    with pytest.raises(ValueError):
        macro_data.parse_vintage_csv(raw, "DGS10", vintage, NOW)


def test_partial_fetch_is_explicit_and_does_not_substitute_missing(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params):
            assert url == macro_data.BASE
            if params["id"] == "DFII10":
                raise httpx.ConnectError("offline")
            value = f"observation_date,{params['id']}_20261114\n2026-11-12,4.2\n"
            return httpx.Response(200, content=value.encode(), request=httpx.Request("GET", url))
    monkeypatch.setattr(macro_data.httpx, "Client", FakeClient)
    out = macro_data.fetch_alfred_context(now=NOW)
    assert out["complete"] is False
    assert out["errors"]["DFII10"] == "ConnectError"
    assert "DFII10" not in out["series"]
    assert out["historical_replay_supported"] is False
    json.dumps(out)


def test_extended_fetch_keeps_missing_series_explicit(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params):
            assert url == macro_data.BASE
            if params["id"] == "WDTGAL":
                raise httpx.ConnectError("offline")
            value = f"observation_date,{params['id']}_20261114\n2026-11-12,4.2\n"
            return httpx.Response(200, content=value.encode(), request=httpx.Request("GET", url))
    monkeypatch.setattr(macro_data.httpx, "Client", FakeClient)
    out = macro_data.fetch_alfred_extended_context(now=NOW, snapshot_dir=tmp_path)
    assert out["historical_replay_supported"] is False
    assert out["complete"] is False
    assert "WDTGAL" not in out["series"]
    assert out["errors"]["WDTGAL"] == "ConnectError"
    assert "DTWEXBGS" in out["series"]
    assert out["series"]["DTWEXBGS"]["unit"] == "index"
    item = out["series"]["DTWEXBGS"]
    assert macro_data.verify_vintage_snapshot(item, tmp_path)["ok"]
    assert not macro_data.verify_vintage_snapshot(item, tmp_path)["historical_replay_supported"]
    (tmp_path / item["raw_snapshot_file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        macro_data.verify_vintage_snapshot(item, tmp_path)
    assert not (tmp_path / "WDTGAL_2026-11-14.csv").exists()
    json.dumps(out)


def test_http_failure_records_status_without_response_content_or_substitution(monkeypatch):
    from piia.intelligence.chat_context import compact_macro
    from piia.intelligence.research_cycle import _critical_blockers
    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params):
            request = httpx.Request("GET", url)
            if params["id"] == "RRPONTSYD":
                return httpx.Response(503, text="provider-private-body", request=request)
            value = f"observation_date,{params['id']}_20261114\n2026-11-12,4.2\n"
            return httpx.Response(200, content=value.encode(), request=request)
    monkeypatch.setattr(macro_data.httpx, "Client", FakeClient)
    out = macro_data.fetch_alfred_extended_context(now=NOW)
    assert out["errors"]["RRPONTSYD"] == "HTTPStatusError"
    assert out["error_details"]["RRPONTSYD"] == {"type": "HTTPStatusError", "http_status": 503}
    assert "RRPONTSYD" not in out["series"] and not out["complete"]
    assert "provider-private-body" not in json.dumps(out)
    compact = json.loads(compact_macro(out)["raw"])
    assert compact["error_details"]["RRPONTSYD"]["http_status"] == 503
    blockers = _critical_blockers({"snapshot_ok": True, "instruments": {}, "missing_datasets": []},
                                  {"weights_ran": False}, out)
    assert "macro_series_missing:RRPONTSYD" in blockers["packet"]
