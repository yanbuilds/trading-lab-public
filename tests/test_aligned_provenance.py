from trading_lab.perps.aligned_compare import input_manifest


def test_manifest_tracks_data_bytes_and_marks_missing_funding(tmp_path):
    bars = tmp_path / "bars.csv"
    bars.write_text("date,close\n2023-01-01,1\n", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    first = input_manifest(bars, tmp_path)
    assert first["files"]["btc_funding"] is None
    assert first["files"]["risk.py"]["sha256"]
    bars.write_text("date,close\n2023-01-01,2\n", encoding="utf-8")
    assert input_manifest(bars, tmp_path)["files"]["btc_bars"]["sha256"] != first["files"]["btc_bars"]["sha256"]
