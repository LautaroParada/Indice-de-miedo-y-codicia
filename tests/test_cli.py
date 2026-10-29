import importlib
import json

import pytest

import indice
from imc.config import Methodology
from imc.engine import calculate_index
from imc.report import export_run


def test_import_performs_no_api_calls(monkeypatch):
    import bcch

    def forbidden(*args, **kwargs):
        pytest.fail("Network client created at import")

    monkeypatch.setattr(bcch, "BancoCentralDeChile", forbidden)
    importlib.reload(indice)


def test_catalog_works_without_token(monkeypatch, capsys):
    monkeypatch.delenv("BCCH_API_KEY", raising=False)
    assert indice.main(["catalogo", "dólar observado", "--frecuencia", "diaria"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert any(item["code"] == "F073.TCO.PRE.Z.D" for item in result["results"])
    assert result["catalog"]["unique_series"] == 25291


def test_missing_token_has_readable_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("BCCH_API_KEY", raising=False)
    assert indice.main(["calcular", "--env-file", str(tmp_path / "absent.env")]) == 2
    assert "BCCH_API_KEY" in capsys.readouterr().err


def test_inverted_dates_fail_before_network(capsys):
    assert indice.main(["calcular", "--inicio", "2026-02-01", "--fin", "2026-01-01"]) == 2
    assert "posterior" in capsys.readouterr().err


def test_export_contains_reproducibility_and_source_ages(frames, tmp_path):
    result = calculate_index(frames)
    export_run(
        result,
        frames,
        {},
        Methodology(),
        output=tmp_path,
        start="2020-01-01",
        end="2024-01-01",
        plot=False,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["sha256"]) == 5
    assert metadata["methodology_version"] == "2.0.0"
    assert len(metadata["latest_source_dates"]) == 4
    assert "token" not in json.dumps(metadata).lower()
