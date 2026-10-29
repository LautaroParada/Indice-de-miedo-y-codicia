"""Temporal and provenance guarantees for the retrospective research prototype."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from research import analyze as research


def test_percentile_reference_excludes_current_observation():
    series = pd.Series([0.0, 1.0, 2.0, 1.0])
    result = research.percentile_past(series, window=2, minimum=2)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 100.0  # Both previous observations are smaller.
    assert result.iloc[3] == 25.0  # Prior [1, 2]: one tie, no smaller value.


def test_percentiles_are_unchanged_when_future_is_added():
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    past = pd.Series(np.sin(np.arange(120) * 0.37), index=dates)
    future = pd.Series([1e12, -1e12], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    before = research.percentile_past(past)
    after = research.percentile_past(pd.concat([past, future])).loc[past.index]
    assert_series_equal(before, after)


def test_missing_score_stays_missing_and_does_not_count_as_history():
    result = research.percentile_past(
        pd.Series([1.0, np.nan, 2.0, np.nan, 3.0]), window=4, minimum=2
    )
    assert pd.isna(result.iloc[2])  # Only one valid past observation.
    assert pd.isna(result.iloc[3])  # Missing current value is not replaced.
    assert result.iloc[4] == 100.0


def test_incomplete_component_never_reweights_mean():
    values = pd.DataFrame({"a": [20.0, 20.0, np.nan], "b": [80.0, np.nan, np.nan]})
    result = research.complete_mean(values, ["a", "b"])
    assert result.iloc[0] == 50.0
    assert result.iloc[1:].isna().all()


def test_regression_rejects_overlapping_development_and_test_periods():
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    x = pd.DataFrame({"x": np.arange(120)}, index=dates)
    y = pd.Series(np.arange(120), index=dates)
    with pytest.raises(ValueError, match="superponerse"):
        research.ridge_reconstruction(x, y, train_end="2018-12-31", test_start="2018-12-31")


def test_test_values_cannot_change_fitted_scaling_or_coefficients(monkeypatch):
    dates = pd.date_range("2010-01-31", periods=132, freq="ME")
    n = np.arange(len(dates))
    base = pd.DataFrame({"a": np.sin(n * 0.3), "b": np.cos(n * 0.17)}, index=dates)
    target = pd.Series(3 * base.a - 2 * base.b, index=dates)
    captured = []
    original_solve = np.linalg.solve

    def capture(matrix, rhs):
        captured.append((matrix.copy(), rhs.copy()))
        return original_solve(matrix, rhs)

    monkeypatch.setattr(research.np.linalg, "solve", capture)
    first = research.ridge_reconstruction(
        base, target, train_end="2016-12-31", test_start="2017-01-01"
    )
    altered_base, altered_target = base.copy(), target.copy()
    altered_base.loc["2017":] *= 10000
    altered_target.loc["2017":] += 10000
    second = research.ridge_reconstruction(
        altered_base, altered_target, train_end="2016-12-31", test_start="2017-01-01"
    )
    assert first["status"] == second["status"] == "ok"
    assert first["train_n"] == 84 and first["test_n"] == 48
    np.testing.assert_allclose(captured[0][0], captured[1][0])
    np.testing.assert_allclose(captured[0][1], captured[1][1])
    assert first["r2_test"] != second["r2_test"]


def test_regression_reports_insufficient_evidence_instead_of_fit():
    dates = pd.date_range("2018-01-31", periods=24, freq="ME")
    base = pd.DataFrame({"a": np.arange(24)}, index=dates)
    result = research.ridge_reconstruction(base, base.a)
    assert result == {"status": "insufficient", "train_n": 12, "test_n": 12}


@pytest.fixture
def research_snapshot(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    months = pd.date_range("2010-01-01", "2026-10-01", freq="MS")
    days = pd.bdate_range("2010-01-01", "2026-10-15")
    aliases = [
        "usd",
        "copper",
        "idie",
        "turnover_value",
        "market_cap",
        "interbank",
        "tpm",
        "nominal2",
        "nominal5",
        "real5",
        "m1",
        "m2",
        "eee_fx12",
        "ipsa",
        "price_book",
        "ipec",
        "imce_nonmining",
        "bank_spread",
        "spot_position",
    ]
    candidates, records = [], []

    def add(alias, frequency, dates, values):
        code = "TEST." + alias
        frame = pd.DataFrame({"value": values, "status_code": "OK"}, index=dates)
        frame.index.name = "date"
        path = raw / (alias + ".csv")
        frame.to_csv(path)
        candidates.append(
            {
                "alias": alias,
                "code": code,
                "frequency": frequency,
                "transform": "level",
                "block": "test",
                "sign": 0,
            }
        )
        records.append(
            {
                "alias": alias,
                "code": code,
                "status": "ok",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    for i, alias in enumerate(aliases):
        n = np.arange(len(months))
        values = 100 + i + 10 * np.sin(n * (0.17 + i * 0.011)) + np.cos(n * 0.47)
        add(alias, "Mensual", months, values)
    n = np.arange(len(days))
    rate = 5 + np.sin(n * 0.03) + (0.1 + 0.05 * np.sin(n * 0.003)) * np.sin(n * 0.7)
    add("nominal10", "Diaria", days, rate)
    quarters = pd.date_range("2010-01-01", "2026-10-01", freq="QS")
    add("credit_supply", "Trimestral", quarters, np.arange(len(quarters)))
    manifest = {"start": "2010-01-01", "end": "2026-10-15", "series": records}
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def baseline(_frames):
        n = np.arange(len(days))
        return pd.DataFrame(
            {
                "score_a": 50 + 20 * np.sin(n * 0.02),
                "score_b": 50 + 20 * np.cos(n * 0.03),
                "complete": True,
            },
            index=days,
        )

    monkeypatch.setattr(research, "calculate_index", baseline)
    return candidates, raw, manifest


def test_incomplete_month_excluded_and_quarters_not_repeated(research_snapshot, tmp_path):
    candidates, raw, _manifest = research_snapshot
    output = tmp_path / "analysis"
    summary = research.analyze(candidates, raw, output)
    monthly = pd.read_csv(output / "monthly_levels.csv", index_col="period_end", parse_dates=True)
    quarterly = pd.read_csv(output / "quarterly.csv", index_col="period_end", parse_dates=True)
    assert summary["last_complete_month"] == "2026-09-30"
    assert monthly.index.max() == pd.Timestamp("2026-09-30")
    assert "credit_supply" not in monthly
    assert len(quarterly) == 67
    assert quarterly.index.month.isin([3, 6, 9, 12]).all()
    assert quarterly.index.to_period("Q").is_unique
    assert quarterly.index.max() == pd.Timestamp("2026-09-30")


def test_failed_optional_source_ignores_stale_csv(research_snapshot, tmp_path):
    candidates, raw, manifest = research_snapshot
    stale = {
        "alias": "stale_optional",
        "code": "TEST.stale",
        "frequency": "Mensual",
        "transform": "level",
        "block": "context",
        "sign": 0,
    }
    candidates.append(stale)
    (raw / "stale_optional.csv").write_text(
        "date,value,status_code\n2026-01-01,999,OK\n", encoding="utf-8"
    )
    manifest["series"].append(
        {"alias": "stale_optional", "code": "TEST.stale", "status": "api_error"}
    )
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "analysis"
    research.analyze(candidates, raw, output)
    monthly = pd.read_csv(output / "monthly_levels.csv")
    assert "stale_optional" not in monthly


def test_snapshot_hash_mismatch_is_rejected(research_snapshot, tmp_path):
    candidates, raw, _manifest = research_snapshot
    with (raw / "usd.csv").open("a", encoding="utf-8") as handle:
        handle.write("2026-10-20,1000000,OK\n")
    with pytest.raises(ValueError, match="snapshot"):
        research.analyze(candidates, raw, tmp_path / "analysis")


def test_missing_required_source_has_explicit_coverage_error(research_snapshot, tmp_path):
    candidates, raw, manifest = research_snapshot
    for entry in manifest["series"]:
        if entry["alias"] == "ipsa":
            entry["status"] = "unavailable_or_invalid"
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(cobertura|faltan|requiere)"):
        research.analyze(candidates, raw, tmp_path / "analysis")


def test_future_rows_do_not_rewrite_closed_month_scores(research_snapshot, tmp_path):
    candidates, raw, manifest = research_snapshot
    first, second = tmp_path / "first", tmp_path / "second"
    research.analyze(candidates, raw, first)
    for entry in manifest["series"]:
        path = raw / (entry["alias"] + ".csv")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("2027-01-01,1000000,OK\n")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    research.analyze(candidates, raw, second)
    a = pd.read_csv(first / "scores.csv")
    b = pd.read_csv(second / "scores.csv")
    assert_frame_equal(a, b)


def _credit_distribution_fixture():
    return pd.DataFrame(
        {
            "ecb_supply_sme_less": [20.0, 0.0],
            "ecb_supply_sme_moderately_less": [30.0, 0.0],
            "ecb_supply_sme_unchanged": [0.0, 100.0],
            "ecb_supply_sme_moderately_more": [30.0, 0.0],
            "ecb_supply_sme_more": [20.0, 0.0],
            "ecb_supply_sme": [0.0, 0.0],
        },
        index=pd.date_range("2020-03-31", periods=2, freq="QE"),
    )


def test_equal_credit_balance_can_hide_different_distribution():
    result = research.credit_distribution(_credit_distribution_fixture())
    assert result.saldo_reconstruido.tolist() == [0.0, 0.0]
    assert result.saldo_publicado.tolist() == [0.0, 0.0]
    assert result.amplitud_cambio.tolist() == [100.0, 0.0]
    assert result.coexistencia_direcciones.tolist() == [100.0, 0.0]
    assert result.sin_cambios.tolist() == [0.0, 100.0]


@pytest.mark.parametrize("invalid_value, message", [(21.0, "suman"), (-1.0, "rango")])
def test_credit_distribution_rejects_inconsistent_percentages(invalid_value, message):
    proportions = _credit_distribution_fixture()
    proportions.iloc[0, 0] = invalid_value
    with pytest.raises(ValueError, match=message):
        research.credit_distribution(proportions)


def test_credit_distribution_checks_reconstructed_against_published_balance():
    proportions = _credit_distribution_fixture()
    proportions.loc[proportions.index[0], "ecb_supply_sme"] = 10.0
    with pytest.raises(ValueError, match="coincide"):
        research.credit_distribution(proportions)
