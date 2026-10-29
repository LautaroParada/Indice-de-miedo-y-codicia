"""Synthetic invariants for the registered retrospective campaign; no live API."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from campaign import economics, engine
from imc.config import WEIGHTS


@pytest.fixture
def config():
    path = Path(__file__).parents[1] / "campaign" / "contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def synthetic(config):
    dates = pd.date_range("2006-01-31", "2023-12-31", freq="ME")
    columns = list(
        dict.fromkeys(
            [c for group in config["families"].values() for c in group] + config["pca"]["features"]
        )
    )
    rng = np.random.default_rng(8721)
    features = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))), index=dates, columns=columns
    )
    levels = pd.DataFrame({"real10": rng.normal(size=len(dates))}, index=dates)
    baseline = pd.DataFrame(
        rng.uniform(0, 100, size=(len(dates), len(WEIGHTS))),
        index=dates,
        columns=["score_" + name for name in WEIGHTS],
    )
    return features, levels, baseline


def test_rank_reference_excludes_current_and_uses_midrank_for_ties():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 1.0]})
    result = engine.rank_past(frame, window=2, minimum=2)
    assert result.x.iloc[:2].isna().all()
    assert result.x.iloc[2:].tolist() == [100.0, 25.0]


def test_rank_never_changes_past_when_future_values_are_added():
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    past = pd.DataFrame({"x": np.sin(np.arange(120) * 0.37)}, index=dates)
    future = pd.DataFrame(
        {"x": [1e12, -1e12]}, index=pd.date_range("2020-01-31", periods=2, freq="ME")
    )
    assert_frame_equal(
        engine.rank_past(past), engine.rank_past(pd.concat([past, future])).loc[past.index]
    )


def test_missing_and_infinite_observations_never_count_as_valid_history():
    frame = pd.DataFrame({"x": [-np.inf, 0.0, 1.0, np.nan, np.inf]})
    result = engine.rank_past(frame, window=2, minimum=1)
    assert pd.isna(result.x.iloc[1])
    assert result.x.iloc[2] == 100.0
    assert result.x.iloc[3:].isna().all()


def test_missing_component_abstains_without_reweighting_other_component():
    scores = pd.DataFrame({"a": [20.0, 20.0], "b": [80.0, np.nan], "c": [45.0, 45.0]})
    groups = {"first": ["a", "b"], "second": ["c"]}
    result = engine.family_profiles(scores, groups, weights={"a": 1.2, "b": 0.8})
    assert result.loc[0, "first"] == pytest.approx(44.0)
    assert pd.isna(result.loc[1, "first"])
    assert result.second.tolist() == [45.0, 45.0]


def test_equal_family_orients_friction_and_preserves_missingness():
    panel = pd.DataFrame(
        {"market": [80.0, 80.0], "confidence": [20.0, 20.0], "friction": [20.0, np.nan]}
    )
    result = engine.equal_family(panel)
    assert result.iloc[0] == pytest.approx(60.0)
    assert pd.isna(result.iloc[1])


def test_category_boundaries_are_central_and_missing_is_not_central():
    scores = pd.Series([24.9, 25.0, 75.0, 75.1, np.nan])
    result = engine.categories(scores)
    assert result.iloc[:4].tolist() == [0.0, 1.0, 1.0, 2.0]
    assert pd.isna(result.iloc[4])


def test_pca_fit_and_development_scores_ignore_evaluation_values(config, synthetic):
    features, levels, _ = synthetic
    before, meta_before = engine.fit_pca(features, levels, config)
    modified, modified_levels = features.copy(), levels.copy()
    evaluation = features.index >= config["data"]["test_start"]
    modified.loc[evaluation] = modified.loc[evaluation] * -1e6 + 1e9
    modified_levels.loc[evaluation] += 1e10
    after, meta_after = engine.fit_pca(modified, modified_levels, config)
    assert meta_before == meta_after
    development = features.index <= config["data"]["train_end"]
    assert_series_equal(before.loc[development], after.loc[development])
    assert not np.allclose(before.loc[evaluation], after.loc[evaluation])


def test_pca_uses_real_rate_level_and_abstains_on_missing_retained_input(config, synthetic):
    features, levels, _ = synthetic
    _, meta = engine.fit_pca(features, levels, config)
    x = engine.pca_inputs(features, levels, config)
    assert_series_equal(x.real10, levels.real10)
    x.loc[x.index[-1], meta["retained"][0]] = np.nan
    projected = engine.pca_project(x, meta)
    assert pd.isna(projected.iloc[-1])


def test_pca_rejects_degenerate_training_with_clear_error(config, synthetic):
    features, levels, _ = synthetic
    features.loc[:] = 1.0
    levels.loc[:] = 1.0
    with pytest.raises(ValueError, match="(?i)(pca|vari|column|compon)"):
        engine.fit_pca(features, levels, config)


def test_stable_envelope_coverage_counts_months_without_long_history(config):
    cfg = copy.deepcopy(config)
    cfg["minimal_families"] = {"market": ["x"]}
    dates = pd.date_range("2011-07-31", "2021-12-31", freq="ME")
    features = pd.DataFrame({"x": np.arange(len(dates), dtype=float)}, index=dates)
    metrics, rows = engine.stable_envelopes(features, cfg)
    market = metrics["market"]
    assert market["denominator_all_test_months"] == 12
    assert market["complete_windows_n"] == 6
    assert market["stable_n"] == 6
    assert market["stable_coverage"] == 0.5
    assert rows.iloc[:6].label.str.contains("abstencion").all()
    assert rows.iloc[6:].label.eq("alto").all()


def test_change_metrics_do_not_bridge_missing_months():
    dates = pd.date_range("2021-01-31", periods=5, freq="ME")
    base = pd.Series([10.0, 40.0, np.nan, 80.0, 20.0], index=dates)
    changed = pd.Series([90.0, 60.0, np.nan, 20.0, 80.0], index=dates)
    result = engine.metric_change(base, changed)
    assert result["n"] == 4
    assert result["direction_comparable_n"] == 2
    assert result["direction_inversion_n"] == 2
    assert result["opposite_tail_rate"] == 0.75


def test_weight_stability_cannot_pass_with_too_few_common_months(config, synthetic, monkeypatch):
    features, levels, baseline = synthetic
    cfg = copy.deepcopy(config)
    cfg["stability"]["weight_draws"] = 1
    reps, scores, meta = engine.build_representations(features, levels, baseline, cfg)
    short_sample = {
        "n": 12,
        "median_abs": 0.0,
        "p95_abs": 0.0,
        "max_abs": 0.0,
        "category_change_rate": 0.0,
        "opposite_tail_rate": 0.0,
    }
    monkeypatch.setattr(engine, "metric_change", lambda *args: dict(short_sample))
    summary, _ = engine.sensitivity(features, levels, baseline, reps, scores, meta, cfg)
    assert summary["weight_cases_total"] > 0
    assert summary["weight_cases_passed"] == 0


def test_dynamic_identifiability_requires_validation_length(config):
    cfg = copy.deepcopy(config)
    dates = pd.date_range("2006-01-31", "2021-12-31", freq="ME")
    rng = np.random.default_rng(9761)
    profiles = pd.DataFrame(
        rng.uniform(0, 100, (len(dates), 3)),
        index=dates,
        columns=["market", "confidence", "friction"],
    )
    result = engine.dynamics({"minimal3": profiles}, cfg)
    assert min(result["effective_n"].values()) >= 100
    assert result["training_transitions"] >= 100
    assert result["test_months"] == 12
    assert result["sindy_identifiable_gate"] is False


def test_ridge_predictions_use_train_scale_and_ignore_other_test_rows():
    x = np.array([[1.0, 7.0], [2.0, 7.0], [4.0, 7.0], [8.0, 7.0]])
    y = np.array([4.0, 3.0, 9.0, 12.0])
    test = np.array([[3.0, 7.0], [1e10, -1e10]])
    actual = economics.ridge_predict(x, test, y)
    isolated = economics.ridge_predict(x, test[:1], y)
    assert actual[0] == pytest.approx(isolated[0])
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - x.mean(axis=0)) / scale
    coef = np.linalg.solve(z.T @ z + 10 * np.eye(2), z.T @ (y - y.mean()))
    expected = ((test - x.mean(axis=0)) / scale) @ coef + y.mean()
    np.testing.assert_allclose(actual, expected)


def test_quarterly_average_requires_three_complete_months_and_never_repeats():
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    monthly = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, np.nan, 3, 4, 5, 6]}, index=dates)
    quarters, counts = economics._quarter_mean(monthly)
    assert len(quarters) == 2
    assert counts.tolist() == [2, 3]
    assert quarters.iloc[0].isna().all()
    assert quarters.iloc[1].tolist() == [5.0, 5.0]
    assert quarters.index.is_unique


def test_duplicate_observations_in_one_month_are_rejected():
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2021-01-01", "2021-01-31"]))
    with pytest.raises(ValueError, match="unique months"):
        economics._monthly(frame, pd.date_range("2021-01-31", periods=1, freq="ME"))


def test_economic_evaluation_rejects_overlapping_train_test(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["data"]["test_start"] = cfg["data"]["train_end"]
    reps = {name: pd.DataFrame({"x": [1.0]}) for name in economics.ARCHITECTURES}
    with pytest.raises(ValueError, match="must not overlap"):
        economics.evaluate_economics(reps, pd.DataFrame(), pd.DataFrame(), cfg, tmp_path)


def test_block_bootstrap_abstains_if_isolated_quarters_cannot_enter_blocks():
    contiguous = pd.date_range("2020-03-31", periods=6, freq="QE")
    isolated = contiguous.append(pd.DatetimeIndex(["2023-03-31"]))
    result = economics._block_indices(isolated, 400, 4, np.random.default_rng(123))
    assert result is None


def test_paired_bootstrap_recovers_known_proportional_error_reduction():
    quarters = pd.date_range("2020-03-31", periods=12, freq="QE")
    indices = economics._block_indices(quarters, 400, 4, np.random.default_rng(123))
    errors = np.arange(1, 13, dtype=float)
    result = economics._gain(errors, 0.75 * errors, indices)
    assert result["status"] == "exploratory"
    assert result["bootstrap_draws"] == 400
    assert result["relative_mae_gain"] == pytest.approx(0.25)
    np.testing.assert_allclose(result["ci95"], [0.25, 0.25])


def test_zero_baseline_error_does_not_claim_improvement():
    result = economics._gain(np.zeros(4), np.zeros(4), np.tile(np.arange(4), (10, 1)))
    assert result["relative_mae_gain"] is None
    assert result["ci95"] is None
    assert result["status"] == "undefined_zero_baseline_mae"
