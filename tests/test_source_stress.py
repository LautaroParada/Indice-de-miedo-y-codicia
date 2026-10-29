"""Finite source stress, positive price domains and unchanged development fit."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from campaign import engine, source_stress
from imc.config import WEIGHTS


def test_price_revisions_preserve_positivity_and_do_not_change_other_years():
    index = pd.to_datetime(["2020-12-31", "2021-01-31", "2021-12-31", "2022-01-31"])
    original = pd.Series([1.0, 0.0001, 100.0, 3.0], index=index)
    for direction in [-1, 1]:
        modified = source_stress._revision(original, "dolar", direction, 10000)
        assert modified.gt(0).all()
        assert modified.iloc[0] == original.iloc[0] and modified.iloc[-1] == original.iloc[-1]
        np.testing.assert_allclose(
            modified.iloc[1:3] / original.iloc[1:3], np.exp(direction * 0.025)
        )


def test_invalid_idie_revision_abstains_instead_of_clipping():
    values = pd.Series([1.0], index=pd.to_datetime(["2021-01-31"]))
    with pytest.raises(ValueError, match="negative"):
        source_stress._revision(values, "incertidumbre", -1, 100)
    assert values.iloc[0] == 1.0


def test_v2_ablation_renormalizes_only_declared_weights_and_requires_other_components():
    columns = ["score_" + k for k in WEIGHTS]
    scores = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], columns=columns)
    result = source_stress._weighted(scores, omit="momentum_cobre")
    expected = (10 * 0.125 + 20 * 0.125 + 40 * 0.25 + 50 * 0.25) / 0.75
    assert result.iloc[0] == pytest.approx(expected)
    scores.loc[0, "score_incertidumbre"] = np.nan
    assert pd.isna(source_stress._weighted(scores, omit="momentum_cobre").iloc[0])


def test_54_cases_freeze_pca_for_post_development_revisions_and_do_not_mutate_inputs(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(17)
    index = pd.date_range("2015-01-31", "2022-12-31", freq="ME")
    columns = [
        "usd",
        "ipsa",
        "real10",
        "bank_spread",
        "rate_volatility",
        "real_credit_growth",
        "copper",
    ]
    features = pd.DataFrame(rng.normal(size=(len(index), 7)), index=index, columns=columns)
    levels = pd.DataFrame({"real10": rng.normal(size=len(index))}, index=index)
    config = {
        "data": {"train_end": "2020-12-31", "test_start": "2021-01-01"},
        "pca": {
            "features": columns,
            "orientation": dict.fromkeys(columns, 1),
            "minimum_train_months": 36,
        },
        "resource_limits": {"sensitivity_scenarios_max": 200},
    }
    frames = {
        name: pd.DataFrame(
            {"value": 100 + np.arange(len(index)) * 0.2 + i, "status_code": "OK"}, index=index
        )
        for i, name in enumerate(source_stress.RAW_ALIASES)
    }
    windows = []

    def fake_v2(raw, dates, methodology=None):
        windows.append(methodology)
        values = sum(f.value for f in raw.values()) / 8
        return pd.DataFrame({"score_" + k: values for k in WEIGHTS}, index=dates)

    monkeypatch.setattr(source_stress, "_raw_frames", lambda _: frames)
    monkeypatch.setattr(source_stress, "_v2_scores", fake_v2)
    baseline = fake_v2(frames, index)
    pc, _ = engine.fit_pca(features, levels, config)
    reps = {
        "v2": source_stress._weighted(baseline).to_frame("aggregate"),
        "dt905_adapted": pc.to_frame(),
    }
    before_f, before_l, before_config = features.copy(), levels.copy(), copy.deepcopy(config)
    before_raw = {name: frame.copy() for name, frame in frames.items()}
    fit_calls = []
    original_fit = engine.fit_pca

    def audited_fit(f, level, settings):
        fit_calls.append(settings["data"]["train_end"])
        return original_fit(f, level, settings)

    monkeypatch.setattr(engine, "fit_pca", audited_fit)
    result = source_stress.evaluate_source_stress(
        tmp_path, features, levels, baseline, reps, config, tmp_path / "results"
    )
    assert result["scenario_count"] == 54
    assert result["metric_cases"] == 47 and result["missingness_cases"] == 7
    assert len(fit_calls) == 15  # Baseline + seven ablations + seven historical shocks.
    assert set(fit_calls) == {"2020-12-31"}  # Fourteen revisions never fit a 2021 model.
    assert {(m.normalization_window, m.min_history) for m in windows if m} == {
        (756, 756),
        (1260, 1260),
        (2520, 2520),
    }
    assert all(row["status"] == "ok" for row in result["records"])
    assert all(row["all_abstained"] for row in result["missingness"] if row["retained"])
    assert_frame_equal(features, before_f)
    assert_frame_equal(levels, before_l)
    assert config == before_config
    for name in frames:
        assert_frame_equal(frames[name], before_raw[name])
    table = pd.read_csv(tmp_path / "results/source_stress.csv")
    assert len(table) == 47 and table.scenario.is_unique
    assert "not point-in-time" in result["interpretation"]


def test_pca_ablation_really_excludes_real_rate_level():
    features = pd.DataFrame({"usd": [1.0, 2.0], "real10": [4.0, 5.0]})
    levels = pd.DataFrame({"real10": [40.0, 50.0]})
    assert list(engine.pca_inputs(features, levels, {"pca": {"features": ["usd"]}})) == ["usd"]
