import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from imc.config import WEIGHTS, Methodology
from imc.data import DataError
from imc.engine import calculate_index, latest_complete, rolling_percentile


def test_appending_future_data_never_rewrites_past_scores(frames):
    cutoff = frames["dolar"].index[600]
    before = calculate_index({key: value.loc[:cutoff] for key, value in frames.items()})
    after = calculate_index(frames).loc[:cutoff]
    assert_frame_equal(before, after)


def test_constant_history_is_neutral_and_missing_current_is_not_filled():
    values = pd.Series([5.0, 5.0, 5.0, np.nan, 5.0])
    result = rolling_percentile(values, window=4, min_periods=2)
    assert result.iloc[1] == result.iloc[2] == result.iloc[4] == 50
    assert pd.isna(result.iloc[3])


def test_percentile_ties_have_midrank():
    result = rolling_percentile(pd.Series([1.0, 2.0, 2.0, 4.0]), window=4, min_periods=3)
    assert result.iloc[2] == pytest.approx(100 * 2 / 3)
    assert result.iloc[3] == 87.5


def test_index_has_complete_coverage_and_exact_weighted_sum(frames):
    result = calculate_index(frames)
    valid = result.loc[result.complete]
    assert not valid.empty
    assert valid["index"].between(0, 100).all()
    assert valid.available_components.eq(5).all()
    expected = sum(valid[f"score_{factor}"] * weight for factor, weight in WEIGHTS.items())
    assert np.allclose(valid["index"], expected)
    assert result["index"].iloc[:271].isna().all()


def test_missing_source_is_never_reweighted(frames):
    cutoff = frames["cobre"].index[-30]
    frames["cobre"] = frames["cobre"].loc[:cutoff]
    result = calculate_index(frames)
    assert pd.isna(result["index"].iloc[-1])
    assert result.available_components.iloc[-1] == 4
    with pytest.raises(DataError, match="desactualizados"):
        latest_complete(result, as_of=str(result.index[-1].date()))


def test_alignment_only_uses_past_observations(frames):
    result = calculate_index(frames)
    for alias in frames:
        dates = result[f"date_{alias}"].dropna()
        assert (dates <= dates.index.to_series()).all()


def test_idie_tolerance_and_oldest_source_are_visible(frames):
    target = frames["dolar"].index[-1]
    cutoff = target - pd.Timedelta(days=5)
    frames["incertidumbre"] = frames["incertidumbre"].loc[:cutoff]
    result = calculate_index(frames)
    latest, _ = latest_complete(result, as_of=str(target.date()))
    assert 5 <= latest["age_incertidumbre_days"] <= 7
    assert latest["oldest_input_date"] == latest["date_incertidumbre"]
    with pytest.raises(DataError, match="desactualizados"):
        latest_complete(result, as_of=str((target + pd.Timedelta(days=4)).date()))


def test_long_gap_restarts_calculation_and_normalization(frames):
    source = frames["tasa_10y"]
    frames["tasa_10y"] = pd.concat([source.iloc[:400], source.iloc[450:]])
    result = calculate_index(frames)
    assert pd.isna(result.loc[source.index[550], "score_variabilidad_tasas"])
    assert pd.notna(result.loc[source.index[-1], "score_variabilidad_tasas"])


def test_rates_use_changes_in_percentage_points_and_allow_negative_yields(frames):
    frames["tasa_10y"]["value"] = -2.0
    result = calculate_index(frames)
    assert result.variabilidad_tasas.dropna().eq(0).all()
    assert result.score_variabilidad_tasas.dropna().eq(50).all()


def test_price_factors_match_log_returns(frames):
    result = calculate_index(frames)
    day = result.index[-1]
    for alias, factor in [("dolar", "presion_cambiaria"), ("cobre", "momentum_cobre")]:
        values = frames[alias]["value"]
        expected = np.log(values.iloc[-1] / values.iloc[-21])
        assert result.loc[day, factor] == pytest.approx(expected)


def test_latest_does_not_use_observations_after_cutoff(frames):
    result = calculate_index(frames)
    cutoff = result.index[-15]
    latest, _ = latest_complete(result, as_of=str(cutoff.date()))
    assert latest.name == cutoff


@pytest.mark.parametrize("alias", ["dolar", "cobre"])
def test_nonpositive_prices_are_rejected(frames, alias):
    frames[alias].iloc[-1, 0] = 0
    with pytest.raises(DataError, match="positivos"):
        calculate_index(frames)


def test_duplicate_dates_are_rejected(frames):
    frames["dolar"] = pd.concat([frames["dolar"], frames["dolar"].iloc[-1:]])
    with pytest.raises(DataError, match="duplicadas"):
        calculate_index(frames)


def test_incomplete_input_set_is_rejected(frames):
    del frames["cobre"]
    with pytest.raises(DataError, match="cuatro"):
        calculate_index(frames)


@pytest.mark.parametrize(
    "options",
    [{"horizon": 0}, {"horizon": True}, {"min_history": 600}, {"normalization_window": 1}],
)
def test_invalid_windows_fail_locally(options):
    with pytest.raises(ValueError):
        Methodology(**options)
