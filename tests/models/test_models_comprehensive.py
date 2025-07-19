"""
Comprehensive tests for all models including heavy foundation models.
These tests are marked as slow and excluded from regular CI runs.
"""
import sys

import pandas as pd
import pytest
from utilsforecast.data import generate_series as _generate_series

from ..conftest import all_models


def generate_series(n_series, freq, **kwargs):
    df = _generate_series(n_series, freq, **kwargs)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)
@pytest.mark.parametrize("freq", ["D", "W-MON"])  # Reduced freq combinations for speed
def test_freq_inferred_correctly_comprehensive(model, freq):
    n_series = 2
    df = generate_series(
        n_series,
        freq=freq,
    )
    fcsts_no_freq = model.forecast(df, h=3)
    fcsts_with_freq = model.forecast(df, h=3, freq=freq)
    cv_no_freq = model.cross_validation(df, h=3)
    cv_with_freq = model.cross_validation(df, h=3, freq=freq)
    # some foundation models produce different results
    # each time they are called
    cols_to_check = ["unique_id", "ds"]
    cols_to_check_cv = ["unique_id", "ds", "y", "cutoff"]
    pd.testing.assert_frame_equal(
        fcsts_no_freq[cols_to_check],
        fcsts_with_freq[cols_to_check],
    )
    pd.testing.assert_frame_equal(
        cv_no_freq[cols_to_check_cv],
        cv_with_freq[cols_to_check_cv],
    )


@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)
@pytest.mark.parametrize("freq", ["D", "W-MON"])  # Reduced freq combinations
@pytest.mark.parametrize("h", [1, 12])
def test_correct_forecast_dates_comprehensive(model, freq, h):
    n_series = 3  # Reduced from 5 for speed
    df = generate_series(
        n_series,
        freq=freq,
    )
    df_test = df.groupby("unique_id").tail(h)
    df_train = df.drop(df_test.index)
    fcst_df = model.forecast(
        df_train,
        h=h,
        freq=freq,
    )
    exp_n_cols = 3
    assert fcst_df.shape == (n_series * h, exp_n_cols)
    exp_cols = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        fcst_df[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        df_test[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)
@pytest.mark.parametrize("freq", ["D"])  # Only test daily for foundation models
@pytest.mark.parametrize("n_windows", [1])  # Reduced windows for speed
def test_cross_validation_comprehensive(model, freq, n_windows):
    h = 6  # Reduced from 12 for speed
    n_series = 3  # Reduced from 5 for speed
    df = generate_series(n_series, freq=freq, equal_ends=True)
    cv_df = model.cross_validation(
        df,
        h=h,
        freq=freq,
        n_windows=n_windows,
    )
    exp_n_cols = 5  # unique_id, cutoff, ds, y, model
    assert cv_df.shape == (n_series * h * n_windows, exp_n_cols)
    cutoffs = cv_df["cutoff"].unique()
    assert len(cutoffs) == n_windows
    df_test = df.groupby("unique_id").tail(h * n_windows)
    exp_cols = ["unique_id", "ds", "y"]
    pd.testing.assert_frame_equal(
        cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        df_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
    )


@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)
def test_using_quantiles_comprehensive(model):
    qs = [0.1, 0.5, 0.9]  # Reduced quantiles for speed
    df = generate_series(n_series=2, freq="D")  # Reduced series
    fcst_df = model.forecast(
        df=df,
        h=2,
        freq="D",
        quantiles=qs,
    )
    exp_qs_cols = [f"{model.alias}-q-{int(100 * q)}" for q in qs]
    assert all(col in fcst_df.columns for col in exp_qs_cols)
    assert not any(("-lo-" in col or "-hi-" in col) for col in fcst_df.columns)


@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)
def test_using_level_comprehensive(model):
    level = [20, 80]  # Reduced levels for speed
    df = generate_series(n_series=2, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=2,
        freq="D",
        level=level,
    )
    exp_lv_cols = []
    for lv in level:
        if lv == 0:
            continue
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df.columns)