"""Unit tests for qc/checks.py: verify the actual computed numbers are correct, not just
whether a fault gets flagged. Complements test_checks.py, which tests the full check_store
pipeline against synthetic zarr stores and injected faults.

Inputs here are small, hand-computable, in-memory DataArrays (no zarr I/O). Expected values
are derived independently by hand in each test's comment -- never by re-running the function
under test -- so a bug shared between the test and the implementation can't hide.
Run with: python -m pytest tests
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from qc import checks
from qc.checks import _boxcar_smooth, _flag_high, _flag_repeated, _laplacian_1d, _mode_deviation


def compute_all(lazy_dict):
    """Mirror run.check_store's own compute step (wrap every value in dask.delayed first --
    see the dask gotcha noted in docs/llm/README.md -- then compute them all together)."""
    return dict(zip(lazy_dict, dask.compute(*[dask.delayed(v) for v in lazy_dict.values()])))


TIME = pd.date_range("2000-01-01", periods=6)


# ---- low-level delayed helpers, tested as pure functions -------------------------------


def test_mode_deviation():
    nan_count = np.array([5, 5, 5, 7, 5, 7])
    mode, n_deviating, pct_deviating = dask.compute(*_mode_deviation(nan_count))
    assert (mode, n_deviating) == (5, 2)
    assert pct_deviating == pytest.approx(100 * 2 / 6)


def test_flag_high_zero_mad_edge_case():
    # sorted [1,1,1,1,10]: median=1.0, MAD=median(|values-1|)=median([0,0,0,0,9])=0.0, so the
    # threshold collapses to the median itself -- only the strict outlier should flag.
    values = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    n_flagged, dates = dask.compute(*_flag_high(values, TIME[:5], k=2))
    assert n_flagged == 1
    assert dates == [str(TIME[4])]


def test_flag_high_ignores_nan():
    # nanmedian([1,2,3,100]) = 2.5; nanmedian(|[1,2,3,100]-2.5|) = nanmedian([1.5,.5,.5,97.5]) = 1.0
    # threshold = 2.5 + 1*1.0 = 3.5 -- only 100 clears it; the NaN itself must never flag.
    values = np.array([1.0, 2.0, 3.0, np.nan, 100.0])
    n_flagged, dates = dask.compute(*_flag_high(values, TIME[:5], k=1))
    assert n_flagged == 1
    assert dates == [str(TIME[4])]


def test_flag_repeated_matches_either_min_or_max():
    daily_min = np.array([1.0, 2.0, 2.0, 2.0, 5.0, 6.0])  # 2.0 repeats 3x
    daily_max = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 10.0])  # 10.0 repeats only 2x
    n_flagged, dates = dask.compute(*_flag_repeated(daily_min, daily_max, TIME, min_repeats=3))
    assert n_flagged == 3
    assert set(dates) == {str(TIME[1]), str(TIME[2]), str(TIME[3])}


def test_laplacian_1d_zero_on_a_ramp_spikes_at_a_jump():
    # a linear ramp has exactly zero second difference; a jump from 4 to 10 produces exactly
    # one non-zero value, f[i-1] - 2*f[i] + f[i+1] = 4 - 2*10 + (next value beyond the array,
    # here nothing) -- computed directly below instead of restating the formula in prose.
    profile = xr.DataArray([1.0, 2.0, 3.0, 4.0, 10.0], dims="lat", coords={"lat": range(5)})
    result = _laplacian_1d(profile, "lat").values
    assert np.isnan(result[0]) and np.isnan(result[-1])  # no neighbour on one side
    np.testing.assert_allclose(result[1:4], [0.0, 0.0, 5.0])


def test_boxcar_smooth_hand_computed_value():
    x = np.ones((1, 5, 5))
    x[0, 2, 2] = 10.0
    smoothed = _boxcar_smooth(x, window=3)
    # centre cell's 3x3 window: eight 1.0s plus the 10.0 itself = (8*1 + 10) / 9
    assert smoothed[0, 2, 2] == pytest.approx((8 * 1.0 + 10.0) / 9)
    # a cell far from the spike: its window is untouched, so smoothing is a no-op
    assert smoothed[0, 0, 0] == pytest.approx(1.0)


def test_boxcar_smooth_excludes_nan_from_the_window_and_leaves_it_nan():
    x = np.ones((1, 5, 5))
    x[0, 2, 2] = 10.0
    x[0, 1, 1] = np.nan  # inside the centre cell's window
    smoothed = _boxcar_smooth(x, window=3)
    # the NaN cell drops out of both the sum and the count: (7*1 + 10) / 8, not /9
    assert smoothed[0, 2, 2] == pytest.approx((7 * 1.0 + 10.0) / 8)
    assert np.isnan(smoothed[0, 1, 1])  # the NaN cell itself stays NaN, not averaged over


# ---- check functions, tested directly on small in-memory DataArrays --------------------


def test_check_missing_data_values():
    data = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[1.0, 2.0], [3.0, np.nan]],
            [[1.0, 2.0], [np.nan, np.nan]],  # one extra NaN this day
        ]
    )
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME[:3]})
    result = compute_all(checks.check_missing_data(da, "tas"))
    assert result["nan_count"].values.tolist() == [1, 1, 2]
    assert result["missing_mode"] == 1
    assert result["missing_n_deviating"] == 1
    assert result["missing_pct_deviating"] == pytest.approx(100 / 3)


def test_check_physical_range_values():
    low, high = checks.PHYSICAL_LIMITS["tas"]
    data = np.array(
        [
            [[low - 10.0, 300.0], [310.0, np.nan]],  # one cell below low
            [[low - 1.0, high + 1.0], [300.0, np.nan]],  # one below low, one above high
        ]
    )
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME[:2]})
    result = compute_all(checks.check_physical_range(da, "tas"))
    assert float(result["range_min"]) == pytest.approx(low - 10.0)
    assert float(result["range_max"]) == pytest.approx(high + 1.0)
    assert int(result["range_n_out"]) == 3  # 6 valid cells total, 3 outside the bound
    assert float(result["range_pct_out"]) == pytest.approx(100 * 3 / 6)


def test_check_temperature_order_counts_each_violation_kind_separately():
    tas = xr.DataArray(np.full((1, 2, 2), 290.0), dims=("time", "lat", "lon"))
    tasmin = tas - 5  # 285 everywhere by default
    tasmax = tas + 5  # 295 everywhere by default
    tasmin = tasmin.copy()
    tasmax = tasmax.copy()
    tasmin[0, 0, 0] = 292.0  # tasmin > tas here (292 > 290); tasmin <= tasmax still holds
    tasmax[0, 1, 1] = 288.0  # tas > tasmax here (290 > 288); tasmin <= tasmax still holds
    result = compute_all(checks.check_temperature_order(tasmin, tas, tasmax))
    assert int(result["range_n_tasmin_gt_tas"]) == 1
    assert int(result["range_n_tas_gt_tasmax"]) == 1
    assert int(result["range_n_tasmin_gt_tasmax"]) == 0


def test_check_spatial_banding_row_direction():
    # constant across lon, a ramp-then-jump down the lat axis -- see
    # test_laplacian_1d_zero_on_a_ramp_spikes_at_a_jump for why the peak is exactly 5.0
    row_values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    data = np.tile(row_values[:, None], (1, 3))[None, :, :]
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME[:1]})
    result = compute_all(checks.check_spatial_banding(da, "tas", k=1000))
    assert result["banding_score"].values[0] == pytest.approx(5.0)


def test_check_spatial_banding_column_direction():
    # same shape, transposed: constant across lat, the ramp-then-jump runs along lon instead,
    # to confirm the column-profile path (not just the row-profile path) is wired correctly
    col_values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    data = np.tile(col_values[None, :], (3, 1))[None, :, :]
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME[:1]})
    result = compute_all(checks.check_spatial_banding(da, "tas", k=1000))
    assert result["banding_score"].values[0] == pytest.approx(5.0)


def test_check_spatial_roughness_uniform_field_is_exactly_zero():
    da = xr.DataArray(np.full((3, 6, 6), 300.0), dims=("time", "lat", "lon"), coords={"time": TIME[:3]})
    result = compute_all(checks.check_spatial_roughness(da, "tas", k=1000, window=3))
    assert result["roughness_score"].values == pytest.approx([0.0, 0.0, 0.0])


def test_check_spatial_roughness_spike_matches_hand_computed_value():
    data = np.full((1, 5, 5), 300.0)
    data[0, 2, 2] = 309.0
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME[:1]})
    result = compute_all(checks.check_spatial_roughness(da, "tas", k=1000, window=3))
    # the window=3 spike ripples into its 8 immediate neighbours: centre residual = +8.0,
    # each of the 8 neighbours = -1.0 (their window's mean gets pulled up by the spike), the
    # other 16 of the 5x5=25 cells are untouched (residual 0.0) -- std of that 25-cell array:
    expected_residual = np.array([8.0] + [-1.0] * 8 + [0.0] * 16)
    assert result["roughness_score"].values[0] == pytest.approx(expected_residual.std())


def test_check_repeated_extreme_values_and_flagging():
    daily_min = [1.0, 2.0, 2.0, 2.0, 5.0, 6.0]
    daily_max = [10.0, 11.0, 12.0, 13.0, 14.0, 10.0]
    data = np.array(list(zip(daily_min, daily_max)))[:, None, :]  # (time, lat=1, lon=2)
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": TIME})
    result = compute_all(checks.check_repeated_extreme(da, "tas", min_repeats=3))
    assert result["repeated_daily_min"].values.tolist() == pytest.approx(daily_min)
    assert result["repeated_daily_max"].values.tolist() == pytest.approx(daily_max)
    assert result["repeated_n_flagged"] == 3
    assert set(result["repeated_flagged_dates"]) == {str(TIME[1]), str(TIME[2]), str(TIME[3])}
