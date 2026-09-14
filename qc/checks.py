"""QC checks 2-4 (check 1, "file exists", lives in run.py).

Each check takes a lazy DataArray and returns a dict of LAZY results. Nothing here reads
data: run.py computes every check's results in one dask.compute so each store is read once.
Results that are 1-D over time go to the per-store .nc; everything else goes to the JSON.
"""

import dask
import numpy as np
import xarray as xr

# (low, high) in the units assumed below. Starting points, not yet confirmed against the
# units in the stores. See docs/llm/README.md §4.
PHYSICAL_LIMITS = {
    "tas": (180.0, 340.0),  # K
    "tasmax": (180.0, 340.0),  # K
    "tasmin": (180.0, 340.0),  # K
    "pr": (0.0, 0.03),  # kg m-2 s-1
    "hurs": (0.0, 100.0),  # %
    "huss": (0.0, 0.1),  # kg kg-1
    "rsds": (0.0, 500.0),  # W m-2
    "rlds": (0.0, 700.0),  # W m-2
    "sfcWind": (0.0, 50.0),  # m s-1
}

# A day-to-day correlation is "low" if it is below median - K * MAD of the store's own series.
SPATIAL_MAD_K = 5.0


def spatial_dims(da):
    return [d for d in da.dims if d != "time"]


def check_missing_data(da, variable):
    """Count NaNs on each day. The land/ocean mask is fixed, so every day should have exactly
    the most common (mode) count. Any day with a different count fails."""
    nan_count = da.isnull().sum(spatial_dims(da))
    mode, n_deviating, pct_deviating = _mode_deviation(nan_count.data)
    return {
        "nan_count": nan_count,
        "missing_mode": mode,
        "missing_n_deviating": n_deviating,
        "missing_pct_deviating": pct_deviating,
    }


@dask.delayed(nout=3)
def _mode_deviation(nan_count):
    values, freq = np.unique(nan_count, return_counts=True)
    mode = values[freq.argmax()]
    deviating = nan_count != mode
    return int(mode), int(deviating.sum()), float(100 * deviating.mean())


def check_physical_range(da, variable):
    """Every non-NaN value must lie within PHYSICAL_LIMITS[variable]. Any value outside fails."""
    low, high = PHYSICAL_LIMITS[variable]
    n_out = ((da < low) | (da > high)).sum()  # NaN compares False, so NaNs are never counted
    n_valid = da.notnull().sum()
    return {
        "range_min": da.min(),
        "range_max": da.max(),
        "range_n_out": n_out,
        "range_pct_out": 100 * n_out / n_valid,
    }


def check_temperature_order(tasmin, tas, tasmax):
    """Temperature-only part of the range check: tasmin <= tas <= tasmax in every cell on
    every day. Any cell that breaks the order fails. The three grids must match exactly."""
    tasmin, tas, tasmax = xr.align(tasmin, tas, tasmax, join="exact")
    return {
        "range_n_tasmin_gt_tas": (tasmin > tas).sum(),
        "range_n_tas_gt_tasmax": (tas > tasmax).sum(),
        "range_n_tasmin_gt_tasmax": (tasmin > tasmax).sum(),
    }


def check_spatial_pattern(da, variable, k=SPATIAL_MAD_K):
    """Normalize each day's map, then correlate it with the next day's map. A day fails if it
    correlates unusually poorly with BOTH the day before and the day after."""
    dims = spatial_dims(da)
    x = da.astype("float64")
    z = (x - x.mean(dims)) / x.std(dims)
    r = (z * z.shift(time=-1)).mean(dims)  # r[t] = corr(day t, day t+1); the last is NaN
    n_flagged, flagged_dates = _flag_days(r.data, da.indexes["time"], k)
    return {
        "spatial_r": r,
        "spatial_n_flagged": n_flagged,
        "spatial_flagged_dates": flagged_dates,
    }


@dask.delayed(nout=2)
def _flag_days(r, time, k):
    median = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - median))
    low = r < median - k * mad  # NaN (e.g. an all-zero pr day) compares False
    before = np.concatenate([[False], low[:-1]])  # corr(day t-1, day t)
    flagged = before & low  # the first and last days have one neighbour, so they are never flagged
    return int(flagged.sum()), [str(t) for t in time[flagged]]
