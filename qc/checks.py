"""QC checks 2-4 (check 1, "file exists", lives in run.py).

Each check takes a lazy DataArray and returns a dict of LAZY results. Nothing here reads
data: run.py computes every check's results in one dask.compute so each store is read once.
Results that are 1-D over time go to the per-store .nc; everything else goes to the JSON.
"""

import dask
import numpy as np
import xarray as xr
from scipy import ndimage

# (low, high) in the units assumed below. See docs/llm/README.md §4.
#
# tas/tasmax/tasmin: derived from the real BCC-CSM2-MR ssp585 tas sample store, not a generic
# global bound. The store's field-wide daily min/max, excluding known-corrupt values, ranges
# 258.01-317.25 K over 2015-2100 (a plausible once-in-decades January cold snap up to a
# late-century SSP5-8.5 heatwave); padded ~8 K past each end for tasmax running warmer and
# tasmin running colder than tas on the same day, and for models/scenarios not yet sampled.
# Still shared across tas/tasmax/tasmin, matching the existing simplification here.
#
# That same scan found 37 days whose field-wide min is an exact repeated value (183.1576 K on
# 15 days, 185.0881 K on 22 days) spread across random months -- including summer -- which is
# physically impossible for genuine cold extremes and reads as a fixed sentinel/fill value
# leaking into the array rather than being masked as NaN. Excluded from the bound below, but
# worth its own investigation: check_physical_range only catches these because the new bound
# is now tight enough to exclude them, not because the sentinel value was identified as such.
PHYSICAL_LIMITS = {
    "tas": (250.0, 325.0),  # K
    "tasmax": (250.0, 325.0),  # K
    "tasmin": (250.0, 325.0),  # K
    "pr": (0.0, 0.03),  # kg m-2 s-1
    "hurs": (0.0, 100.0),  # %
    "huss": (0.0, 0.1),  # kg kg-1
    "rsds": (0.0, 500.0),  # W m-2
    "rlds": (0.0, 700.0),  # W m-2
    "sfcWind": (0.0, 50.0),  # m s-1
}

# A per-day score is "high" if it is above median + K * MAD of the store's own series.
BANDING_MAD_K = 8.0
ROUGHNESS_MAD_K = 15.0
ROUGHNESS_WINDOW = 9  # boxcar smoothing window (cells), one axis at a time

# A day is flagged if its field-wide min (or max) exactly matches at least this many other
# days' -- real, continuously-varying weather essentially never repeats a spatial extreme to
# full float precision, so a value repeated this often is the signature of a fixed sentinel/
# fill value leaking into the array. Calibrated against the real tas sample store: benign
# floating-point/order-statistic coincidences repeat at most 3 times; the confirmed sentinel
# values (see PHYSICAL_LIMITS comment above) repeat 15 and 22 times.
REPEATED_VALUE_MIN_REPEATS = 5


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


def check_spatial_banding(da, variable, k=BANDING_MAD_K):
    """Row-mean and column-mean profile for each day, then the discrete second difference
    (Laplacian) along each profile. A day fails if its worst row or column spikes far above
    the store's usual profile smoothness -- catches a straight seam or tile-boundary artifact
    baked into one day's field, which a day-to-day comparison can miss since both sides of a
    seam are individually smooth and still correlate fine with a normal neighbouring day."""
    dim0, dim1 = spatial_dims(da)  # exactly one row-like and one column-like axis (e.g. lat, lon)
    row_score = np.abs(_laplacian_1d(da.mean(dim1), dim0)).max(dim0)
    col_score = np.abs(_laplacian_1d(da.mean(dim0), dim1)).max(dim1)
    score = np.maximum(row_score, col_score)
    n_flagged, flagged_dates = _flag_high(score.data, da.indexes["time"], k)
    return {
        "banding_score": score,
        "banding_n_flagged": n_flagged,
        "banding_flagged_dates": flagged_dates,
    }


def _laplacian_1d(profile, dim):
    return profile.shift({dim: -1}) - 2 * profile + profile.shift({dim: 1})


def check_spatial_roughness(da, variable, k=ROUGHNESS_MAD_K, window=ROUGHNESS_WINDOW):
    """Each day's residual from a NaN-aware boxcar smooth over space (spatial std of
    raw - smoothed). A day fails if its residual is far rougher than typical -- catches
    corruption localized to a small patch of cells, which barely moves a whole-field
    correlation or mean when the rest of the day still looks plausible."""
    dim0, dim1 = spatial_dims(da)
    smoothed = xr.apply_ufunc(
        _boxcar_smooth,
        da,
        input_core_dims=[[dim0, dim1]],
        output_core_dims=[[dim0, dim1]],
        kwargs={"window": window},
        dask="parallelized",
        output_dtypes=[da.dtype],
    )
    roughness = (da - smoothed).std((dim0, dim1))
    n_flagged, flagged_dates = _flag_high(roughness.data, da.indexes["time"], k)
    return {
        "roughness_score": roughness,
        "roughness_n_flagged": n_flagged,
        "roughness_flagged_dates": flagged_dates,
    }


def _boxcar_smooth(x, window):
    """NaN-aware boxcar mean over the last two axes: mean over just the valid (non-NaN) cells
    in each window. One scipy call per chunk -- xarray's dask-backed rolling(min_periods=1)
    explodes into millions of graph tasks at this grid size and OOMs (see docs/llm/README.md)."""
    size = (1,) * (x.ndim - 2) + (window, window)
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    sums = ndimage.uniform_filter(filled, size=size, mode="nearest") * (window * window)
    counts = ndimage.uniform_filter(valid.astype(x.dtype), size=size, mode="nearest") * (window * window)
    with np.errstate(invalid="ignore"):
        return np.where(valid, sums / counts, np.nan)


@dask.delayed(nout=2)
def _flag_high(values, time, k):
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    flagged = values > median + k * mad  # NaN never flagged
    return int(flagged.sum()), [str(t) for t in time[flagged]]


def check_repeated_extreme(da, variable, min_repeats=REPEATED_VALUE_MIN_REPEATS):
    """Each day's field-wide min and max. A day fails if its min or max is a value shared by
    at least `min_repeats` days total across the series -- real weather essentially never
    reproduces a spatial extreme to full float precision, so a value repeated this often
    across unrelated dates is the signature of a fixed sentinel/fill value leaking into the
    array instead of being masked as NaN."""
    dims = spatial_dims(da)
    daily_min = da.min(dims)
    daily_max = da.max(dims)
    n_flagged, flagged_dates = _flag_repeated(daily_min.data, daily_max.data, da.indexes["time"], min_repeats)
    return {
        "repeated_daily_min": daily_min,
        "repeated_daily_max": daily_max,
        "repeated_n_flagged": n_flagged,
        "repeated_flagged_dates": flagged_dates,
    }


@dask.delayed(nout=2)
def _flag_repeated(daily_min, daily_max, time, min_repeats):
    def repeated_mask(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return counts[inverse] >= min_repeats

    flagged = repeated_mask(daily_min) | repeated_mask(daily_max)
    return int(flagged.sum()), [str(t) for t in time[flagged]]
