"""Each check on small synthetic stores: clean data passes, injected faults are caught.
Run with: python -m pytest tests"""

import numpy as np
import pandas as pd
import xarray as xr

from qc.catalog import store_path
from qc.checks import REPEATED_VALUE_MIN_REPEATS
from qc.run import check_store

MODEL, SCENARIO = "CESM2", "hist"


def make_field(n_days=50, n=20, seed=0):
    """A smooth, slowly moving daily field (in K) plus noise, with a fixed NaN 'ocean'."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 3, n), np.linspace(0, 3, n), indexing="ij")
    days = np.arange(n_days)[:, None, None]
    data = 290 + 5 * np.sin(x + 0.05 * days) * np.cos(y) + rng.normal(0, 0.3, (n_days, n, n))
    data[:, :3, :3] = np.nan
    return data


def write(root, variable, data):
    time = pd.date_range("1950-01-01", periods=len(data))
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time}, name=variable)
    da.chunk({"time": 10}).to_dataset().to_zarr(store_path(root, variable, MODEL, SCENARIO))


def run(root, variable="tas"):
    return check_store(root, variable, MODEL, SCENARIO)


def test_missing(tmp_path):
    assert run(tmp_path)[0]["status"] == "missing"


def test_unreadable(tmp_path):
    path = store_path(tmp_path, "tas", MODEL, SCENARIO)
    path.mkdir(parents=True)
    (path / "zarr.json").write_text("not json")
    row, series = run(tmp_path)
    assert row["status"] == "unreadable" and series is None


def test_clean_store_passes(tmp_path):
    write(tmp_path, "tas", make_field())
    row, series = run(tmp_path)
    assert row["status"] == "ok"
    assert row["missing_n_deviating"] == 0
    assert row["range_n_out"] == 0
    assert row["banding_n_flagged"] == 0
    assert row["roughness_n_flagged"] == 0
    assert row["repeated_n_flagged"] == 0
    assert set(series) == {"nan_count", "banding_score", "roughness_score", "repeated_daily_min", "repeated_daily_max"}


def test_extra_nans_on_one_day(tmp_path):
    data = make_field()
    data[17, 10, 10] = np.nan
    write(tmp_path, "tas", data)
    row, series = run(tmp_path)
    assert row["missing_n_deviating"] == 1
    assert int(series["nan_count"][17]) == row["missing_mode"] + 1


def test_value_out_of_range(tmp_path):
    data = make_field()
    data[5, 5, 5] = 400
    write(tmp_path, "tas", data)
    assert run(tmp_path)[0]["range_n_out"] == 1


def test_temperature_order(tmp_path):
    data = make_field()
    tasmin, tasmax = data - 5, data + 5
    tasmin[3, 8, 8] = data[3, 8, 8] + 1
    write(tmp_path, "tas", data)
    write(tmp_path, "tasmin", tasmin)
    write(tmp_path, "tasmax", tasmax)
    row = run(tmp_path)[0]
    assert (row["range_n_tasmin_gt_tas"], row["range_n_tas_gt_tasmax"], row["range_n_tasmin_gt_tasmax"]) == (1, 0, 0)


def test_seam_is_flagged_by_banding(tmp_path):
    data = make_field()
    data[25, :, 10:] += 15.0  # a hard step along one axis: not a real weather pattern
    write(tmp_path, "tas", data)
    row = run(tmp_path)[0]
    assert row["banding_n_flagged"] == 1
    assert row["banding_flagged_dates"][0].startswith("1950-01-26")


def test_localized_corruption_is_flagged_by_roughness(tmp_path):
    data = make_field()
    rng = np.random.default_rng(2)
    data[30, 5:8, 5:8] = rng.uniform(270, 310, size=(3, 3))  # a small patch of incoherent noise
    write(tmp_path, "tas", data)
    row = run(tmp_path)[0]
    assert row["roughness_n_flagged"] == 1
    assert row["roughness_flagged_dates"][0].startswith("1950-01-31")


def test_repeated_value_is_flagged(tmp_path):
    data = make_field()
    bad_days = [5 * (i + 1) for i in range(REPEATED_VALUE_MIN_REPEATS)]
    for d in bad_days:
        data[d, 8, 8] = 260.0  # far below the natural range: becomes that day's min
    write(tmp_path, "tas", data)
    row = run(tmp_path)[0]

    time = pd.date_range("1950-01-01", periods=50)
    assert row["repeated_n_flagged"] == len(bad_days)
    assert set(row["repeated_flagged_dates"]) == {str(time[d]) for d in bad_days}
