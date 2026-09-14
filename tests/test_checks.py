"""Each check on small synthetic stores: clean data passes, injected faults are caught.
Run with: python -m pytest tests"""

import numpy as np
import pandas as pd
import xarray as xr

from qc.catalog import store_path
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
    assert row["spatial_n_flagged"] == 0
    assert set(series) == {"nan_count", "spatial_r"}


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


def test_scrambled_day_is_flagged(tmp_path):
    data = make_field()
    data[25] = np.random.default_rng(1).normal(290, 5, data[25].shape)
    data[25, :3, :3] = np.nan  # keep the mask so only the spatial check should fire
    write(tmp_path, "tas", data)
    row = run(tmp_path)[0]
    assert row["missing_n_deviating"] == 0
    assert row["spatial_n_flagged"] == 1
    assert row["spatial_flagged_dates"][0].startswith("1950-01-26")
