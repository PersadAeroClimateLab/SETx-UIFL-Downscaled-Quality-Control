"""Run every QC check on every expected store and write results. See docs/llm/README.md."""

import argparse
import itertools
import json
import traceback
from pathlib import Path

import dask
import pandas as pd
import xarray as xr

from qc import checks
from qc.catalog import MODELS, SCENARIOS, VARIABLES, store_path

CHECKS = [
    checks.check_missing_data,
    checks.check_physical_range,
    checks.check_spatial_banding,
    checks.check_spatial_roughness,
    checks.check_repeated_extreme,
]


def open_store(path, variable):
    # TBD: confirm the data variable name once the sample store arrives.
    return xr.open_zarr(path, chunks={})[variable]


def check_store(root, variable, model, scenario):
    """Check 1 (file exists), then every check in CHECKS in a single compute.
    Returns (row of scalar results for the JSON, Dataset of time series or None)."""
    path = store_path(root, variable, model, scenario)
    row = {"variable": variable, "model": model, "scenario": scenario, "path": str(path)}
    if not path.exists():
        return {**row, "status": "missing"}, None
    try:
        da = open_store(path, variable)
    except Exception as e:
        return {**row, "status": "unreadable", "error": repr(e)}, None

    try:
        lazy = {}
        for check in CHECKS:
            lazy.update(check(da, variable))
        if variable == "tas":
            lazy.update(temperature_order(root, model, scenario, da))
        # Wrap everything in delayed: dask 2026.8.0 returns mixed xarray/delayed results grouped
        # by type instead of in argument order, which pairs values with the wrong keys.
        results = dict(zip(lazy, dask.compute(*[dask.delayed(v) for v in lazy.values()])))
    except Exception as e:
        traceback.print_exc()
        return {**row, "status": "failed", "error": repr(e)}, None

    series = {k: v for k, v in results.items() if isinstance(v, xr.DataArray) and "time" in v.dims}
    scalars = {k: v.item() if isinstance(v, xr.DataArray) else v for k, v in results.items() if k not in series}
    return {**row, "status": "ok", **scalars}, xr.Dataset(series)


def temperature_order(root, model, scenario, tas):
    """Run the tasmin <= tas <= tasmax check once per model/scenario, from the tas store."""
    paths = [store_path(root, v, model, scenario) for v in ("tasmin", "tasmax")]
    if not all(p.exists() for p in paths):
        return {}  # check 1 already reports the missing store
    # ponytail: re-reads tasmin and tasmax (~2/9 extra I/O over the full run). If that
    # matters, check all three temperature stores together in one compute.
    return checks.check_temperature_order(open_store(paths[0], "tasmin"), tas, open_store(paths[1], "tasmax"))


def write_summary(out):
    rows = [json.loads(p.read_text()) for p in sorted((out / "stores").glob("*.json"))]
    df = pd.DataFrame(rows)
    list_cols = [c for c in df.columns if c.endswith("_dates")]  # lists stay in the JSON only
    df = df.drop(columns=list_cols)
    df.to_csv(out / "summary.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/local1/SETx_UIFL_Zarr")
    p.add_argument("--out", default="results", type=Path)
    p.add_argument("--variables", nargs="+", default=VARIABLES, choices=VARIABLES)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--n-workers", type=int, default=42)
    p.add_argument("--threads-per-worker", type=int, default=4)
    p.add_argument("--local-directory", help="dask spill directory; use fast local disk")
    p.add_argument("--overwrite", action="store_true", help="re-check stores that already have results")
    args = p.parse_args(argv)

    from dask.distributed import Client, LocalCluster

    stores_dir = args.out / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        local_directory=args.local_directory,
    )
    with cluster, Client(cluster):
        for variable, model, scenario in itertools.product(args.variables, args.models, args.scenarios):
            name = f"{model}_{scenario}_{variable}"
            json_path, nc_path = stores_dir / f"{name}.json", stores_dir / f"{name}.nc"
            if json_path.exists() and not args.overwrite:
                continue
            print(name, flush=True)
            row, series = check_store(args.root, variable, model, scenario)
            nc_path.unlink(missing_ok=True)
            if series is not None:
                series.to_netcdf(nc_path, encoding={k: {"zlib": True} for k in series})
            json_path.write_text(json.dumps(row, indent=2))  # written last: marks the store as done
    write_summary(args.out)


if __name__ == "__main__":
    main()
