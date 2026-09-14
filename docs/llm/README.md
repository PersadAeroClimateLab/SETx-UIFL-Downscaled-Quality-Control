# LLM Guide: SETx UIFL Downscaled Quality Control

This file is for LLM coding agents working in this repo. Read it before you change anything.
It is the spec for the QC framework. **Status: initial framework built (`qc/`, `tests/`,
`Dockerfile`). It has not yet run on real data.** A sample zarr store is still downloading.
Anything marked **TBD** depends on inspecting it.

This is being developed on a local development machine that doesnt have access to the filesystem or compute node. The dev machine has 12 cores and 32 GB of RAM for testing, the final specs should cater to the compute node hardware.
(On the dev machine, pass something like `--n-workers 3 --threads-per-worker 4`. The CLI
defaults target the compute node.)

The human-facing overview is in [../../README.md](../../README.md). If this file and that one
disagree, the top-level README wins. Flag the conflict to the user. Known conflict: the
top-level README says check 2 compares against the **mean**. The user has since decided on
the **mode** (§11).

---

## 1. Goal

Write Python scripts that run quality-control (QC) checks over ~30 TB of 1-km daily downscaled
climate data stored as zarr. The scripts run inside a Docker image. That image is converted to
an Apptainer image and run on one large machine.

Priorities, in order:

1. **Correct.** A QC tool that gives wrong answers is worse than having no tool.
2. **Simple and readable.** A climate scientist should be able to read a check function and
   understand what it tests.
3. **Fast enough.** Performance matters, but never at the cost of 1 or 2. See §6.

## 2. Target machine

| Resource | Amount |
|---|---|
| CPU | 168 cores |
| RAM | 1.5 TB DDR5 |
| Storage | NVMe, ~6 GB/s read |
| Data | ~30 TB of zarr under `/local1/SETx_UIFL_Zarr/` |

Rough numbers: reading 30 TB at 6 GB/s takes at least ~85 minutes. Decompression and compute
will probably make the full run CPU-bound rather than I/O-bound. The main performance rule is
therefore **read every store exactly once** (see §6).

## 3. The data

### Layout

```
/local1/SETx_UIFL_Zarr/<variable>/<model>_<scenario>_<variable>_<start>-<end>.zarr
```

Example: `/local1/SETx_UIFL_Zarr/tas/BCC-CSM2-MR_ssp585_tas_2015-2100.zarr`

| Dimension | Values |
|---|---|
| Variables (9) | `hurs, huss, pr, rlds, rsds, sfcWind, tas, tasmax, tasmin` |
| Models (10) | `BCC-CSM2-MR, CESM2, CMCC-ESM2, CNRM-ESM2-1, EC-Earth3, FGOALS-g3, GFDL-CM4, MPI-ESM1-2-HR, MRI-ESM2-0, NorESM2-MM` |
| Scenarios (5) | `hist` (years `1950-2014`), `ssp126, ssp245, ssp370, ssp585` (years `2015-2100`) |

This gives 9 × 10 × 5 = **450 expected stores**. Not every store exists. For example, the
`tas` listing has no `GFDL-CM4_ssp126` or `GFDL-CM4_ssp370`. Finding the missing ones is
check 1.

### Resolved (from `BCC-CSM2-MR_ssp585_tas_2015-2100.zarr`, the sample `tas` store)

Confirmed against a real store; the other 8 variables are still unverified since the sample
only contains `tas`.

- Dimension names: `time, lat, lon`. 630 lat x 480 lon, 0.008333 deg (~1 km) resolution,
  covering roughly lat 28.5-33.75, lon -97.5 to -93.5 (SE Texas). `time` is 31390 steps,
  2015-01-01 to 2100-12-31.
- The data variable name inside the store matches the variable directory (`tas`), as assumed
  by `run.open_store`. The store also contains `tas_SE` (per-cell standard error from the
  downscaling method) and a scalar `crs` grid-mapping variable; both are outside `VARIABLES`
  and are correctly ignored since `open_store` selects only the named variable.
- Chunk shape: `(90, 630, 480)` — chunked along time only (90 days per chunk, full spatial
  extent per chunk, ~108 MB decompressed per chunk as float32), with one shorter final chunk.
  This matches the "chunks along time" case `xr.open_zarr(path, chunks={})` was written for;
  no rechunking needed.
- Units: `tas` is in K, matching the assumed `PHYSICAL_LIMITS`. Confirmed by the run: observed
  range was 183.16-317.25 K, comfortably inside `(180, 340)`, with `range_n_out = 0`.
- Calendar: `noleap`, decoded by xarray into a `CFTimeIndex`. `check_spatial_pattern`'s use of
  `da.indexes["time"]` and `.shift(time=-1)` both work unmodified against it.
- Zarr format: v3, with inline consolidated metadata. zarr-python 3.3.0 reads it; xarray emits
  a benign `ZarrUserWarning` that consolidated metadata isn't yet part of the v3 spec.
- Fill value: `NaN`, for `tas`/`lat`/`lon`/`crs` alike, confirming NaN is the missing-data
  marker. The land/ocean mask is fixed: `nan_count` was exactly 30582 (~10.1% of the 302400
  spatial cells) on every one of the 31390 days, so `missing_n_deviating = 0`.

## 4. QC checks

The top-level README defines these checks. The catalog is the full Cartesian product of
variable × model × scenario, and every check runs on every store.

### Check 1: File exists (`run.check_store`)

- Input: the expected path.
- Result: `status` is one of:
  - `missing`: the path does not exist.
  - `unreadable`: the path exists but opening it failed.
  - `failed`: it opened, but computing the checks raised an error (for example a corrupt
    chunk). The traceback is printed to stderr.
  - `ok`.
- For `unreadable` and `failed`, the error goes in the `error` column. Checks 2–4 produce
  results only when the status is `ok`.

### Check 2: Missing data per time step (`checks.check_missing_data`)

- For every time step, count the NaNs over the spatial dimensions. This gives
  `nan_count[time]`.
- The land/ocean mask is fixed, so every day should have exactly the **mode** of that series.
  Any day whose count differs from the mode deviates. There is no tolerance.
- Report `missing_mode`, `missing_n_deviating`, and `missing_pct_deviating`. The full
  `nan_count` series goes in the per-store `.nc`.

### Check 3: Within physical range (`checks.check_physical_range`, `checks.check_temperature_order`)

- Over all non-NaN values, report `range_min`, `range_max`, `range_n_out`, and
  `range_pct_out` against a per-variable `(low, high)` limit.
- Limits live in `PHYSICAL_LIMITS` in `checks.py`. The user accepted these values "for now".
  Recheck the units once the sample store arrives.

| Variable | Assumed units | low | high |
|---|---|---|---|
| tas, tasmax, tasmin | K | 180 | 340 |
| pr | kg m-2 s-1 | 0 | 0.03 |
| hurs | % | 0 | 100 |
| huss | kg kg-1 | 0 | 0.1 |
| rsds | W m-2 | 0 | 500 |
| rlds | W m-2 | 0 | 700 |
| sfcWind | m s-1 | 0 | 50 |

- **Temperature only:** also check `tasmin <= tas <= tasmax` in every cell on every day. This
  runs once per model/scenario from the `tas` store, and its results go in the `tas` row:
  `range_n_tasmin_gt_tas`, `range_n_tas_gt_tasmax`, `range_n_tasmin_gt_tasmax`. It is skipped
  (columns left empty) if `tasmin` or `tasmax` is missing. The three grids must match exactly,
  otherwise the `tas` store gets `failed`.

### Check 4: Spatial pattern (`checks.check_spatial_pattern`)

The goal is to catch days whose spatial field does not look like the fields next to them
(for example shifted, flipped, scrambled, or from the wrong day).

- Normalize each day's field over space: `z_t = (x_t - mean_t) / std_t`, ignoring NaNs.
- Compute the spatial pattern correlation between consecutive days:
  `r[t] = mean_space(z_t * z_{t+1})`, saved as `spatial_r`.
- A correlation is "low" if `r < median(r) - K * MAD(r)` (`SPATIAL_MAD_K = 5`, using the
  store's own series). Day `t` is **flagged** only if both `r[t-1]` and `r[t]` are low, meaning
  the day correlates poorly with both neighbours. One low value alone is usually a real
  weather change. The first and last days can never be flagged.
- A day with zero spatial variance (for example no rain anywhere) gives `r = NaN` and is never
  flagged.
- Report `spatial_n_flagged`, and `spatial_flagged_dates` (JSON only).

### Adding a new check

A check is a plain function in `qc/checks.py`:

```python
def check_name(da: xr.DataArray, variable: str) -> dict:
    """Plain language: what this check tests and what counts as a failure."""
    ...  # build LAZY results only; never call .compute() / .load() / .values here
    return {"name_metric": lazy_scalar, "name_series": lazy_1d_over_time}
```

- Values must be lazy: xarray objects, or `dask.delayed` for small post-processing on a
  reduced 1-D series (see `_mode_deviation`, `_flag_days`). `run.check_store` computes the
  values of all checks together in one call, so the store is read once (§6).
- Prefix keys with the check name. 1-D results over `time` go in the per-store `.nc`, and
  everything else goes in the JSON and `summary.csv`.
- Add the check to `CHECKS` in `run.py`, and add a clean case and a fault case to
  `tests/test_checks.py`.

## 5. Outputs (committed to git)

```
results/
  stores/<model>_<scenario>_<variable>.json   # status, error, scalar results; one per store
  stores/<model>_<scenario>_<variable>.nc     # time series: nan_count, spatial_r (status ok only)
  summary.csv                                  # one row per JSON (list fields dropped)
```

- Per-store files are written as each store finishes, with the JSON written **last**. On
  restart, stores that already have a JSON are skipped. Use `--overwrite` to redo them. A
  crash at store 300 does not force a re-read of 20 TB.
- `summary.csv` is rebuilt from all the JSONs at the end of every run.
- Size estimate: `spatial_r` is float64 and ~31k days long, so roughly 100 MB of `.nc` for
  the whole run. If that is too big for git, store it as float32.

## 6. Performance rules

- **One read per store.** All checks' lazy results are computed in one `dask.compute`. Never
  run a separate compute per check.
- **Never materialize a full store.** No `.load()`, `.values`, or `np.asarray` on the full
  array. Only reductions (per-time-step or global) get computed.
- Open with `xr.open_zarr(path, chunks={})` so dask uses the store's native chunks.
  Do not rechunk unless profiling shows it helps. Rechunking 30 TB is expensive.
- Process stores **one at a time**, giving the whole cluster to each one. This is simpler than
  running stores concurrently, and one store (~70 GB average) already has far more chunks than
  there are cores.
- Cluster: `dask.distributed.LocalCluster` with `--n-workers` (default 42) and
  `--threads-per-worker` (default 4). Tune after profiling the sample. Numcodecs decompression
  and numpy reductions release the GIL, so several threads per worker is fine.
- Check 4 casts to float64 (avoiding float32 precision loss over ~10⁸ cells), and its `shift`
  along time adds cross-chunk tasks. Only optimize this if profiling shows it dominates.
- Known extra cost: the temperature order check re-reads `tasmin` and `tasmax` (~2/9 extra
  I/O). It is marked `# ponytail:` in `run.py`.

## 7. Repo layout

Keep it this small. Do not add modules, config systems, or abstractions until they are
needed. `qc/` is a namespace package (no `__init__.py`).

```
qc/
  catalog.py        # VARIABLES, MODELS, SCENARIOS, YEARS, store_path()
  checks.py         # PHYSICAL_LIMITS, SPATIAL_MAD_K, one function per check (2, 3, 4)
  run.py            # CLI: cluster, loop stores, check 1 + CHECKS, write results, summary.csv
tests/
  test_checks.py    # synthetic small stores with injected faults
docs/llm/README.md  # this file
Dockerfile
requirements.txt    # direct dependencies, pinned
```

CLI:

```
python -m qc.run --root /local1/SETx_UIFL_Zarr --out results \
    [--variables tas pr] [--models CESM2] [--scenarios hist] \
    [--n-workers 42] [--threads-per-worker 4] [--local-directory /fast/scratch] [--overwrite]
```

The filter flags let you run one variable or one store during development.

## 8. Testing

- Run `pytest` in the container. The tests use small **synthetic** stores (50 days × 20 × 20,
  chunked 10 days) written to `tmp_path`. They must never touch `/local1`.
- Each check has a clean case and an injected-fault case: missing path, garbage store,
  one extra NaN, one out-of-range value, one `tasmin > tas` cell, and one scrambled day.

```bash
docker build -t setx-qc:latest .
docker run --rm --entrypoint python setx-qc:latest -m pytest -q -p no:cacheprovider tests
# while iterating, mount the working tree instead of rebuilding:
docker run --rm -v "$PWD":/opt/qc:ro --entrypoint python setx-qc:latest -m pytest -q -p no:cacheprovider tests
```

### Sample store run (2026-09-14)

Ran the full CLI (default catalog, all 450 combinations) against a sandbox with the one real
store (`BCC-CSM2-MR_ssp585_tas_2015-2100.zarr`, ~48 GB on disk / ~38 GB decompressed) bind-mounted
at the expected `<root>/tas/...` path; the other 449 combinations correctly reported `missing`
(check 1 only touches `path.exists()`, so they added negligible time). `pytest` passed 7/7 first.

- **This sandbox is smaller than the dev machine described in §1** (15 GB RAM / 24 cores, vs.
  the documented 12 cores / 32 GB), so `--n-workers 3 --threads-per-worker 4` was already close
  to its ceiling here — treat the numbers below as sandbox-specific, not dev-machine numbers.
- Wall time for the one real store: ~6 minutes end to end (cluster startup + one store).
- Peak container memory: ~13.9 GiB (`docker stats`, sampled every 5s). Dask's own memory
  manager repeatedly paused/resumed a worker at its 5.02 GiB auto-computed limit (80% threshold)
  during the run — visible as `distributed.worker.memory` WARNING lines — but never OOM'd.
  Check 4 (`check_spatial_pattern`) is the most likely driver: it casts to float64.
- Every check passed clean on the real store: `missing_n_deviating = 0` (fixed NaN mask, exactly
  30582 of 302400 spatial cells masked every day), `range_n_out = 0` (183.16-317.25 K, inside the
  180-340 K limit). `check_spatial_pattern` flagged 804/31390 days (2.6%) — worth a look, but the
  flagged dates cluster heavily in May-September, consistent with real day-to-day variability from
  Gulf Coast convective weather rather than a data artifact; nothing here suggests the check itself
  needs retuning yet.
- `results/` in this repo holds the real output from this run (`git status` will show it as
  untracked/modified — not committed, per §10).

- After the sample store arrives, run `python -m qc.run` against it and record the wall time
  and peak memory in this file.

## 9. Dependencies and container

`requirements.txt` pins the direct dependencies: `numpy`, `xarray`, `zarr`,
`dask[distributed]`, `pandas`, `netCDF4`, `cftime`, `pytest`. Do not add anything else without
a clear reason. The image is `python:3.12-slim` with the code in `/opt/qc` (see `Dockerfile`).

```bash
docker build -t setx-qc:latest .
apptainer build setx-qc.sif docker-daemon://setx-qc:latest
apptainer run --bind /local1:/local1 --bind "$PWD/results":/results setx-qc.sif \
    --root /local1/SETx_UIFL_Zarr --out /results --local-directory /fast/scratch
apptainer exec setx-qc.sif python -m pytest -q -p no:cacheprovider /opt/qc/tests
```

Apptainer mounts the image read-only, so all output goes to a bound host directory. Point the
dask spill directory at fast local disk (`--local-directory`), not at `/tmp` inside the
container.

## 10. Rules for agents

- **Do not commit.** The user reviews every change first. When you reach a commit point, stop
  and propose a commit message.
- Do not run anything against `/local1` unless the user asks. Develop against synthetic data.
- Keep checks readable. Each check function should have a docstring that states, in plain
  language, what it tests and what counts as a failure.
- Mark deliberate shortcuts with a `# ponytail:` comment that names the limit and how to
  upgrade it.
- When the sample store arrives, resolve the **TBD** items in §3 and §4 and update this file.
- **dask gotcha (dask 2026.8.0):** `dask.compute(...)` on a mix of collection types (xarray or
  dask arrays together with `dask.delayed`) returns results **grouped by type, not in argument
  order**. `run.check_store` works around this by wrapping every value in `dask.delayed`
  first. Keep that wrapper. The tests catch it if it breaks.

## 11. Decisions (answered by the user)

1. Check 2 compares against the **mode**, and counts must match exactly.
2. The physical limits in §4 are good for now.
3. Check 4's "poor correlation with both neighbours" reading and the MAD outlier rule are
   confirmed.
4. `results/` is committed to git.
5. The cross-variable `tasmin <= tas <= tasmax` check is part of the physical range test, for
   temperature only.
