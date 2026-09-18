# SETx-UIFL Downscaled Quality Control

> The per-store JSON/NetCDF outputs and `summary.csv` (~200 MB total) are not committed to
> this repo. They're attached as assets on the corresponding GitHub release instead (see
> this repo's Releases page for results from this quality control pipeline).

Quality control scripts and results produced for the SETx UIFL 1km downscaled climate dataset.

- `qc/catalog.py`: expected variable x model x scenario stores and paths
- `qc/checks.py`: missing-data (mode), physical range + `tasmin<=tas<=tasmax`,
  spatial artifacts (banding, roughness, repeated extreme values)
- `qc/run.py`: CLI; file-exists check, one `dask.compute` per store,
  resumable per-store JSON/NetCDF output, `summary.csv`
- `tests/`: synthetic stores with injected faults, plus unit tests against
  hand-computed values
- `Dockerfile` + pinned `requirements.txt` (Apptainer-convertible)
- `docs/llm/README.md`: full spec for LLM coding agents

```bash
python -m qc.run --root /local1/SETx_UIFL_Zarr --out results \
    [--variables tas pr] [--models CESM2] [--scenarios hist] \
    [--n-workers 42] [--threads-per-worker 4] [--local-directory /fast/scratch] [--overwrite]
```

Expected outputs, written under `--out` (`results/` by default):

```
results/
  stores/<model>_<scenario>_<variable>.json   # status (missing/unreadable/failed/ok),
                                               #   error, and scalar check results; one per store
  stores/<model>_<scenario>_<variable>.nc     # per-day time series (status ok only): nan_count,
                                               #   banding_score, roughness_score,
                                               #   repeated_daily_min, repeated_daily_max
  summary.csv                                 # one row per JSON, rebuilt from all JSONs at the
                                               #   end of every run (list-valued fields dropped)
```

Each store's `.json`/`.nc` pair is written as soon as that store finishes, with the JSON
written last. On restart, any store that already has a JSON is skipped; pass
`--overwrite` to redo it anyway.

## Running from source

If your environment already has Python 3.12 and can install packages, you can run directly from source:

```bash
pip install -r requirements.txt
python -m qc.run --root /local1/SETx_UIFL_Zarr --out results
```

## Building the Docker container and converting to `.sif`

The image is `python:3.12-slim` with the code copied to `/opt/qc`. On your local machine with `docker` and `apptainer` installed:

```bash
docker build -t setx-qc:latest .
apptainer build setx-qc.sif docker-daemon://setx-qc:latest
```

Running the `.sif` on the target machine:

```bash
apptainer run --bind /local1:/local1 --bind "$PWD/results":/results setx-qc.sif \
    --root /local1/SETx_UIFL_Zarr --out /results --local-directory /fast/scratch
```

Apptainer mounts the image read-only, so all output must go to a bound host
directory. Point the dask spill directory at fast local disk (`--local-directory`),
not at `/tmp` inside the container.

## (Dev Guide) Running tests

With Docker:

```bash
docker build -t setx-qc:latest .
docker run --rm --entrypoint python setx-qc:latest -m pytest -q -p no:cacheprovider tests
# while iterating, mount the working tree instead of rebuilding:
docker run --rm -v "$PWD":/opt/qc:ro --entrypoint python setx-qc:latest -m pytest -q -p no:cacheprovider tests
```

With Apptainer:

```bash
apptainer exec setx-qc.sif python -m pytest -q -p no:cacheprovider /opt/qc/tests
```

From source (dependencies already installed):

```bash
pytest
```

Tests never touch `/local1` and instead use small synthetic zarr stores.

## (Dev Guide) Direct your LLM coding agent to `docs/llm`

`docs/llm/README.md` is the spec written specifically for LLM coding agents:
it documents the data layout, each check's exact behavior, performance rules,
and decisions made. Point any coding agent at it before it touches this repo, for example
by adding an instruction such as:

> Read `docs/llm/README.md` before you change anything in this repo.

