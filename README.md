# SETx-UIFL-Downscaled-Quality-Control
Quality control scripts and results produced for the SETx UIFL 1km downscaled climate dataset.


Add initial QC framework: catalog, checks, runner, tests, container

- qc/catalog.py: expected variable x model x scenario stores and paths
- qc/checks.py: missing-data (mode), physical range + tasmin<=tas<=tasmax,
  spatial pattern (neighbour-day correlation, MAD outliers)
- qc/run.py: CLI; file-exists check, one dask.compute per store,
  resumable per-store JSON/NetCDF output, summary.csv
- tests/test_checks.py: synthetic stores with injected faults
- Dockerfile + pinned requirements.txt (Apptainer-convertible)
- docs/llm/README.md: spec updated to match implementation and decisions
