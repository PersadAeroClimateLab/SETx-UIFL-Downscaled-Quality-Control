"""Expected zarr stores: every variable x model x scenario combination."""

from pathlib import Path

VARIABLES = ["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
MODELS = [
    "BCC-CSM2-MR", "CESM2", "CMCC-ESM2", "CNRM-ESM2-1", "EC-Earth3",
    "FGOALS-g3", "GFDL-CM4", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM",
]
YEARS = {
    "hist": "1950-2014",
    "ssp126": "2015-2100",
    "ssp245": "2015-2100",
    "ssp370": "2015-2100",
    "ssp585": "2015-2100",
}
SCENARIOS = list(YEARS)


def store_path(root, variable, model, scenario):
    """e.g. <root>/tas/BCC-CSM2-MR_ssp585_tas_2015-2100.zarr"""
    return Path(root) / variable / f"{model}_{scenario}_{variable}_{YEARS[scenario]}.zarr"
