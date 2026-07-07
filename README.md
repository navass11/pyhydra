# pyhydra

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.20932555.svg)](https://doi.org/10.5281/zenodo.20932555)

**pyhydra** is a modular Python library for hydrological and climate analysis.
It provides reusable components for data acquisition, time-series analysis,
extreme-value statistics, stochastic rainfall generation, bias correction,
hybrid downscaling, hydrological/hydraulic model automation and flood-risk
post-processing.

The package is currently released as **v0.1.0** and should be treated as a
research software package in active development. Some modules are mature enough
for reproducible workflows, while others provide adapters around external models
or data services and depend on third-party executables, credentials or local
project files.

## Scope

pyhydra is designed to make complex hydrology and climate workflows easier to
compose from Python:

- download and harmonise hydroclimatic data from public services;
- extract events and fit extreme-value distributions;
- generate stochastic rainfall and synthetic flood-event catalogues;
- apply climate-change bias-correction methods;
- prepare and run selected external hydrological and hydraulic models;
- analyse ensembles, uncertainty and spatial flood outputs.

The companion platform [`HYDRA`](https://github.com/navass11/HYDRA) provides the
web interface, FastAPI backend, Jupyter environment and deployment layer built
around this package.

## Modules

| Module | Main capabilities |
| --- | --- |
| `pyhydra.data_sources.rainfall` | GPM, ERA5, AEMET, OGIMET, Meteostat and PERSIANN access helpers, plus CSV loaders for station data. |
| `pyhydra.data_sources.river_discharge` | GloFAS, GRDC and USGS discharge data access utilities. |
| `pyhydra.data_sources.climate_change` | Copernicus/CDS and ESGF helpers for climate projection workflows. |
| `pyhydra.data_sources.soils` | SoilGrids download and preprocessing helpers. |
| `pyhydra.climate.time_series` | Event extraction, time-series statistics and extreme-value analysis. |
| `pyhydra.climate.bias_correction` | Delta method, empirical quantile mapping, quantile delta mapping and scaled distribution mapping. |
| `pyhydra.climate.spatial_analysis` | Regional frequency analysis, copulas, interpolation and Bayesian hierarchical modelling. |
| `pyhydra.climate.stochastic_generation` | Point rainfall generation and spatial stochastic fields, including NSRP/STNSRP-style workflows and CoSMoS-based fields when the corresponding dependencies are available. |
| `pyhydra.climate.hybrid_downscaling` | Flood-event classification, synthetic event generation, MaxDiss-type selection, hydrograph reconstruction, map interpolation and return-period mapping. |
| `pyhydra.modeling.hydrology` | HEC-HMS file generation/runtime helpers and SWAT+ climate input generation/execution helpers. |
| `pyhydra.modeling.hydraulic` | HEC-RAS project-file/runtime helpers, SFINCS setup/execution helpers and Manning roughness sensitivity utilities. |

External models are not bundled with pyhydra. HEC-HMS, HEC-RAS, SWAT+ and
SFINCS workflows require the corresponding model installation, executable or
project structure.

## Installation

Install the latest code from GitHub:

```bash
pip install git+https://github.com/navass11/pyhydra.git
```

For development:

```bash
git clone https://github.com/navass11/pyhydra.git
cd pyhydra
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional geospatial dependencies can be installed with:

```bash
pip install "pyhydra[geo] @ git+https://github.com/navass11/pyhydra.git"
```

GDAL-based installations can be platform-sensitive. If GDAL installation fails,
use a Conda environment or the Docker notebook environment described below.

## Quick Examples

### Bias correction

```python
from pyhydra.climate.bias_correction import BiasCorrection

corrected = BiasCorrection.fit_transform(
    obs=obs_series,
    hist=hist_series,
    future=future_series,
    var="pr",
    method="qdm",
)
```

### Extreme-value analysis

```python
from pyhydra.climate.time_series import fit_gev

params = fit_gev(annual_maxima, method="mle")
```

### Regional frequency analysis

```python
from pyhydra.climate.spatial_analysis import fit_regional_gev, regional_return_levels

regional_params, index_floods = fit_regional_gev(annual_maxima_by_station)
return_levels = regional_return_levels(
    annual_maxima_by_station,
    T_values=[10, 50, 100, 500],
)
```

### Manning roughness ensemble

```python
from pyhydra.modeling.hydraulic.sensitivity import generate_manning_combinations

combinations = generate_manning_combinations(
    "manning_dist.csv",
    n_samples=1000,
    seed=42,
)
```

## Pilot-Case Data

Some notebooks require large input files, model projects or precomputed model
outputs that are not stored directly in the Git repository. Available pilot-case
datasets can be downloaded with the command-line helper installed with pyhydra:

```bash
# List available datasets
pyhydra-get-data

# Download one dataset into $HYDRA_DATA_DIR or ./data
pyhydra-get-data manning_rugosidades
pyhydra-get-data m30_manzanares
pyhydra-get-data los_corrales_buelna
pyhydra-get-data valencia_dana

# Use a custom destination
pyhydra-get-data manning_rugosidades --dest /path/to/data

# Force re-download of files already present
pyhydra-get-data manning_rugosidades --overwrite
```

The same operation is available from Python:

```python
from pyhydra.data.download import download_pilot_case, list_pilot_cases

print(list_pilot_cases())
download_pilot_case("manning_rugosidades", dest="/path/to/data")
```

Downloads are served from a read-only Azure File Share token embedded in the
helper. Files already present locally are skipped unless `--overwrite` is used.

By convention, notebooks look for pilot-case files under:

```text
${HYDRA_DATA_DIR}/pilot_cases/<case_name>/
```

If `HYDRA_DATA_DIR` is not defined, notebooks fall back to `./data` relative to
the repository root. Large downloaded products should stay in that data
workspace, not in Git. The full platform repository
[`HYDRA`](https://github.com/navass11/HYDRA) documents the broader data
workspace structure used by the web/Jupyter deployment.

## Notebooks

The `notebooks/` directory contains executable examples for the main modules and
end-to-end pilot cases:

```text
notebooks/
|-- climate/                    bias correction, events, extremes, stochastic generation
|   `-- spatial_analysis/       RFA, copulas, interpolation, Bayesian examples
|-- data_sources/               rainfall, discharge, soils and climate-change data
|-- modeling/
|   |-- hydrology/              HEC-HMS and SWAT+
|   `-- hydraulic/              HEC-RAS, SFINCS and Manning sensitivity
`-- pilot_cases/
    |-- los_corrales_buelna/    Besaya flood-risk workflow
    |-- m30_manzanares/         Madrid Calle 30 / Manzanares workflow
    |-- manning_rugosidades/    SFINCS/HEC-RAS roughness sensitivity workflow
    `-- valencia_dana/          October 2024 DANA extreme-event analysis
```

## Docker Notebook Environment

The repository includes a Docker-based JupyterLab environment for running the
notebooks with the required scientific stack. This is the recommended option
when local installation of GDAL, geospatial libraries or model wrappers is
problematic.

### Requirements

- Docker Desktop on macOS/Windows, or Docker Engine on Linux.
- Docker Compose v2, available as `docker compose`.
- Internet access during the first build, because the image installs Python
  dependencies and CoSMoS from GitHub.

Check the installation:

```bash
docker --version
docker compose version
```

### Build and Run

```bash
git clone https://github.com/navass11/pyhydra.git
cd pyhydra
docker compose -f docker/docker-compose.yml up --build
```

Open <http://localhost:8888>. JupyterLab is configured without token or password
for local development, so do not expose this container directly to the internet.

The compose file mounts:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `./notebooks` | `/pyhydra/notebooks` | Editable notebooks; changes persist in Git working tree. |
| `./data` | `/pyhydra/data` | Data workspace for downloads, model inputs and generated outputs. |

### Download Pilot Data Inside Docker

The package is installed in editable mode inside the image, so the
`pyhydra-get-data` command is available from a shell in the running container:

```bash
docker compose -f docker/docker-compose.yml exec jupyter pyhydra-get-data
docker compose -f docker/docker-compose.yml exec jupyter pyhydra-get-data manning_rugosidades
```

Downloaded files are written to `/pyhydra/data/pilot_cases/...`, which is mapped
to `./data/pilot_cases/...` on the host.

### Stop, Rebuild and Clean Up

Stop the container:

```bash
docker compose -f docker/docker-compose.yml down
```

Rebuild after dependency or Dockerfile changes:

```bash
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up
```

Run in the background:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

View logs:

```bash
docker compose -f docker/docker-compose.yml logs -f jupyter
```

### Notes

- The Dockerfile installs `pyhydra` with `pip install -e .`, so edits to mounted
  notebooks are immediate, while package-code changes may require rebuilding the
  image depending on how they are made.
- HEC-HMS, HEC-RAS, SWAT+ and SFINCS workflows may still require external model
  executables or project files. The Docker image provides the Python environment,
  not commercial or platform-specific model installations.
- If port `8888` is already in use, change the host port in
  `docker/docker-compose.yml`, for example `"8890:8888"`.

## Testing

```bash
pip install pytest
pytest tests/
```

Some tests and notebooks require network access, external credentials, large
input data or installed model executables. Keep those constraints in mind when
interpreting local test results.

## Citation

If you use pyhydra in research, cite the Zenodo release:

```bibtex
@software{navas2026pyhydra,
  author    = {Navas Fernández, Salvador},
  title     = {pyhydra: a modular Python library for hydrological and climate analysis},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20932555},
  url       = {https://github.com/navass11/pyhydra}
}
```

The citation metadata is also available in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Salvador Navas - [salvador.navas@alumnos.unican.es](mailto:salvador.navas@alumnos.unican.es)
