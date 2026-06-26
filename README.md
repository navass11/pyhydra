# pyhydra

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**pyhydra** is a modular Python library for hydrological and climate analysis. It covers the full workflow from raw data acquisition to stochastic analysis, flood modelling and uncertainty quantification.

---

## Modules

| Module | What it does |
|--------|-------------|
| `pyhydra.data_sources` | Download rainfall (GPM, ERA5, AEMET, OGIMET, PERSIANN), river discharge (GloFAS, GRDC, USGS), soils (SoilGrids), and climate projections (CMIP6 via CDS/ESGF) |
| `pyhydra.climate.bias_correction` | Delta method, quantile mapping (QM), quantile delta mapping (QDM), scaled distribution mapping (SDM) |
| `pyhydra.climate.spatial_analysis` | Regional frequency analysis, copulas, spatial interpolation, Bayesian hierarchical models |
| `pyhydra.climate.stochastic_generation` | Point stochastic rainfall (NSRP) and spatial fields (CoSMoS) |
| `pyhydra.climate.time_series` | Event extraction, extreme value analysis, synthetic event generation |
| `pyhydra.modeling.hydrology` | HEC-HMS and SWAT+ automation and calibration |
| `pyhydra.modeling.hydraulic` | SFINCS and HEC-RAS automation; Manning roughness Monte Carlo sensitivity |

---

## Installation

```bash
pip install git+https://github.com/navass11/pyhydra.git
```

For geospatial operations (requires GDAL):

```bash
pip install "pyhydra[geo] @ git+https://github.com/navass11/pyhydra.git"
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/navass11/pyhydra.git
cd pyhydra
pip install -e .
```

**Requirements:** Python ≥ 3.9, numpy, pandas, xarray, scipy, statsmodels, scikit-learn, matplotlib, tqdm, openturns, requests.

---

## Quick examples

### Bias correction

```python
from pyhydra.climate.bias_correction import BiasCorrection

bc = BiasCorrection(method="QDM")
corrected = bc.fit_transform(obs=obs_series, hist=hist_series, future=future_series)
```

### Regional frequency analysis

```python
from pyhydra.climate.spatial_analysis import RegionalFrequencyAnalysis

rfa = RegionalFrequencyAnalysis(distribution="GEV")
rfa.fit(annual_maxima)
return_levels = rfa.return_level([10, 50, 100, 500])
```

### Manning roughness Monte Carlo (SFINCS / HEC-RAS)

```python
from pyhydra.modeling.hydraulic.sensitivity import generate_manning_combinations

combinations = generate_manning_combinations("manning_dist.csv", n_samples=1000, seed=42)
# Returns a 1000×9 DataFrame, one row per simulation
```

### Download pilot case data

```python
from pyhydra.data.download import download_pilot_case

# Downloads 301 MB of SFINCS/HEC-RAS TIF ensembles to ./data/pilot_cases/manning_rugosidades/
download_pilot_case("manning_rugosidades")
```

### Download OGIMET SYNOP data

```python
from pyhydra.data_sources.rainfall import OgimetDownloader

dl = OgimetDownloader(station_id="08487", start="2020-01-01", end="2023-12-31")
df = dl.download()
```

---

## Pilot case data

Pilot case notebooks require large input files (TIF ensembles, simulation results). Download them with a single command — no Azure account needed:

```bash
# List available datasets
pyhydra-get-data

# Download a dataset (saves to HYDRA_DATA_DIR or ./data by default)
pyhydra-get-data manning_rugosidades
pyhydra-get-data los_corrales_buelna
pyhydra-get-data valencia_dana

# Custom destination
pyhydra-get-data manning_rugosidades --dest /path/to/data
```

Data is downloaded from an Azure File Share with a built-in read-only token. Files already present are skipped automatically (`--overwrite` to force re-download).

The notebooks look for data under `HYDRA_DATA_DIR/pilot_cases/<dataset>/`. Set this variable if you use a non-default location:

```bash
export HYDRA_DATA_DIR=/path/to/data    # Linux / macOS
set HYDRA_DATA_DIR=C:\path\to\data     # Windows
```

Or from Python:

```python
from pyhydra.data.download import download_pilot_case

download_pilot_case("manning_rugosidades", dest="/path/to/data")
```

---

## Notebooks

The `notebooks/` folder contains tutorial notebooks covering all modules and three end-to-end pilot cases:

```
notebooks/
├── climate/              bias correction, event extraction, extremes,
│   └── spatial_analysis/ stochastic generation, copulas, interpolation, RFA
├── data_sources/         GPM, ERA5, AEMET, OGIMET, PERSIANN, GloFAS, GRDC,
│                         USGS, SoilGrids, CDS, ESGF
├── modeling/
│   ├── hydraulic/        HEC-RAS, SFINCS, Manning sensitivity (7 notebooks)
│   └── hydrology/        HEC-HMS, SWAT+
└── pilot_cases/
    ├── los_corrales_buelna/   end-to-end flood risk (Besaya river, 8 notebooks)
    ├── manning_rugosidades/   Manning roughness sensitivity (7 notebooks)
    └── valencia_dana/         DANA extreme event analysis (2 notebooks)
```

Browse them directly on GitHub — notebooks render automatically in the browser.

---

## Running notebooks with Docker

The easiest way to run all notebooks — all dependencies (GDAL, openturns, hydromt-sfincs, …) come pre-installed:

```bash
git clone https://github.com/navass11/pyhydra.git
cd pyhydra
docker compose -f docker/docker-compose.yml up --build
```

Open [http://localhost:8888](http://localhost:8888) in your browser. Notebooks are mounted from the host so edits and outputs persist locally.

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## Citation

If you use pyhydra in your research, please cite:

```bibtex
@software{navas2025pyhydra,
  author  = {Navas, Salvador},
  title   = {pyhydra: a modular Python library for hydrological and climate analysis},
  year    = {2025},
  url     = {https://github.com/navass11/pyhydra},
}
```

---

## Contact

**Salvador Navas** — [salvador.navas@hidralab.com](mailto:salvador.navas@hidralab.com)

---

## Licence

MIT — see [LICENSE](LICENSE).
