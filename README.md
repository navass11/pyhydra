# pyhydra

**pyhydra** is a modular Python library for hydrological and climate analysis. It provides tools for data downloading and processing, statistical climate analysis, stochastic generation, and support for hydrological and hydraulic models.

## Installation

```bash
pip install git+https://github.com/SalvaNavas/pyhydra.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/SalvaNavas/pyhydra.git
cd pyhydra
pip install -e .
```

## Package structure

```
pyhydra/
├── climate/
│   ├── bias_correction/       # Delta method, quantile mapping (QM, QDM, SDM)
│   ├── spatial_analysis/      # Regional frequency analysis, copulas, interpolation, Bayesian hierarchical
│   ├── stochastic_generation/ # Point and spatial stochastic rainfall generation (NSRP, CoSMoS)
│   └── time_series/           # Event extraction, extremes, synthetic generation
├── data_sources/
│   ├── climate_change/        # CMIP6 via CDS/ESGF
│   ├── rainfall/              # GPM, PERSIANN, ERA5, AEMET, OGIMET
│   ├── river_discharge/       # GloFAS, GRDC, USGS
│   └── soils/                 # SoilGrids
└── modeling/
    ├── hydrology/             # HEC-HMS, SWAT automation
    └── hydraulic/             # SFINCS, HEC-RAS automation + Manning sensitivity
```

## Requirements

- Python ≥ 3.9
- numpy, pandas, xarray, scipy, statsmodels, scikit-learn, matplotlib, tqdm, openturns

Optional:
- gdal (for geospatial operations): `pip install pyhydra[geo]`

## Examples

See the `examples/` folder for standalone usage scripts.

## Tests

```bash
pytest tests/
```

## Licence

MIT
