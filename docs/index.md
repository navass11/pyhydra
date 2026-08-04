# pyhydra

**pyhydra** is a modular Python library for hydrological and climate
analysis: data acquisition, extreme-value and dependence statistics,
stochastic rainfall generation, climate bias correction, hybrid
downscaling, hydrological/hydraulic model automation, and deciding how to
propagate uncertainty through an expensive model. See
[`instalacion.md`](instalacion.md) for installation and the main
[`README`](../README.md) for a quick-start overview and citation
information.

The companion platform [`HYDRA`](https://github.com/navass11/HYDRA)
provides an optional web/API/Jupyter/Docker deployment layer built around
this package; pyhydra itself has no dependency on it and installs and
runs standalone via `pip`.

## Package structure

Every sub-package exposes its public API through its `__init__.py`, so the
import pattern is always `from pyhydra.<block>.<module> import <name>`.

```
src/pyhydra/
├── data_sources/               # Harmonised access to public hydro-climatic data
│   ├── rainfall/                # ERA5, GPM, PERSIANN-CCS, AEMET, OGIMET, Meteostat
│   ├── river_discharge/         # GloFAS, GRDC, USGS
│   ├── climate_change/          # CMIP6 via CDS and ESGF
│   └── soils/                   # SoilGrids download & USDA texture classification
│
├── climate/                    # Statistics, generation and downscaling
│   ├── time_series/             # Event extraction, GEV/GPD fitting, NSE/KGE/PBIAS
│   ├── spatial_analysis/        # Regional frequency analysis, copulas, kriging/GP
│   ├── stochastic_generation/   # NEOPRENE (point/spatial), CoSMoS random fields
│   ├── bias_correction/         # Delta, empirical/quantile-delta/scaled-distribution
│   └── hybrid_downscaling/      # Hydrograph classification, MaxDiss, flood-map recon.
│
├── modeling/                   # Hydrological/hydraulic model automation
│   ├── hydrology/                # HEC-HMS, SWAT+
│   └── hydraulic/                # SFINCS, HEC-RAS, Manning sensitivity
│
├── uq/                         # Deciding how to propagate uncertainty
│   ├── design.py                # MaxDiss reduction, pilots, Voronoi weighting
│   ├── emulability.py           # Cross-validated skill over a declared family set
│   ├── strategy.py              # The pilot-based rule: emulate or Monte Carlo
│   ├── stopping.py              # Sequential design with a three-state stop
│   └── metrics.py               # Population-weighted error measures
│
└── data/                        # `pyhydra-get-data`: pilot-case dataset downloads
```

This mirrors the module table in the project's SoftwareX paper (Navas &
del Jesus, 2026); see the paper for the underlying statistical/hydraulic
methodology and worked examples at three Spanish pilot sites.

## Extending pyhydra

Each block communicates with the others through standard data structures
(`pandas.DataFrame`, `xarray.Dataset`, GeoTIFF/`geopandas.GeoDataFrame`)
rather than block-specific formats, so a new data source, statistical
method or model adapter can be added without changing the others. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the contribution workflow and
`tests/` for the existing test suite's conventions.
