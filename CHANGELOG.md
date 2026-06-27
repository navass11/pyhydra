# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **M30 Manzanares pilot case** — 6 end-to-end notebooks demonstrating the multivariate
  copula methodology from Navas et al. (2024) (*Ingeniería del Agua* 28(4), 263–279,
  DOI 10.4995/ia.2024.22293):
  - `01_rain_data` — gauge network, elevation–rainfall relationship, event characterisation
  - `02_classification` — hyetograph shape classification (PCA + K-Means)
  - `03_copula_generation` — Gaussian copula over 68 variables (4 params × 17 gauges)
  - `04_maxdiss_selection` — MaxDiss representative event selection
  - `05_hms_ras_simulation` — HEC-HMS + HEC-RAS 1D hydraulic simulation results
  - `06_knn_return_periods` — kNN depth reconstruction and return period curves
- Dataset `m30_manzanares` available via `pyhydra-get-data m30_manzanares`

## [0.1.0] - 2025-06-17

### Added
- Initial release migrated from HYDRA monorepo
- `pyhydra.climate`: bias correction, spatial analysis, stochastic generation, time series
- `pyhydra.data_sources`: rainfall (GPM, ERA5, AEMET, OGIMET, PERSIANN), river discharge (GloFAS, GRDC, USGS), soils (SoilGrids), climate change (CDS, ESGF)
- `pyhydra.modeling`: HEC-HMS, SWAT, HEC-RAS, SFINCS automation; Manning roughness sensitivity
- 26 tutorial notebooks covering all modules
- Full test suite
