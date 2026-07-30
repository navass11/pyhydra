# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.1] - 2026-07-30

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
- SWAT+ calibration API (`calibrate_swat_sceua`), DSS/SWAT+ output readers, and
  Hosking–Wallis discordancy/heterogeneity diagnostics for regional frequency analysis
- GitHub Actions workflow running the `pytest` suite on every push/PR

### Changed
- Package layout migrated from `pyhydra/` to `src/pyhydra/` (setuptools src-layout)
- `test_fit_regional_gev_bayes_returns_posterior_dataframe` marked
  `optional_dependency` and guarded with `importorskip`/skip-on-failure, so a missing
  or broken PyMC/pytensor C-compiler toolchain is reported as a skip, not a failure

## [0.1.0] - 2025-06-17

### Added
- Initial release migrated from HYDRA monorepo
- `pyhydra.climate`: bias correction, spatial analysis, stochastic generation, time series
- `pyhydra.data_sources`: rainfall (GPM, ERA5, AEMET, OGIMET, PERSIANN), river discharge (GloFAS, GRDC, USGS), soils (SoilGrids), climate change (CDS, ESGF)
- `pyhydra.modeling`: HEC-HMS, SWAT, HEC-RAS, SFINCS automation; Manning roughness sensitivity
- 26 tutorial notebooks covering all modules
- Full test suite
