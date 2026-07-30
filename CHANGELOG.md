# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.3] - 2026-07-30

### Fixed
- Moved the bundled OGIMET station catalogue from a stray repo-root
  `Data_Sources/` (inconsistent with the project's `notebooks/data_sources/`
  convention, and dropped from `.gitignore` only in 0.1.2) to
  `notebooks/data_sources/rainfall/estaciones_ogimet_all.csv`, alongside the
  other data-source download notebooks; removed the unused legacy
  `OGIMET.py` scraping script that lived next to it

## [0.1.2] - 2026-07-30

### Fixed
- CI (added in 0.1.1) surfaced three regressions invisible on the maintainer's local
  environment, where every optional dependency happened to be installed:
  - `ogimet.get_default_ogimet_stations_csv`: repo-root resolution was off by one
    directory level after the 0.1.1 `src/` layout migration, so the bundled station
    catalogue could no longer be found
  - the bundled OGIMET station catalogue (`Data_Sources/Rainfall/OGIMET/data/`) was
    excluded by `.gitignore` and had never actually been shipped in a tagged release
  - `ogimet.process_all_meteorological_variables` and `aemet.AemetCSVLoader.load_series_data`
    used pandas patterns (in-place dtype-changing `.loc` assignment; unsorted
    `os.listdir`) that raised or returned non-deterministic column order on pandas
    versions newer than the maintainer's local install
- CI now installs `lmoments3`, required (not just optional) for the default
  L-moment regional-frequency path exercised by several `test_rfa_regional` tests

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
