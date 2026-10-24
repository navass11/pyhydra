# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- Pilot-case notebooks run when the shared data folder (`HYDRA_DATA_DIR`) is
  read-only: `los_corrales_buelna/03_extreme_value_analysis` writes its outputs
  to a session copy and reads `pma_idw_daily.csv` from the session copy or the
  shared folder, and `m30_manzanares/03_copula_generation` and
  `04_maxdiss_selection` write `events_synthetic_pyhydra.npz` and
  `events_selected_pyhydra_idx.npy` to the session directory when the shared
  `events/` folder is read-only (and read them back from there).
- `los_corrales_buelna/09_uncertainty_propagation_strategy`: a `print` whose
  continuation line started with `%` was taken for an IPython magic by
  line-based notebook runners; the call is now on one line.
- `data_sources/river_discharge/USGS_download`: the two plotting cells no longer
  raise `NameError` when `RUN_DOWNLOADS = False`; they print a short notice.
- `modeling/hydrology/SWAT`: `parse_swatplus_station_file` accepts the `long`
  column that the public SWATdata LREW demo writes, as well as `lon`.

## [0.3.1] - 2026-10-23

### Changed
- `pyhydra-get-data` and `pyhydra.data.download` now download the pilot-case
  data from the open Zenodo record 10.5281/zenodo.23599126 instead of a
  private Azure file share. Dataset names, command-line options, destination
  layout (`<dest>/pilot_cases/<dataset>/`) and behaviour are unchanged; files
  already present are skipped unless `--overwrite` is used. Zip files are
  checked against their MD5 and partial downloads are resumed. The read-only
  Azure token is no longer embedded in the code.
- The archive adds the files the notebooks need but the original share did
  not have: `manning_rugosidades/dem_corrales.asc` (roughness notebooks 05 and
  06) and the legacy matrices and `stan_reference/` results of Valencia
  (notebook 03). `los_corrales_buelna` has exactly the 5,910 files it had.

### Fixed
- `best_distribution` (`modeling.hydraulic.sensitivity`) and
  `GaussianCopulaSampler.from_data` (`climate.spatial_analysis.copulas`) failed
  with SciPy 1.18 (`kstest(x, 'norm', args=...)` raises `TypeError` in `ndtr`);
  they now pass the distribution's `cdf`. This broke the Manning notebooks.
- Pilot-case notebooks, checked against the Zenodo data: Besaya 05 used the
  pandas offset alias `'M'`, removed in pandas 3; Manning 01 read
  `manning_mapping.csv` although it is not distributed; Manning 05 treated the
  DEM as available when only the HEC-RAS rasters were found; Manning 06 used
  the failing `kstest` call; Valencia 03 used `boxplot(labels=...)`, removed
  from matplotlib. With these fixes and the archived data, all 25 pilot-case
  notebooks run to the end. Those that need HEC-HMS or HEC-RAS skip the engine
  steps by design.

### Added
- `notebooks` extra (seaborn, contextily, xlrd, openpyxl): packages the
  pilot-case notebooks import but no extra declared. Without `xlrd`, Besaya 01
  wrote empty gauge files without any error. Included in `all`.
- `src/pyhydra/data/zenodo_index.json`: index of the record's files,
  `tests/test_data_download.py` (6 tests, no network access) and
  `tests/test_sensitivity_distributions.py`.

## [0.3.0] - 2026-10-23

### Changed
- **Licence: MIT to GPL-3.0-only.** pyhydra builds on code released under the
  GNU General Public License v3, which must be kept in derived works.
  `LICENSE` and `LICENSE.txt` now hold the GPLv3 text, and the package
  metadata (`pyproject.toml`, `CITATION.cff`, `.zenodo.json`, README) follows.
  Releases up to 0.2.2 were published under the MIT licence and stay as they
  are.

## [0.2.2] - 2026-10-14

### Added
- `LICENSE.txt`: an identical copy of `LICENSE` (MIT), because SoftwareX
  requires a `Licence.txt` file in the repository linked from the article.

### Fixed
- Removed a stray `wiki` gitlink (no `.gitmodules`) that made the checkout
  clean-up step of the CI workflow fail with `exit code 128`, and added
  `wiki/` to `.gitignore`.

## [0.2.1] - 2026-10-13

### Added
- `FloodEventCopula` options, all backwards compatible (defaults reproduce
  the previous behaviour):
  - `families`: restricts the BIC marginal search to a list of openturns
    families, or to a per-variable dict.
  - `zero_inflated_vars` / `zero_threshold`: mixture marginals
    `p0·δ0 + (1 − p0)·F+` for variables with a probability mass at zero,
    such as dry gauges within a storm event. A randomised PIT is used, with
    one latent uniform per event shared across those variables so that
    simultaneous zeros stay concordant.
  - `random_state`: seeds both the PIT and sampling; `verbose` is also new.
- `fit_marginal` accepts `families` and `verbose`.
- `tests/test_flood_event_copula.py` (6 tests).
- Besaya notebook `09_uncertainty_propagation_strategy` applying `pyhydra.uq`
  to the real MaxDiss design, and Valencia notebook
  `03_real_regional_study_pyhydra` (regional/hierarchical GEV).

### Changed
- `FloodEventCopula.sample` evaluates marginal quantiles vectorised: more
  than an order of magnitude faster on an 84-variable model.
- M-30 pilot case:
  - Notebook 03 now fits the full 84-variable Gaussian copula with
    `FloodEventCopula` and validates the regenerated catalogue against its
    calibration set. The previous version only re-plotted archived
    synthetic events.
  - Notebook 04 runs `pyhydra` MaxDiss for 625 events in the full
    84-dimensional space.
  - Notebooks 05–06 now state explicitly that their design depths are
    archived results of Navas et al. (2024), not computed by `pyhydra`.

### Fixed
- `FloodEventCopula` with the default all-family BIC search could select
  heavy-tailed marginals (e.g. Fréchet) for zero-inflated rainfall. Synthetic
  values of order 1e70 mm resulted; the new `families` / `zero_inflated_vars`
  options prevent it.

### Removed
- Stale duplicate Besaya notebook `07_hec_ras_hydraulics_run.ipynb`.

## [0.2.0] - 2026-08-04

### Added
- `pyhydra.uq`: a decision layer for uncertainty propagation through expensive
  hydraulic models. Given an ensemble and a simulation budget it measures, from
  a pilot run charged against that budget, whether the model's response can be
  emulated at the design size the budget allows, and routes the run to
  reduction-plus-emulation or to direct Monte Carlo accordingly
  - `emulability()` -- best cross-validated R^2 over a declared family set;
    the same procedure serves as the pilot diagnostic and as the selection step
    of the surrogate that is actually built, which is what makes the threshold
    interpretable
  - `select_strategy()` / `propagate()` -- the rule, by hand or end to end.
    Both branches are held to `n_max` solver evaluations: because `maxdiss_order()`
    is nested, an emulated run keeps its pilot points, while a Monte Carlo
    fallback has spent them and draws only `n_max - n_pilot`
  - `maxdiss_order()`, `pilot_design()` -- nested maximum-dissimilarity design
    reduction over an arbitrary input matrix. Deliberately *not* named
    `maxdiss`: `climate.hybrid_downscaling.reconstruction.maxdiss` already
    exists, is specific to synthetic flood events (one column treated as
    circular, one seed per hydrograph shape type) and returns a different order
    on the same data. The two are not interchangeable and the names now say so
  - `voronoi_weights()`, `error_metrics()` -- population weighting, so error is
    reported over the ensemble rather than over a design that over-samples its
    tails by construction
  - `sequential_design()` -- incremental design growth with a three-state
    stopping rule that distinguishes a flattened error curve from a rising one
  - `reduce_and_emulate()` -- reconstruction that keeps simulated values exactly
    and reports which family was selected
- 34 tests covering the properties the rule's validity rests on: design
  nestedness, budget accounting on both branches, diagnostic/estimator identity,
  and refusal to route unlearnable responses to an emulator


## [0.1.7] - 2026-07-30

### Fixed
- `docs/index.md` and `docs/instalacion.md` still described the pre-split
  HYDRA monorepo (PascalCase `Data_Sources/`/`Climate/`/`Modeling/` at the
  repo root, `git clone HYDRA.git`) rather than pyhydra as it exists today
  (`src/pyhydra/data_sources/`, etc., standalone `pip install`).
  Rewritten to match the current package layout and to point at the
  README as the single source of truth for install instructions, rather
  than duplicating them and risking the same kind of drift fixed in
  v0.1.5/v0.1.6

## [0.1.6] - 2026-07-30

### Added
- `statistics`, `geospatial` and `models` installable `pip` extras
  (e.g. `pip install "pyhydra[statistics]"`), replacing the previous
  undocumented gap where every "Extended" dependency in the README had to
  be installed by hand; `all` now genuinely covers all of them (previously
  it was just an alias for `geo`)
- CI now runs the test suite on a Python 3.9-3.12 matrix, not just 3.12,
  and installs PyMC (the C-compiler toolchain issue that motivated
  excluding it was specific to local macOS setups, not Ubuntu/CI)

### Fixed
- `test_fit_regional_gev_bayes_returns_posterior_dataframe` failed outright
  on Python 3.9: pip resolves an older PyMC there (PyMC >=6 requires
  Python >=3.12), and that older PyMC's import path calls a NumPy API
  removed in NumPy 2.0. The test now treats any import or fit failure in
  this optional dependency the same way: a skip, not a failure
- `pyproject.toml`'s `Homepage` URL pointed at hidralab.com; now points at
  the repository, consistent with the authorship fix in v0.1.5

## [0.1.5] - 2026-07-30

### Fixed
- README.md, CITATION.cff and .zenodo.json still described v0.1.0 (badge,
  "currently released as", citation DOI/version, single-author metadata)
  despite four releases since; every prior release inherited this drift
  because these files were never part of what got checked before tagging.
  Badges and citation DOIs now point at the *concept* DOI (which always
  resolves to the latest version), so this cannot recur; author list and
  affiliations now match the Zenodo record (Salvador Navas, IH Cantabria;
  Manuel del Jesus, Universidad de Cantabria)
- README now documents installing the exact tagged version
  (`pip install "pyhydra @ git+...@vX.Y.Z"`), not just `main`

## [0.1.4] - 2026-07-30

### Changed
- Re-executed `notebooks/pilot_cases/valencia_dana/01_data_exploration.ipynb`
  and `02_extreme_value_analysis.ipynb` end to end against a clean data
  download, so their stored outputs match the current codebase; removed
  diagnostic prints that leaked the executing machine's absolute local path
- Added a Bayesian (MCMC) $T{=}100$yr return-level cell to notebook 02,
  used to reproduce the paper's Figure 5

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
