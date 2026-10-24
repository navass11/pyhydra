# Pilot-case notebooks: how to run them

Four case studies, 25 notebooks. Each folder is a sequence: run its notebooks in
numeric order from a Jupyter session started in this repository.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: pip install "pyhydra[all] @ git+https://github.com/navass11/pyhydra.git@v0.3.1"
pip install jupyterlab
```

`all` includes the `notebooks` extra (seaborn, contextily, xlrd, openpyxl).
CoSMoS_py (used for spatial random fields) is not on PyPI and is installed
separately from its own repository if a notebook asks for it.

## 2. Get the data

```bash
export HYDRA_DATA_DIR=/path/to/data
pyhydra-get-data <dataset>       # downloads from Zenodo 10.5281/zenodo.23599126, checks MD5
jupyter lab
```

| Folder | Dataset | Size | External engine | Without the engine |
|---|---|---|---|---|
| `valencia_dana/` (3) | `valencia_dana` | 30 MB | none | all 3 notebooks run in full |
| `m30_manzanares/` (6) | `m30_manzanares` | 13 MB | HEC-HMS + HEC-RAS (step 05) | 03 and 04 re-run copula generation and MaxDiss; the design depths are read from archived results |
| `manning_rugosidades/` (7) | `manning_rugosidades` | 328 MB | SFINCS, HEC-RAS | aggregation re-runs from the archived 2 x 1,000 outputs; re-simulation needs the engines |
| `los_corrales_buelna/` (9) | `los_corrales_buelna` | 18 GB | HEC-HMS, HEC-RAS (04, 05, 07, 08) | engine steps are skipped; the statistical steps run |

Some files derive from AEMET, CHC, SIAR, AVAMET and CORDEX records and keep their
providers' terms of use (see the README of the Zenodo record).

## 3. Switches

Engine and heavy steps are off by default and are enabled with environment
variables, set before starting Jupyter:

| Variable | Enables |
|---|---|
| `HYDRA_RUN_HEC_HMS=1`, `HYDRA_RUN_HEC_RAS=1` | the HEC-HMS / HEC-RAS runs (needs the program, see below) |
| `HYDRA_RUN_MODELS=1` | the HMS/RAS scenario generation of `los_corrales_buelna/08` |
| `HYDRA_TOOLS_DIR`, `HEC_HMS_BIN`, `HEC_RAS_BIN` | where the engine programs are found (default `./tools`) |
| `HYDRA_RUN_PYMC_POINT`, `HYDRA_RUN_PYMC_RFA`, `HYDRA_RUN_PYMC_HIER_LOCAL`, `HYDRA_RUN_PYMC_HIER_GENERAL` | the PyMC fits of `valencia_dana/03` (needs PyMC; `HYDRA_PYMC_CHAINS`, `_DRAWS`, `_WARMUP`, `_SEED` set the sampler) |
| `HYDRA_RUN_GIS_PLOTS`, `HYDRA_RUN_MAP_FIGURES`, `HYDRA_RUN_RASTER_INTERPOLATION` | map and raster steps that need extra rasters |

The HEC-HMS and HEC-RAS programs are not distributed with pyhydra or the data;
install them yourself and point the variables above at them. The model
projects they run are in the `los_corrales_buelna` and `m30_manzanares` datasets.

## 4. Where outputs go

Notebooks write generated files to a session folder (`HYDRA_SESSION_DIR`, or
`.hydra_runtime` in the repository) when `HYDRA_DATA_DIR` is read-only, and read
them back from there. The shared data tree is never modified.
