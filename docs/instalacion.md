# Installation

pyhydra is a standalone `pip`-installable package; it does not require
Docker, HYDRA, or any other repository to be installed and used. The
canonical installation instructions live in the
[README](../README.md#installation) so there is a single place to keep
them current — see there for:

- installing the latest `main` or a specific tagged release;
- the `statistics`, `geospatial`, `models`, `geo` and `all` extras, and
  which functions each one unlocks;
- development install (`pip install -e .`);
- the `CoSMoS_py` dependency, which is not on PyPI and must be installed
  separately from [`navass11/CoSMoS_py`](https://github.com/navass11/CoSMoS_py):

  ```bash
  pip install git+https://github.com/navass11/CoSMoS_py.git
  ```

- GDAL/geospatial installation caveats.

## Running notebooks without installing anything locally

If you would rather not manage a local Python environment at all, the
companion [`HYDRA`](https://github.com/navass11/HYDRA) platform provides a
Docker Compose environment with pyhydra and every extended dependency
pre-installed, exposed through JupyterLab and a browser UI. See HYDRA's
own README for that setup; it is optional and does not change how pyhydra
itself is installed or used directly.
