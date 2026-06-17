# Instalación

## Con Docker (recomendado)

La forma más sencilla de ejecutar los notebooks sin instalar dependencias manualmente.

**Requisitos:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.

```bash
# 1. Clonar el repositorio
git clone https://github.com/navass11/HYDRA.git
cd HYDRA

# 2. Construir la imagen y lanzar JupyterLab
docker compose -f docker/docker-compose.yml up --build

# 3. Abrir en el navegador
#    http://localhost:8888
```

Los cambios realizados en los notebooks se guardan directamente en el repositorio local.

Para parar el contenedor:

```bash
docker compose -f docker/docker-compose.yml down
```

## Instalación manual

### Requisitos

- Python 3.9 o superior
- pip

### Dependencias principales

```bash
pip install numpy pandas xarray dask scipy statsmodels scikit-learn \
            matplotlib tqdm openturns lmoments3 requests NEOPRENE
```

### CoSMoS_py

```bash
pip install git+https://github.com/navass11/CoSMoS_py.git
```

### pyhydra

```bash
git clone https://github.com/navass11/HYDRA.git
cd HYDRA
pip install -e .
```

### JupyterLab

```bash
pip install jupyterlab
jupyter lab --notebook-dir=notebooks/
```
