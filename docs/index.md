# HYDRA

**HYDRA** es una librería modular de Python para el análisis hidrológico y climático. Integra herramientas de descarga y procesamiento de datos, análisis estadístico del clima, y soporte para modelos hidrológicos e hidráulicos.

## Estructura de la librería

```
HYDRA/
├── Data_Sources/              # Descarga y preprocesado de datos
│   ├── Climate_Change/        # Proyecciones de cambio climático
│   │   ├── COPERNICUS/        # Descarga CMIP6 via CDS API (cdsapi + Selenium)
│   │   ├── ESGF/              # Descarga CMIP6 via ESGF (OPeNDAP / HTTP)
│   │   └── utils.py           # Utilidades compartidas + clase bias_correction
│   ├── Rainfall/              # Datos de precipitación
│   │   ├── GPM/               # IMERG NASA via earthaccess (clase GPMDownloader)
│   │   ├── PERSSIAN/          # PERSIANN-CCS via FTP (clase PERSSIANDownloader)
│   │   ├── ERA-5/             # ERA5 via CDS API (función download_era5)
│   │   ├── AEMET/             # Red AEMET Spain via API OpenData + widget Jupyter
│   │   └── OGIMET/            # Estaciones SYNOP globales via scraping + widget Jupyter
│   ├── River_Discharge/       # Caudal fluvial (GloFAS, RivDIS, NCAR)
│   └── Soils/                 # Texturas de suelo SoilGrids → clase USDA
│
├── Climate/                   # Análisis estadístico del clima
│   ├── Time_Series_Analysis/
│   │   ├── Extremes/          # Análisis de valores extremos
│   │   └── Discretization/    # Separación de eventos + generación sintética
│   ├── Spatial_Analysis/
│   │   ├── RFA/               # Análisis de frecuencia regional
│   │   ├── Bayes_Hierarchical/# Modelos jerárquicos bayesianos
│   │   ├── Interpolation/     # Interpolación + Gaussian Processes
│   │   └── Copulas/           # Dependencia multivariante
│   ├── Stochastic_Generation/ # Generación estocástica (NEOPRENE, CoSMoS_py)
│   ├── Bias_Correction/
│   │   ├── Delta/             # Método Delta
│   │   └── QQ_Mapping/        # QM, QDM y SDM
│   └── Hybrid_Downscaling/    # Downscaling estadístico
│
└── Modeling/                  # Modelos hidrológicos e hidráulicos
    ├── Hydrology/
    │   ├── SWAT/              # Automatización SWAT+
    │   └── HEC-HMS/           # Automatización HEC-HMS
    └── Hydraulic/
        ├── SFINCS/            # Automatización SFINCS
        └── HEC-RAS/           # Automatización HEC-RAS
```

## Contribución

Cada módulo tiene su propio `README.md` describiendo el código existente. Consulta el repositorio en [GitHub](https://github.com/navass11/HYDRA) para la visión global de la arquitectura.
