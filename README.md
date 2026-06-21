# Segmentación de Clientes según Perfil de Riesgo

Trabajo Fin de Máster — Máster Universitario en Análisis y Visualización de Datos Masivos / Visual Analytics & Big Data, UNIR.

## Qué hace

Segmenta clientes con créditos activos de una entidad financiera peruana en cuatro perfiles de riesgo usando MiniBatchKMeans. Los resultados se cargan en BigQuery y se visualizan en un dashboard de Power BI Service.

## Arquitectura

```
Python (ETL + clustering) → GCP Cloud Storage → BigQuery → Power BI Service
```

Ver `docs/arquitectura_gcp.md` para más detalle.

## Scripts

| Script | Descripción |
|--------|-------------|
| `scripts/01_eda_cartera.py` | Carga de datos y análisis exploratorio |
| `scripts/02_preprocesamiento_clustering.py` | Preprocesamiento, winsorización, escalado y clustering (MiniBatchKMeans, k=4) |
| `scripts/03_carga_gcp.py` | Carga de resultados a Cloud Storage y BigQuery |

## Ejecución

Los scripts se ejecutan en orden desde Anaconda Prompt con el entorno base:

```bash
python scripts/01_eda_cartera.py
python scripts/02_preprocesamiento_clustering.py
python scripts/03_carga_gcp.py
```

Dependencias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `google-cloud-storage`, `google-cloud-bigquery`, `pyarrow`.

## Datos

Los datasets provienen de una entidad financiera (cartera propia + RCC-SBS, noviembre 2024). Son datos anonimizados y no están incluidos en el repositorio. El `.gitignore` excluye archivos de datos y credenciales.

## Autores

- Flores Mamani, Lourdes Marylyn
- Parra Florecin, Angel Raul

Director: Javier Escobar Ortiz
