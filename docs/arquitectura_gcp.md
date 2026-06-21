# Arquitectura Cloud del Proyecto

## Componentes

- **Python (local):** scripts de ETL y clustering ejecutados en Anaconda (entorno base). Procesan los datasets originales, ejecutan MiniBatchKMeans y generan el CSV con la asignación de clústeres.

- **Cloud Storage:** bucket `tfm-segmentacion-datos` en el proyecto `tfm-segmentacion-riesgo`. Recibe el CSV segmentado como capa de staging.

- **BigQuery:** dataset `cartera_riesgo`, tabla `clientes_segmentados` (604,129 registros, 22 columnas). Funciona como data warehouse para consumo desde Power BI.

- **Power BI Service:** conectado a BigQuery con conector nativo. Dashboard de 4 páginas (Vista General, Perfil de Riesgo, Gestión de Cobranzas, Cartera Vigente) con modelo dimensional en estrella.

## Flujo de datos

```
Datasets originales (TXT)
  → [01_eda_cartera.py] Análisis exploratorio
  → [02_preprocesamiento_clustering.py] Limpieza + MiniBatchKMeans (k=4)
  → [03_carga_gcp.py] Carga a Cloud Storage y BigQuery
  → Power BI Service (dashboard)
```

## Credenciales

Cuenta de servicio `tfm-pipeline` con roles BigQuery Job User, BigQuery Data Editor y Storage Object Admin. El JSON de credenciales no está en el repositorio (excluido por `.gitignore`).
