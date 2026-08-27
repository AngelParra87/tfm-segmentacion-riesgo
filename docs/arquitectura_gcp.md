# Arquitectura del proyecto

El proyecto separa el procesamiento analítico de la capa de visualización. Python se utiliza para preparar los datos y ejecutar el clustering; Google Cloud Storage y BigQuery conforman la capa de almacenamiento en GCP; Power BI consume la tabla final para el análisis interactivo.

## Flujo de trabajo

```text
cartera_creditos + reporte_crediticio_rcc
        │
        ▼
01_eda_cartera.py
Análisis exploratorio y controles de calidad
        │
        ▼
02_preprocesamiento_clustering.py
Integración RCC, preprocesamiento y MiniBatchKMeans
        │
        ▼
resultado_clustering.csv
        │
        ▼
03_carga_gcp.py
Construcción de la tabla de consumo
        │
        ▼
Google Cloud Storage
        │
        ▼
BigQuery
        │
        ▼
Power BI
```

## 1. Procesamiento en Python

Los scripts se ejecutan de forma local.

`01_eda_cartera.py` realiza la carga inicial, los controles de calidad y el análisis exploratorio de la cartera y del RCC.

`02_preprocesamiento_clustering.py` integra ambas fuentes mediante `Codigo_Cliente_Sbs`, prepara las siete variables del modelo, aplica winsorización P1-P99 a las variables continuas, utiliza `RobustScaler` y evalúa MiniBatchKMeans para valores de `k` entre 2 y 10.

La cartera contiene 597.127 créditos activos. El cruce con el RCC conserva ese mismo número de operaciones. Se identificaron 585.924 operaciones con correspondencia RCC, equivalentes al 98,12 % de la cartera a nivel de operación.

El modelo final utiliza `k=4` y asigna la totalidad de los créditos a uno de los cuatro segmentos de riesgo.

## 2. Google Cloud Storage

`03_carga_gcp.py` genera el archivo `clientes_segmentados_bq.csv` y lo carga en Cloud Storage como etapa previa a BigQuery.

El archivo contiene la asignación de clúster y las variables necesarias para las visualizaciones. `Codigo_Cliente_Sbs` se utiliza durante la integración y la trazabilidad técnica, pero no se incorpora a la tabla final de consumo.

## 3. BigQuery

La tabla final es:

```text
tfm-segmentacion-riesgo.cartera_riesgo.clientes_segmentados
```

Después de la carga se validan nuevamente:

- 597.127 registros.
- 22 columnas.
- La distribución exacta de los cuatro clústeres.

La carga se realiza con reemplazo de la tabla para evitar acumular versiones previas del mismo corte.

## 4. Power BI

Power BI consume la tabla de BigQuery y presenta cuatro vistas:

- Vista General.
- Perfil de Riesgo.
- Gestión de Cobranzas.
- Cartera Vigente.

Los conteos mostrados en el dashboard corresponden a créditos activos, salvo que una visualización indique expresamente que utiliza clientes únicos.

## Credenciales y datos

Las credenciales de GCP se mantienen fuera del repositorio. Los archivos de datos utilizados en el TFM tampoco se publican.

Antes de ejecutar `03_carga_gcp.py`, debe configurarse localmente la ruta del archivo de credenciales y verificarse que la cuenta utilizada disponga de permisos suficientes para trabajar con Cloud Storage y BigQuery.
