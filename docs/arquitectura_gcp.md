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
Integración RCC, preprocesamiento, MiniBatchKMeans y validación
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

Los scripts se ejecutan de forma local y corresponden a las tres etapas principales del pipeline.

### `01_eda_cartera.py`

Realiza la carga inicial, los controles de calidad y el análisis exploratorio de `cartera_creditos` y `reporte_crediticio_rcc`. El script reúne el EDA principal y la caracterización ampliada que durante el desarrollo se trabajaron en archivos separados.

Entre sus salidas se encuentran:

- distribución de créditos activos por calificación SBS;
- distribución de mora;
- diagnóstico de variables financieras;
- matriz de correlaciones;
- composición del RCC por tipo de crédito;
- distribución por tipo de préstamo;
- Top 20 de unidades ejecutoras;
- distribución por sexo;
- créditos activos por año de apertura.

### `02_preprocesamiento_clustering.py`

Integra ambas fuentes mediante `Codigo_Cliente_Sbs` y construye `Peor_Calificacion_Rcc`. La cartera contiene 597.127 créditos activos y se conserva ese mismo número de operaciones después de la integración.

La cobertura medida antes de completar los valores no encontrados en el RCC es:

- **593.587** códigos SBS válidos y únicos en la cartera;
- **583.252** códigos SBS con correspondencia en el RCC;
- **98,26 %** de cobertura por código SBS;
- **593.638** clientes únicos en la cartera;
- **583.251** clientes únicos con correspondencia en el RCC;
- **98,25 %** de cobertura por cliente.

El conjunto final de modelado utiliza siete variables. Las cinco continuas (`Dias_Mora`, `Cuotas_Vencidas`, `Abono_Promedio`, `Saldo_Desembolsado` y `Saldo_Vigente`) se winsorizan entre P1 y P99. `Calificacion_Sbs` y `Peor_Calificacion_Rcc` mantienen sus escalas ordinales. Posteriormente se aplica `RobustScaler`.

MiniBatchKMeans se evalúa para valores de `k` entre 2 y 10 con `batch_size=10000`, `n_init=15`, `max_iter=300` y `random_state=42`. Los modelos se ajustan con la cartera completa y las métricas Silhouette, Davies-Bouldin y Calinski-Harabasz se calculan sobre una muestra fija de 30.000 operaciones.

La configuración definitiva utiliza `k=4`. `Estado_Credito` no participa en la formación de los clústeres y se reserva para la validación externa.

El mismo script regenera además las figuras definitivas de selección de `k`, perfil de segmentos y distribución de créditos. También genera las figuras de la comparación con DBSCAN utilizando los resultados validados del experimento documentado en la memoria. Ese bloque visual no vuelve a ejecutar la búsqueda completa de hiperparámetros DBSCAN.

## 2. Google Cloud Storage

`03_carga_gcp.py` genera el archivo `clientes_segmentados_bq.csv` y lo carga en Cloud Storage como etapa previa a BigQuery.

Antes de la carga se valida:

- que la cartera y `resultado_clustering.csv` contengan 597.127 operaciones;
- la alineación fila a fila mediante `Cliente` y `Codigo_Cliente_Sbs`;
- que la incorporación de `Peor_Calificacion_Rcc` mediante `Codigo_Cliente_Sbs` no aumente el número de filas;
- la distribución exacta de los cuatro clústeres;
- que la tabla final contenga 22 columnas.

`Codigo_Cliente_Sbs` se utiliza durante la integración y la trazabilidad técnica, pero no se incorpora a la tabla final de consumo.

## 3. BigQuery

La tabla final es:

```text
tfm-segmentacion-riesgo.cartera_riesgo.clientes_segmentados
```

La carga se realiza con reemplazo de la tabla (`WRITE_TRUNCATE`) para evitar acumular versiones previas del mismo corte.

Después de la carga se validan nuevamente:

- **597.127 registros**;
- **22 columnas**;
- distribución exacta de los cuatro clústeres.

La distribución esperada es:

| Clúster | Segmento | Créditos activos |
| ---: | --- | ---: |
| 0 | Cartera en Riesgo | 11.865 |
| 1 | Cartera Vigente | 565.639 |
| 2 | Cartera Castigada | 9.797 |
| 3 | Cartera Judicial | 9.826 |

## 4. Power BI

Power BI consume la tabla de BigQuery y presenta cuatro vistas:

- Vista General;
- Perfil de Riesgo;
- Gestión de Cobranzas;
- Cartera Vigente.

Los conteos mostrados en el dashboard corresponden a créditos activos, salvo que una visualización indique expresamente que utiliza clientes únicos.

## Credenciales y datos

Las credenciales de GCP se mantienen fuera del repositorio. Los archivos de datos utilizados en el TFM tampoco se publican.

Antes de ejecutar `03_carga_gcp.py`, debe configurarse localmente la ruta del archivo de credenciales y verificarse que la cuenta utilizada disponga de permisos suficientes para trabajar con Cloud Storage y BigQuery.
