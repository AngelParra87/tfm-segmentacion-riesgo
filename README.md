# Segmentación de clientes con créditos activos según perfil de riesgo

Repositorio del Trabajo Fin de Máster del **Máster Universitario en Análisis y Visualización de Datos Masivos / Visual Analytics & Big Data de UNIR**.

El proyecto analiza la cartera de créditos activos de una entidad financiera peruana con corte a noviembre de 2024. A partir de variables de comportamiento de pago, saldos y calificaciones crediticias, se construyó una segmentación descriptiva de riesgo con MiniBatchKMeans y se preparó una capa de datos para su análisis en Power BI.

## Alcance del proyecto

La unidad de análisis es el **crédito activo**. El conjunto principal contiene **597.127 operaciones correspondientes a 593.638 clientes únicos**. La información se complementa con **1.191.640 registros** del Reporte Crediticio Consolidado (RCC).

El modelo final utiliza siete variables:

- `Dias_Mora`
- `Cuotas_Vencidas`
- `Abono_Promedio`
- `Saldo_Desembolsado`
- `Saldo_Vigente`
- `Calificacion_Sbs`
- `Peor_Calificacion_Rcc`

`Peor_Calificacion_Rcc` se construye a partir del RCC y se integra mediante `Codigo_Cliente_Sbs`. `Estado_Credito` se conserva para la validación externa y no participa en la formación de los clústeres.

La cobertura de la integración con el RCC es de **98,26 %** sobre los códigos SBS válidos y únicos de la cartera (583.252 de 593.587) y de **98,25 %** a nivel de clientes únicos (583.251 de 593.638).

## Modelo de segmentación

El modelo seleccionado utiliza **MiniBatchKMeans con `k=4`**. Los cuatro segmentos resultantes son:

| Segmento | Créditos activos | Participación |
| --- | ---: | ---: |
| Cartera Vigente | 565.639 | 94,73 % |
| Cartera en Riesgo | 11.865 | 1,99 % |
| Cartera Judicial | 9.826 | 1,65 % |
| Cartera Castigada | 9.797 | 1,64 % |
| **Total** | **597.127** | **100,00 %** |

Para la evaluación de `k=2` a `k=10`, los modelos se ajustan sobre el conjunto completo. Silhouette, Davies-Bouldin y Calinski-Harabasz se calculan sobre una muestra fija de 30.000 operaciones. Para `k=4`, las métricas obtenidas fueron:

- Silhouette: **0,9546**
- Davies-Bouldin: **0,4806**
- Calinski-Harabasz: **312.656** aproximadamente

La elección de cuatro clústeres no responde únicamente a la mejor métrica aislada, sino al equilibrio entre separación estadística, granularidad e interpretabilidad de los perfiles.

## Comparación complementaria con DBSCAN

Como contraste metodológico, el TFM evaluó DBSCAN sobre una **muestra aleatoria de 50.000 registros**, con semilla 42. Para mantener condiciones comparables dentro de ese experimento, DBSCAN y MiniBatchKMeans se ejecutaron sobre la misma muestra y con un tratamiento común basado en winsorización al percentil 99 y `StandardScaler`.

La configuración DBSCAN seleccionada dentro del experimento fue `eps = 1.0` y `min_samples = 50`. Generó dos clústeres y aproximadamente 1,1 % de ruido. En la comparación controlada:

| Métrica | MiniBatchKMeans (`k=4`) | DBSCAN |
| --- | ---: | ---: |
| Silhouette | 0,479 | 0,735 |
| Davies-Bouldin | 0,823 | 0,423 |
| Calinski-Harabasz | 30.642 | 12.213 |

El script `02_preprocesamiento_clustering.py` incluye la generación de las figuras finales de esta comparación a partir de los resultados validados documentados en la memoria. La búsqueda completa de DBSCAN no se vuelve a ejecutar dentro del repositorio final.

## Arquitectura

```text
Python
  ↓
Google Cloud Storage
  ↓
BigQuery
  ↓
Power BI
```

La tabla final cargada en BigQuery contiene **597.127 registros y 22 columnas**. El dashboard está organizado en cuatro páginas: Vista General, Perfil de Riesgo, Gestión de Cobranzas y Cartera Vigente.

La descripción del flujo y de cada componente se encuentra en [`docs/arquitectura_gcp.md`](docs/arquitectura_gcp.md).

## Estructura del repositorio

```text
tfm-segmentacion-riesgo/
├── docs/
│   └── arquitectura_gcp.md
├── scripts/
│   ├── 01_eda_cartera.py
│   ├── 02_preprocesamiento_clustering.py
│   └── 03_carga_gcp.py
├── .gitignore
├── requirements.txt
└── README.md
```

Los scripts se ejecutan en este orden:

```bash
python scripts/01_eda_cartera.py
python scripts/02_preprocesamiento_clustering.py
python scripts/03_carga_gcp.py
```

Las rutas de entrada, salida y credenciales deben ajustarse al entorno local antes de ejecutar los scripts.

## Función de cada script

`01_eda_cartera.py` realiza los controles de calidad y el análisis exploratorio de las dos fuentes. Incluye las distribuciones de calificación SBS y mora, el diagnóstico de variables financieras, la matriz de correlaciones, el análisis del RCC y la caracterización complementaria por tipo de préstamo, unidad ejecutora, sexo y año de apertura.

`02_preprocesamiento_clustering.py` integra el RCC mediante `Codigo_Cliente_Sbs`, construye `Peor_Calificacion_Rcc`, codifica las variables ordinales, aplica winsorización P1-P99 a las cinco variables continuas, utiliza `RobustScaler`, evalúa MiniBatchKMeans para `k=2` a `k=10`, ajusta el modelo definitivo con `k=4`, realiza la validación externa con `Estado_Credito` y genera las figuras finales del modelado y de la comparación documentada con DBSCAN.

`03_carga_gcp.py` valida la alineación entre la cartera y el resultado del clustering, prepara la tabla desnormalizada de 22 columnas, carga el archivo en Cloud Storage y reemplaza la tabla `clientes_segmentados` en BigQuery después de aplicar los controles de conteo y distribución de clústeres.

## Datos y confidencialidad

Los archivos de cartera y RCC fueron proporcionados por una entidad financiera para el desarrollo académico del proyecto y **no se publican en este repositorio**. Los campos de identificación personal y de cuenta no se utilizan como variables del modelo ni forman parte de los productos publicados.

El repositorio excluye archivos de datos, credenciales y otros elementos sensibles mediante `.gitignore`. Las credenciales de GCP deben mantenerse fuera del control de versiones.

## Tecnologías principales

- Python
- pandas / NumPy
- scikit-learn
- Matplotlib / Seaborn
- Google Cloud Storage
- Google BigQuery
- Power BI

## Autores

- Lourdes Marylyn Flores Mamani
- Angel Raul Parra Florecin

**Director:** Javier Escobar Ortiz
