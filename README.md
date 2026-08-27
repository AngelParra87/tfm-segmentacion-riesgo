# Segmentación de clientes con créditos activos según perfil de riesgo

Repositorio del Trabajo Fin de Máster del Máster Universitario en Análisis y Visualización de Datos Masivos / Visual Analytics & Big Data de UNIR.

El proyecto analiza la cartera de créditos activos de una entidad financiera peruana con corte a noviembre de 2024. A partir de variables de comportamiento de pago, saldos y calificaciones crediticias, se construyó una segmentación de riesgo con MiniBatchKMeans y se preparó una capa de datos para su análisis en Power BI.

## Alcance del proyecto

La unidad de análisis es el crédito activo. El conjunto principal contiene 597.127 operaciones correspondientes a 593.638 clientes únicos. La información se complementa con 1.191.640 registros del Reporte Crediticio Consolidado (RCC).

El modelo final utiliza siete variables:

- `Dias_Mora`
- `Cuotas_Vencidas`
- `Abono_Promedio`
- `Saldo_Desembolsado`
- `Saldo_Vigente`
- `Calificacion_Sbs`
- `Peor_Calificacion_Rcc`

`Peor_Calificacion_Rcc` se construye a partir del RCC y se integra mediante `Codigo_Cliente_Sbs`. `Estado_Credito` se conserva para la validación externa y no participa en la formación de los clústeres.

El modelo seleccionado utiliza MiniBatchKMeans con `k=4`. Los cuatro segmentos resultantes son:

| Segmento | Créditos activos |
| --- | ---: |
| Cartera Vigente | 565.639 |
| Cartera en Riesgo | 11.865 |
| Cartera Judicial | 9.826 |
| Cartera Castigada | 9.797 |
| **Total** | **597.127** |

Para la evaluación de `k=2` a `k=10`, los modelos se ajustan sobre el conjunto completo. Silhouette, Davies-Bouldin y Calinski-Harabasz se calculan sobre una muestra fija de 30.000 operaciones. Para `k=4`, las métricas obtenidas fueron:

- Silhouette: **0,9546**
- Davies-Bouldin: **0,4806**
- Calinski-Harabasz: **312.655,61**

La elección de cuatro clústeres no responde únicamente a la mejor métrica aislada, sino al equilibrio entre separación estadística y utilidad operativa.

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

La tabla final cargada en BigQuery contiene 597.127 registros y 22 columnas. El dashboard está organizado en cuatro páginas: Vista General, Perfil de Riesgo, Gestión de Cobranzas y Cartera Vigente.

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

## Datos

Los archivos de cartera y RCC fueron proporcionados para el desarrollo académico del proyecto y no se publican en este repositorio. Los campos de identificación personal y de cuenta no se utilizan como variables del modelo ni forman parte de los productos publicados.

El repositorio excluye archivos de datos, credenciales y otros elementos sensibles mediante `.gitignore`.

## Autores

- Lourdes Marylyn Flores Mamani
- Angel Raul Parra Florecin

**Director:** Javier Escobar Ortiz

