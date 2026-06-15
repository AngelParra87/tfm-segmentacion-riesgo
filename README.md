# \# Segmentación de Clientes según Perfil de Riesgo

# 

# Trabajo Fin de Máster — Máster Universitario en Análisis y Visualización

# de Datos Masivos / Visual Analytics \& Big Data, UNIR.

# 

# \## Descripción

# 

# Sistema de segmentación de clientes con créditos activos en una entidad

# financiera peruana, utilizando aprendizaje no supervisado (MiniBatchKMeans)

# para clasificar clientes según su perfil de riesgo crediticio. Los resultados

# se despliegan en un dashboard interactivo construido en Power BI Service.

# 

# \## Arquitectura

# 

# Python (ETL + clustering) → GCP Cloud Storage → BigQuery → Power BI Service

# 

# \## Estructura del repositorio

# 

# \- `scripts/01\_eda\_cartera.py` — Análisis exploratorio de datos

# \- `scripts/02\_preprocesamiento\_clustering.py` — Preprocesamiento,

# &#x20; transformación y ejecución del modelo de clustering (MiniBatchKMeans, k=4)

# \- `scripts/03\_carga\_gcp.py` — Carga de resultados a GCP Cloud Storage

# &#x20; y BigQuery

# \- `docs/arquitectura\_gcp.md` — Descripción de la arquitectura cloud

# 

# \## Datos

# 

# Los datos utilizados corresponden a registros anonimizados de una entidad

# financiera y no se incluyen en el repositorio por razones de confidencialidad,

# en cumplimiento de la Ley N.° 29733 de Protección de Datos Personales del Perú.

# 

# \## Tecnologías

# 

# \- Python 3.x (pandas, scikit-learn, matplotlib, seaborn)

# \- Google Cloud Platform (Cloud Storage, BigQuery)

# \- Power BI Service

# \- MiniBatchKMeans (scikit-learn)

# 

# \## Autores

# 

# \- Flores Mamani, Lourdes Marylyn

# \- Parra Florecin, Angel Raul

# 

# Director: Javier Escobar Ortiz

