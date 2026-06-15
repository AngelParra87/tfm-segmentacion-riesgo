\# Arquitectura Cloud del Proyecto



\## Componentes



1\. Python (local): Scripts de ETL y clustering ejecutados en Anaconda

&#x20;  (entorno base). Procesan los datasets originales, ejecutan MiniBatchKMeans

&#x20;  y generan el archivo de resultados con la asignación de clusters.



2\. GCP Cloud Storage: Bucket tfm-segmentacion-datos en el proyecto

&#x20;  tfm-segmentacion-riesgo. Recibe el archivo CSV con los datos segmentados

&#x20;  como capa de staging.



3\. Google BigQuery: Dataset cartera\_riesgo, tabla clientes\_segmentados

&#x20;  (604,129 registros, 22 columnas). Actua como data warehouse para el

&#x20;  consumo analitico.



4\. Power BI Service: Conectado a BigQuery mediante conector nativo.

&#x20;  Dashboard de 4 paginas (Vista General, Perfil de Riesgo, Gestion de

&#x20;  Cobranzas, Cartera Vigente) con modelo dimensional en estrella.



\## Flujo de datos



Datasets originales (TXT)

&#x20; > \[01\_eda\_cartera.py] Analisis exploratorio

&#x20; > \[02\_preprocesamiento\_clustering.py] Limpieza + MiniBatchKMeans (k=4)

&#x20; > \[03\_carga\_gcp.py] Carga a Cloud Storage y BigQuery

&#x20; > \[conector DirectQuery] Power BI Service (dashboard)

