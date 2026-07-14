"""
03_carga_gcp.py
Pipeline de carga: Python → Cloud Storage → BigQuery
TFM: Segmentación de Clientes según Perfil de Riesgo

Requisitos previos:
    pip install google-cloud-storage google-cloud-bigquery pandas pyarrow

Uso:
    python 03_carga_gcp.py
"""

import pandas as pd
import os
from google.cloud import storage, bigquery

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
PROJECT_ID   = "tfm-segmentacion-riesgo"
BUCKET_NAME  = "tfm-segmentacion-datos"
DATASET_ID   = "cartera_riesgo"
TABLE_ID     = "clientes_segmentados"
CREDENTIALS  = r"C:\ANGEL\UNIR\gcp_config\gcp_credentials.json"

DATA_DIR     = r"C:\ANGEL\UNIR\TFM 2026 - v2\Data"
OUTPUT_DIR   = r"C:\ANGEL\UNIR\TFM 2026 - v2\Data\outputs\clustering"

FILE_COD     = os.path.join(DATA_DIR, "cartera_creditos")
FILE_RCC     = os.path.join(DATA_DIR, "reporte_crediticio_rcc")
FILE_CLUSTER = os.path.join(OUTPUT_DIR, "resultado_clustering.csv")

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS)

# ─────────────────────────────────────────────
# 1. CONSTRUCCIÓN DE LA TABLA DESNORMALIZADA
# ─────────────────────────────────────────────
print("[1/5] Cargando datasets...")

cols_cod = [
    "Cliente", "Tipo_Prestamo", "Sexo",
    "Desc_Unidad_Ejecutora", "Desc_Lugar_Emision",
    "Saldo_Desembolsado", "Saldo_Vigente", "Numero_Cuotas",
    "Cuotas_Pagadas", "Cuotas_Vencidas", "Cuotas_Pendientes",
    "Dias_Mora", "Abono_Promedio", "Monto_Cuota", "Tasa_Interes",
    "Estado_Credito", "Fecha_Apertura",
    "Calificacion_Sbs"
]

df_cod = pd.read_csv(
    FILE_COD, sep=";", encoding="latin-1",
    usecols=cols_cod, low_memory=False
)
print(f"  COD: {len(df_cod):,} registros, {len(df_cod.columns)} columnas")

df_rcc = pd.read_csv(
    FILE_RCC, sep=";", encoding="latin-1", low_memory=False
)
print(f"  RCC: {len(df_rcc):,} registros")

df_cluster = pd.read_csv(FILE_CLUSTER)
print(f"  Clustering: {len(df_cluster):,} registros asignados")

# ─────────────────────────────────────────────
# 2. PREPARACIÓN DE VARIABLES DERIVADAS
# ─────────────────────────────────────────────
print("\n[2/5] Preparando variables derivadas...")

# --- Peor_Calificacion_Rcc: peor calificación externa por cliente ---
df_rcc_max = (
    df_rcc
    .groupby("Codigo_Cliente_Sbs")["Calificacion_Entidad"]
    .max()
    .reset_index()
    .rename(columns={
        "Codigo_Cliente_Sbs": "cod_sbs",
        "Calificacion_Entidad": "Peor_Calificacion_Rcc"
    })
)

# Obtener Codigo_Cliente_Sbs desde el dataset principal para el merge
df_cod_sbs = pd.read_csv(
    FILE_COD, sep=";", encoding="latin-1",
    usecols=["Cliente", "Codigo_Cliente_Sbs"], low_memory=False
)
df_cod_sbs = df_cod_sbs.drop_duplicates(subset=["Cliente"])
df_cod_sbs["Codigo_Cliente_Sbs"] = df_cod_sbs["Codigo_Cliente_Sbs"].astype(str).str.strip()

# Merge para obtener Peor_Calificacion_Rcc por Cliente
df_rcc_max["cod_sbs"] = df_rcc_max["cod_sbs"].astype(str).str.strip()
df_cod_sbs = df_cod_sbs.merge(
    df_rcc_max,
    left_on="Codigo_Cliente_Sbs",
    right_on="cod_sbs",
    how="left"
)
df_clasif = df_cod_sbs[["Cliente", "Peor_Calificacion_Rcc"]].copy()
df_clasif["Peor_Calificacion_Rcc"] = df_clasif["Peor_Calificacion_Rcc"].fillna(0).astype(int)

print(f"  Peor_Calificacion_Rcc calculada para {len(df_clasif):,} clientes")

# --- Nombre descriptivo del clúster ---
map_cluster = {
    0: "Cartera en Riesgo",
    1: "Cartera Vigente",
    2: "Cartera Castigada",
    3: "Cartera Judicial"
}

df_cluster["cluster_nombre"] = df_cluster["cluster"].map(map_cluster)

# --- Calificación SBS: mantener texto original limpio ---
map_calf_texto = {
    "2. NORMAL"      : "Normal",
    "3. CPP"         : "CPP",
    "4. DEFICIENTE"  : "Deficiente",
    "5. DUDOSO"      : "Dudoso",
    "6. PERDIDA"     : "Pérdida",
    "1. NO DEFINIDO" : "Normal",
}

df_cod["Calificacion_Sbs_Texto"] = (
    df_cod["Calificacion_Sbs"]
    .astype(str).str.strip()
    .map(map_calf_texto)
    .fillna("Normal")
)

map_calf_num = {
    "Normal": 1, "CPP": 2, "Deficiente": 3,
    "Dudoso": 4, "Pérdida": 5
}
df_cod["Calificacion_Sbs_Cod"] = df_cod["Calificacion_Sbs_Texto"].map(map_calf_num)

# --- Limpiar campos de texto ---
df_cod["Tipo_Prestamo"]          = df_cod["Tipo_Prestamo"].astype(str).str.strip()
df_cod["Desc_Unidad_Ejecutora"]  = df_cod["Desc_Unidad_Ejecutora"].astype(str).str.strip()
df_cod["Desc_Lugar_Emision"]     = df_cod["Desc_Lugar_Emision"].astype(str).str.strip()
df_cod["Estado_Credito"]         = df_cod["Estado_Credito"].astype(str).str.strip()
df_cod["Sexo"]                   = df_cod["Sexo"].astype(str).str.strip()

# ─────────────────────────────────────────────
# 3. MERGE FINAL → TABLA DESNORMALIZADA
# ─────────────────────────────────────────────
print("\n[3/5] Construyendo tabla desnormalizada...")

df_final = (
    df_cod
    .merge(df_cluster[["Cliente", "cluster", "cluster_nombre"]], on="Cliente", how="inner")
    .merge(df_clasif[["Cliente", "Peor_Calificacion_Rcc"]], on="Cliente", how="left")
)

cols_bq = [
    "Cliente",
    "cluster", "cluster_nombre",
    # Variables de clustering
    "Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio",
    "Saldo_Desembolsado", "Saldo_Vigente",
    "Calificacion_Sbs_Texto", "Calificacion_Sbs_Cod", "Peor_Calificacion_Rcc",
    # Variable de validación
    "Estado_Credito",
    # Contexto del crédito
    "Tipo_Prestamo", "Numero_Cuotas", "Cuotas_Pagadas",
    "Cuotas_Pendientes", "Monto_Cuota", "Tasa_Interes", "Fecha_Apertura",
    # Demografía y ubicación
    "Sexo", "Desc_Unidad_Ejecutora", "Desc_Lugar_Emision"
]

df_final = df_final[cols_bq].copy()
df_final["Peor_Calificacion_Rcc"] = df_final["Peor_Calificacion_Rcc"].fillna(0).astype(int)

print(f"  Tabla final: {len(df_final):,} registros, {len(df_final.columns)} columnas")

csv_local = os.path.join(OUTPUT_DIR, "clientes_segmentados_bq.csv")
df_final.to_csv(csv_local, index=False, encoding="utf-8")
print(f"  CSV guardado: {csv_local}")

# ─────────────────────────────────────────────
# 4. SUBIDA A CLOUD STORAGE
# ─────────────────────────────────────────────
print("\n[4/5] Subiendo a Cloud Storage...")

client_gcs = storage.Client(project=PROJECT_ID, credentials=credentials)
bucket = client_gcs.bucket(BUCKET_NAME)
blob = bucket.blob("datos/clientes_segmentados_bq.csv")
blob.upload_from_filename(csv_local)

print(f"  Subido: gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv")

# ─────────────────────────────────────────────
# 5. CARGA EN BIGQUERY
# ─────────────────────────────────────────────
print("\n[5/5] Cargando en BigQuery...")

client_bq = bigquery.Client(project=PROJECT_ID, credentials=credentials)
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=False,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    schema=[
        bigquery.SchemaField("Cliente",                  "INTEGER"),
        bigquery.SchemaField("cluster",                  "INTEGER"),
        bigquery.SchemaField("cluster_nombre",           "STRING"),
        bigquery.SchemaField("Dias_Mora",                "FLOAT"),
        bigquery.SchemaField("Cuotas_Vencidas",          "INTEGER"),
        bigquery.SchemaField("Abono_Promedio",           "FLOAT"),
        bigquery.SchemaField("Saldo_Desembolsado",       "FLOAT"),
        bigquery.SchemaField("Saldo_Vigente",            "FLOAT"),
        bigquery.SchemaField("Calificacion_Sbs_Texto",   "STRING"),
        bigquery.SchemaField("Calificacion_Sbs_Cod",     "INTEGER"),
        bigquery.SchemaField("Peor_Calificacion_Rcc",    "INTEGER"),
        bigquery.SchemaField("Estado_Credito",           "STRING"),
        bigquery.SchemaField("Tipo_Prestamo",            "STRING"),
        bigquery.SchemaField("Numero_Cuotas",            "FLOAT"),
        bigquery.SchemaField("Cuotas_Pagadas",           "FLOAT"),
        bigquery.SchemaField("Cuotas_Pendientes",        "FLOAT"),
        bigquery.SchemaField("Monto_Cuota",              "FLOAT"),
        bigquery.SchemaField("Tasa_Interes",             "FLOAT"),
        bigquery.SchemaField("Fecha_Apertura",           "STRING"),
        bigquery.SchemaField("Sexo",                     "STRING"),
        bigquery.SchemaField("Desc_Unidad_Ejecutora",    "STRING"),
        bigquery.SchemaField("Desc_Lugar_Emision",       "STRING"),
    ]
)

uri = f"gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv"
load_job = client_bq.load_table_from_uri(uri, table_ref, job_config=job_config)
load_job.result()

table = client_bq.get_table(table_ref)
print(f"  Tabla cargada: {table_ref}")
print(f"  Registros en BigQuery: {table.num_rows:,}")

# ─────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("PIPELINE COMPLETADO")
print("=" * 55)
print(f"  CSV local:      {csv_local}")
print(f"  Cloud Storage:  gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv")
print(f"  BigQuery:       {table_ref}")
print(f"  Registros:      {table.num_rows:,}")
print(f"  Columnas:       {len(table.schema)}")
print("=" * 55)
print("\nSiguiente paso: conectar Power BI Service a BigQuery")
print(f"  Proyecto: {PROJECT_ID}")
print(f"  Dataset:  {DATASET_ID}")
print(f"  Tabla:    {TABLE_ID}")
