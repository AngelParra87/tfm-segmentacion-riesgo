"""
03_carga_gcp.py - Carga a Cloud Storage y BigQuery
TFM: Segmentación de Clientes según Perfil de Riesgo
Autores: Lourdes Flores Mamani / Angel Parra Florecin
"""

import pandas as pd
import os
from google.cloud import storage, bigquery
from google.oauth2 import service_account

# --- Configuración GCP ---
PROJECT_ID   = "tfm-segmentacion-riesgo"
BUCKET_NAME  = "tfm-segmentacion-datos"
DATASET_ID   = "cartera_riesgo"
TABLE_ID     = "clientes_segmentados"
CREDENTIALS  = r"C:\ANGEL\UNIR\gcp_config\gcp_credentials.json"

DATA_DIR     = r"C:\ANGEL\UNIR\TFM 2026\Data"
OUTPUT_DIR   = r"C:\ANGEL\UNIR\TFM 2026\Data\outputs\clustering"

FILE_COD     = os.path.join(DATA_DIR, "grf10_1124_cod.txt")
FILE_RCC     = os.path.join(DATA_DIR, "grf10_1124_rcc.txt")
FILE_CLUSTER = os.path.join(OUTPUT_DIR, "resultado_clustering.csv")

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS)

# =============================================
# 1. CARGA DE DATASETS
# =============================================
print("Cargando datasets...")

cols_cod = [
    "CLIENTE", "ATPRESTAMO", "SEXO",
    "AUNID_EJECT", "ALUGA_EMISI",
    "SDSBOLSO", "SACTUAL", "NRO_CUOTAS",
    "NCUOTAS_PAG", "NCUOTAS_VEN", "CUO_PEN_PAGO",
    "DS_MORA", "SABONO_PROM", "SCUOTA", "STASA",
    "BDSBOLSO", "FAPERTURA",
    "calf_sbs_nov24"
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
print(f"  Clustering: {len(df_cluster):,} registros")

# =============================================
# 2. VARIABLES DERIVADAS
# =============================================
print("\nPreparando variables derivadas...")

# clasif_rcc_max: peor calificación externa por cliente
df_rcc_max = (
    df_rcc
    .groupby("cod_cliente_sbs")["CLASIF_EMP"]
    .max()
    .reset_index()
    .rename(columns={
        "cod_cliente_sbs": "cod_sbs",
        "CLASIF_EMP": "clasif_rcc_max"
    })
)

df_cod_sbs = pd.read_csv(
    FILE_COD, sep=";", encoding="latin-1",
    usecols=["CLIENTE", "cod_sbs_nov24"], low_memory=False
)
df_cod_sbs = df_cod_sbs.drop_duplicates(subset=["CLIENTE"])
df_cod_sbs["cod_sbs_nov24"] = df_cod_sbs["cod_sbs_nov24"].astype(str).str.strip()

df_rcc_max["cod_sbs"] = df_rcc_max["cod_sbs"].astype(str).str.strip()
df_cod_sbs = df_cod_sbs.merge(
    df_rcc_max,
    left_on="cod_sbs_nov24",
    right_on="cod_sbs",
    how="left"
)
df_clasif = df_cod_sbs[["CLIENTE", "clasif_rcc_max"]].copy()
df_clasif["clasif_rcc_max"] = df_clasif["clasif_rcc_max"].fillna(0).astype(int)

print(f"  clasif_rcc_max: {len(df_clasif):,} clientes")

# Nombre del clúster
map_cluster = {
    0: "Cartera en Riesgo",
    1: "Cartera Vigente",
    2: "Cartera Castigada",
    3: "Cartera Judicial"
}
df_cluster["cluster_nombre"] = df_cluster["cluster"].map(map_cluster)

# Calificación SBS: texto limpio + codificación ordinal
map_calf_texto = {
    "2. NORMAL"      : "Normal",
    "3. CPP"         : "CPP",
    "4. DEFICIENTE"  : "Deficiente",
    "5. DUDOSO"      : "Dudoso",
    "6. PERDIDA"     : "Pérdida",
    "1. NO DEFINIDO" : "Normal",
}

df_cod["calf_sbs_texto"] = (
    df_cod["calf_sbs_nov24"]
    .astype(str).str.strip()
    .map(map_calf_texto)
    .fillna("Normal")
)

map_calf_num = {
    "Normal": 1, "CPP": 2, "Deficiente": 3,
    "Dudoso": 4, "Pérdida": 5
}
df_cod["calf_sbs_cod"] = df_cod["calf_sbs_texto"].map(map_calf_num)

# Limpiar campos de texto
for col in ["ATPRESTAMO", "AUNID_EJECT", "ALUGA_EMISI", "BDSBOLSO", "SEXO"]:
    df_cod[col] = df_cod[col].astype(str).str.strip()

# =============================================
# 3. TABLA DESNORMALIZADA
# =============================================
print("\nConstruyendo tabla final...")

df_final = (
    df_cod
    .merge(df_cluster[["CLIENTE", "cluster", "cluster_nombre"]], on="CLIENTE", how="inner")
    .merge(df_clasif[["CLIENTE", "clasif_rcc_max"]], on="CLIENTE", how="left")
)

cols_bq = [
    "CLIENTE",
    "cluster", "cluster_nombre",
    "DS_MORA", "NCUOTAS_VEN", "SABONO_PROM",
    "SDSBOLSO", "SACTUAL",
    "calf_sbs_texto", "calf_sbs_cod", "clasif_rcc_max",
    "BDSBOLSO",
    "ATPRESTAMO", "NRO_CUOTAS", "NCUOTAS_PAG",
    "CUO_PEN_PAGO", "SCUOTA", "STASA", "FAPERTURA",
    "SEXO", "AUNID_EJECT", "ALUGA_EMISI"
]

df_final = df_final[cols_bq].copy()
df_final["clasif_rcc_max"] = df_final["clasif_rcc_max"].fillna(0).astype(int)

print(f"  {len(df_final):,} registros, {len(df_final.columns)} columnas")

csv_local = os.path.join(OUTPUT_DIR, "clientes_segmentados_bq.csv")
df_final.to_csv(csv_local, index=False, encoding="utf-8")
print(f"  CSV: {csv_local}")

# =============================================
# 4. SUBIDA A CLOUD STORAGE
# =============================================
print("\nSubiendo a Cloud Storage...")

client_gcs = storage.Client(project=PROJECT_ID, credentials=credentials)
bucket = client_gcs.bucket(BUCKET_NAME)
blob = bucket.blob("datos/clientes_segmentados_bq.csv")
blob.upload_from_filename(csv_local)

print(f"  gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv")

# =============================================
# 5. CARGA EN BIGQUERY
# =============================================
print("\nCargando en BigQuery...")

client_bq = bigquery.Client(project=PROJECT_ID, credentials=credentials)
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=False,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    schema=[
        bigquery.SchemaField("CLIENTE",         "INTEGER"),
        bigquery.SchemaField("cluster",          "INTEGER"),
        bigquery.SchemaField("cluster_nombre",   "STRING"),
        bigquery.SchemaField("DS_MORA",          "FLOAT"),
        bigquery.SchemaField("NCUOTAS_VEN",      "INTEGER"),
        bigquery.SchemaField("SABONO_PROM",      "FLOAT"),
        bigquery.SchemaField("SDSBOLSO",         "FLOAT"),
        bigquery.SchemaField("SACTUAL",          "FLOAT"),
        bigquery.SchemaField("calf_sbs_texto",   "STRING"),
        bigquery.SchemaField("calf_sbs_cod",     "INTEGER"),
        bigquery.SchemaField("clasif_rcc_max",   "INTEGER"),
        bigquery.SchemaField("BDSBOLSO",         "STRING"),
        bigquery.SchemaField("ATPRESTAMO",       "STRING"),
        bigquery.SchemaField("NRO_CUOTAS",       "FLOAT"),
        bigquery.SchemaField("NCUOTAS_PAG",      "FLOAT"),
        bigquery.SchemaField("CUO_PEN_PAGO",     "FLOAT"),
        bigquery.SchemaField("SCUOTA",           "FLOAT"),
        bigquery.SchemaField("STASA",            "FLOAT"),
        bigquery.SchemaField("FAPERTURA",        "STRING"),
        bigquery.SchemaField("SEXO",             "STRING"),
        bigquery.SchemaField("AUNID_EJECT",      "STRING"),
        bigquery.SchemaField("ALUGA_EMISI",      "STRING"),
    ]
)

uri = f"gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv"
load_job = client_bq.load_table_from_uri(uri, table_ref, job_config=job_config)
load_job.result()

table = client_bq.get_table(table_ref)
print(f"  Tabla: {table_ref}")
print(f"  Registros: {table.num_rows:,}")

# =============================================
# RESUMEN
# =============================================
print("\n" + "=" * 55)
print("PIPELINE COMPLETADO")
print("=" * 55)
print(f"  CSV local:     {csv_local}")
print(f"  Cloud Storage: gs://{BUCKET_NAME}/datos/clientes_segmentados_bq.csv")
print(f"  BigQuery:      {table_ref}")
print(f"  Registros:     {table.num_rows:,}")
print(f"  Columnas:      {len(table.schema)}")
print("=" * 55)
