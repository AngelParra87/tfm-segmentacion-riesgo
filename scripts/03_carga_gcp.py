"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 3: preparación y carga a GCP
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Periodo: noviembre de 2024

Construye la tabla de consumo para Power BI a partir de la cartera, el
resultado del clustering y la información RCC. Antes de cargar los datos
valida la alineación de las operaciones, el cruce por Codigo_Cliente_Sbs,
el número total de registros y la distribución de los cuatro clústeres.
La tabla final se carga primero en Cloud Storage y después en BigQuery.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from google.cloud import storage, bigquery
from google.oauth2 import service_account


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

PROJECT_ID = "tfm-segmentacion-riesgo"
BUCKET_NAME = "tfm-segmentacion-datos"
DATASET_ID = "cartera_riesgo"
TABLE_ID = "clientes_segmentados"

# Mantener las credenciales fuera del repositorio.
CREDENTIALS = r"C:\ANGEL\UNIR\gcp_config\gcp_credentials.json"

DATA_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data")
OUTPUT_DIR = DATA_DIR / "outputs" / "clustering"

FILE_COD = DATA_DIR / "cartera_creditos.txt"
FILE_RCC = DATA_DIR / "reporte_crediticio_rcc.txt"
FILE_CLUSTER = OUTPUT_DIR / "resultado_clustering.csv"

CSV_LOCAL = OUTPUT_DIR / "clientes_segmentados_bq.csv"
GCS_OBJECT = "datos/clientes_segmentados_bq.csv"

EXPECTED_ROWS = 597_127
EXPECTED_CLUSTER_COUNTS = {
    0: 11_865,
    1: 565_639,
    2: 9_797,
    3: 9_826,
}

MAP_CLUSTER = {
    0: "Cartera en Riesgo",
    1: "Cartera Vigente",
    2: "Cartera Castigada",
    3: "Cartera Judicial",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_clave(serie: pd.Series) -> pd.Series:
    """Normaliza identificadores sin convertir nulos en el literal 'nan'."""
    return (
        serie.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def normalizar_texto(serie: pd.Series, valor_nulo: str = "Sin identificar") -> pd.Series:
    """Limpia campos de texto conservando explícitamente los valores ausentes."""
    return (
        serie.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna(valor_nulo)
    )


def verificar_archivo(ruta: Path) -> None:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")


for ruta in [FILE_COD, FILE_RCC, FILE_CLUSTER]:
    verificar_archivo(ruta)

if not os.path.exists(CREDENTIALS):
    raise FileNotFoundError(
        "No se encontró el archivo de credenciales GCP en la ruta configurada. "
        f"Ruta: {CREDENTIALS}"
    )

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

print("=" * 72)
print("ETAPA 3 — PREPARACIÓN Y CARGA A GCP")
print("=" * 72)

print("\n[1/6] Cargando datasets...")

COLS_COD = [
    "Cliente",
    "Codigo_Cliente_Sbs",  # solo para trazabilidad / cruce RCC; no se publica en BQ
    "Tipo_Prestamo",
    "Sexo",
    "Desc_Unidad_Ejecutora",
    "Desc_Lugar_Emision",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Numero_Cuotas",
    "Cuotas_Pagadas",
    "Cuotas_Vencidas",
    "Cuotas_Pendientes",
    "Dias_Mora",
    "Abono_Promedio",
    "Monto_Cuota",
    "Tasa_Interes",
    "Estado_Credito",
    "Fecha_Apertura",
    "Calificacion_Sbs",
]

df_cod = pd.read_csv(
    FILE_COD,
    sep=";",
    encoding="latin-1",
    usecols=COLS_COD,
    low_memory=False,
    dtype=str,
)

df_rcc = pd.read_csv(
    FILE_RCC,
    sep=";",
    encoding="latin-1",
    low_memory=False,
    dtype=str,
)

df_cluster = pd.read_csv(
    FILE_CLUSTER,
    dtype={
        "Cliente": "string",
        "Codigo_Cliente_Sbs": "string",
        "cluster": "Int64",
    },
)

print(f"  Cartera   : {len(df_cod):,} registros")
print(f"  RCC       : {len(df_rcc):,} registros")
print(f"  Clustering: {len(df_cluster):,} registros")

if len(df_cod) != EXPECTED_ROWS:
    raise AssertionError(
        f"Cartera contiene {len(df_cod):,} filas; "
        f"para nov-2024 se esperaban {EXPECTED_ROWS:,}."
    )

if len(df_cluster) != len(df_cod):
    raise AssertionError(
        "El resultado de clustering no tiene la misma cantidad de operaciones "
        "que el dataset de cartera."
    )

columnas_cluster_requeridas = ["Cliente", "Codigo_Cliente_Sbs", "cluster"]
faltantes_cluster = [
    c for c in columnas_cluster_requeridas if c not in df_cluster.columns
]
if faltantes_cluster:
    raise KeyError(
        "Faltan columnas en resultado_clustering.csv: "
        + ", ".join(faltantes_cluster)
    )


# ============================================================
# 2. VALIDACIÓN DE ALINEACIÓN CARTERA ↔ CLUSTER
# ============================================================

print("\n[2/6] Validando alineación entre cartera y resultado_clustering...")

df_cod["Cliente"] = normalizar_clave(df_cod["Cliente"])
df_cod["Codigo_Cliente_Sbs"] = normalizar_clave(df_cod["Codigo_Cliente_Sbs"])

df_cluster["Cliente"] = normalizar_clave(df_cluster["Cliente"])
df_cluster["Codigo_Cliente_Sbs"] = normalizar_clave(
    df_cluster["Codigo_Cliente_Sbs"]
)

# resultado_clustering.csv se genera preservando el orden original del dataset.
# Antes de asignar el cluster por posición, se valida fila a fila usando
# Cliente + Codigo_Cliente_Sbs.
clientes_iguales = (
    df_cod["Cliente"].fillna("<NA>").reset_index(drop=True)
    == df_cluster["Cliente"].fillna("<NA>").reset_index(drop=True)
)

codigos_iguales = (
    df_cod["Codigo_Cliente_Sbs"].fillna("<NA>").reset_index(drop=True)
    == df_cluster["Codigo_Cliente_Sbs"].fillna("<NA>").reset_index(drop=True)
)

n_mismatch_cliente = int((~clientes_iguales).sum())
n_mismatch_sbs = int((~codigos_iguales).sum())

print(f"  Diferencias por Cliente          : {n_mismatch_cliente:,}")
print(f"  Diferencias por Codigo_Cliente_Sbs: {n_mismatch_sbs:,}")

if n_mismatch_cliente > 0 or n_mismatch_sbs > 0:
    raise AssertionError(
        "El archivo resultado_clustering.csv no está alineado fila a fila con "
        "cartera_creditos.txt. No se realizará la carga."
    )

print("  Alineación fila a fila: [OK]")


# ============================================================
# 3. PREPARACIÓN DE VARIABLES DERIVADAS
# ============================================================

print("\n[3/6] Preparando variables derivadas...")

# ------------------------------------------------------------
# 3.1 Peor_Calificacion_Rcc mediante Codigo_Cliente_Sbs
# ------------------------------------------------------------

COLUMNAS_RCC_REQUERIDAS = [
    "Codigo_Cliente_Sbs",
    "Calificacion_Entidad",
]

faltantes_rcc = [c for c in COLUMNAS_RCC_REQUERIDAS if c not in df_rcc.columns]
if faltantes_rcc:
    raise KeyError(
        "Faltan columnas RCC obligatorias: " + ", ".join(faltantes_rcc)
    )

df_rcc["Codigo_Cliente_Sbs"] = normalizar_clave(
    df_rcc["Codigo_Cliente_Sbs"]
)
df_rcc["Calificacion_Entidad"] = pd.to_numeric(
    df_rcc["Calificacion_Entidad"],
    errors="coerce",
)

df_rcc_max = (
    df_rcc
    .dropna(subset=["Codigo_Cliente_Sbs"])
    .groupby("Codigo_Cliente_Sbs", as_index=False)
    .agg(
        Peor_Calificacion_Rcc=("Calificacion_Entidad", "max")
    )
)

if df_rcc_max["Codigo_Cliente_Sbs"].duplicated().any():
    raise ValueError(
        "df_rcc_max contiene Codigo_Cliente_Sbs duplicados; "
        "el merge m:1 no es seguro."
    )

filas_antes_rcc = len(df_cod)

df_cod = df_cod.merge(
    df_rcc_max,
    on="Codigo_Cliente_Sbs",
    how="left",
    validate="m:1",
    indicator="_merge_rcc",
)

filas_despues_rcc = len(df_cod)

print(f"  Filas antes del merge RCC : {filas_antes_rcc:,}")
print(f"  Filas después del merge RCC: {filas_despues_rcc:,}")

if filas_despues_rcc != filas_antes_rcc:
    raise AssertionError(
        "El merge RCC modificó la cantidad de operaciones."
    )

operaciones_con_rcc = int((df_cod["_merge_rcc"] == "both").sum())
pct_operaciones_rcc = operaciones_con_rcc / len(df_cod) * 100

print(
    f"  Operaciones con correspondencia RCC: "
    f"{operaciones_con_rcc:,} ({pct_operaciones_rcc:.2f}%)"
)

# Para operaciones sin correspondencia externa se utiliza 0,
# igual que en 02_preprocesamiento_clustering.py.
df_cod["Peor_Calificacion_Rcc"] = (
    df_cod["Peor_Calificacion_Rcc"]
    .fillna(0)
    .astype(int)
)

df_cod.drop(columns="_merge_rcc", inplace=True)

# ------------------------------------------------------------
# 3.2 Asignación de clúster
# ------------------------------------------------------------

df_cluster = df_cluster.reset_index(drop=True)
df_cod = df_cod.reset_index(drop=True)

df_cod["cluster"] = df_cluster["cluster"].astype(int)
df_cod["cluster_nombre"] = df_cod["cluster"].map(MAP_CLUSTER)

if df_cod["cluster_nombre"].isna().any():
    clusters_invalidos = sorted(
        df_cod.loc[df_cod["cluster_nombre"].isna(), "cluster"]
        .dropna()
        .unique()
        .tolist()
    )
    raise ValueError(
        f"Se encontraron clústeres sin nombre descriptivo: {clusters_invalidos}"
    )

# ------------------------------------------------------------
# 3.3 Calificación SBS para visualización
# ------------------------------------------------------------

calif_original = (
    df_cod["Calificacion_Sbs"]
    .astype("string")
    .str.strip()
)

MAP_CALIF_TEXTO = {
    "2. NORMAL": "Normal",
    "3. CPP": "CPP",
    "4. DEFICIENTE": "Deficiente",
    "5. DUDOSO": "Dudoso",
    "6. PERDIDA": "Pérdida",
    "1. NO DEFINIDO": "No definido",
}

df_cod["Calificacion_Sbs_Texto"] = (
    calif_original
    .map(MAP_CALIF_TEXTO)
    .fillna("Sin calificación")
)

# Código ordinal usado para facilitar ordenamientos en Power BI.
MAP_CALIF_NUM = {
    "Sin calificación": 0,
    "No definido": 1,
    "Normal": 1,
    "CPP": 2,
    "Deficiente": 3,
    "Dudoso": 4,
    "Pérdida": 5,
}

df_cod["Calificacion_Sbs_Cod"] = (
    df_cod["Calificacion_Sbs_Texto"]
    .map(MAP_CALIF_NUM)
    .astype(int)
)

# ------------------------------------------------------------
# 3.4 Limpieza de campos de texto
# ------------------------------------------------------------

df_cod["Tipo_Prestamo"] = normalizar_texto(df_cod["Tipo_Prestamo"])
df_cod["Desc_Unidad_Ejecutora"] = normalizar_texto(
    df_cod["Desc_Unidad_Ejecutora"]
)
df_cod["Desc_Lugar_Emision"] = normalizar_texto(
    df_cod["Desc_Lugar_Emision"]
)
df_cod["Estado_Credito"] = normalizar_texto(df_cod["Estado_Credito"])
df_cod["Sexo"] = normalizar_texto(df_cod["Sexo"])

# ------------------------------------------------------------
# 3.5 Conversión explícita de variables numéricas
# ------------------------------------------------------------

NUMERIC_FLOAT = [
    "Dias_Mora",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Monto_Cuota",
    "Tasa_Interes",
]

NUMERIC_INT = [
    "Cuotas_Vencidas",
]

NUMERIC_OPTIONAL = [
    "Numero_Cuotas",
    "Cuotas_Pagadas",
    "Cuotas_Pendientes",
]

for col in NUMERIC_FLOAT:
    df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

for col in NUMERIC_INT:
    df_cod[col] = (
        pd.to_numeric(df_cod[col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

for col in NUMERIC_OPTIONAL:
    df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")


# ============================================================
# 4. CONSTRUCCIÓN DE TABLA DESNORMALIZADA
# ============================================================

print("\n[4/6] Construyendo tabla desnormalizada para Power BI...")

COLS_BQ = [
    "Cliente",
    "cluster",
    "cluster_nombre",
    # Variables de clustering
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Calificacion_Sbs_Texto",
    "Calificacion_Sbs_Cod",
    "Peor_Calificacion_Rcc",
    # Variable de validación
    "Estado_Credito",
    # Contexto del crédito
    "Tipo_Prestamo",
    "Numero_Cuotas",
    "Cuotas_Pagadas",
    "Cuotas_Pendientes",
    "Monto_Cuota",
    "Tasa_Interes",
    "Fecha_Apertura",
    # Demografía y ubicación
    "Sexo",
    "Desc_Unidad_Ejecutora",
    "Desc_Lugar_Emision",
]

df_final = df_cod[COLS_BQ].copy()

print(
    f"  Tabla final: {len(df_final):,} registros, "
    f"{len(df_final.columns)} columnas"
)

if len(df_final) != EXPECTED_ROWS:
    raise AssertionError(
        f"La tabla final tiene {len(df_final):,} filas; "
        f"se esperaban {EXPECTED_ROWS:,}."
    )

if len(df_final.columns) != 22:
    raise AssertionError(
        f"La tabla final tiene {len(df_final.columns)} columnas; se esperaban 22."
    )

# ------------------------------------------------------------
# 4.1 Validación de distribución de clústeres
# ------------------------------------------------------------

cluster_counts = (
    df_final["cluster"]
    .value_counts()
    .sort_index()
    .to_dict()
)

print("\n  Distribución de créditos activos por clúster:")
for cluster_id in sorted(cluster_counts):
    print(
        f"    Clúster {cluster_id} "
        f"({MAP_CLUSTER.get(cluster_id, 'Sin nombre')}): "
        f"{cluster_counts[cluster_id]:,}"
    )

if cluster_counts != EXPECTED_CLUSTER_COUNTS:
    raise AssertionError(
        "La distribución de clústeres no coincide con la salida validada de "
        "02_preprocesamiento_clustering.py.\n"
        f"Esperado: {EXPECTED_CLUSTER_COUNTS}\n"
        f"Actual  : {cluster_counts}"
    )

print("  Distribución de clústeres: [OK]")

# ------------------------------------------------------------
# 4.2 Validaciones adicionales
# ------------------------------------------------------------

if df_final["cluster"].isna().any():
    raise AssertionError("Existen operaciones sin clúster.")

if df_final["cluster_nombre"].isna().any():
    raise AssertionError("Existen operaciones sin nombre de clúster.")

if df_final["Peor_Calificacion_Rcc"].isna().any():
    raise AssertionError("Existen nulos en Peor_Calificacion_Rcc.")

# Verificación de clientes únicos para trazabilidad
clientes_unicos = df_final["Cliente"].nunique(dropna=True)
print(f"  Clientes únicos: {clientes_unicos:,}")

# Guardar CSV local
df_final.to_csv(
    CSV_LOCAL,
    index=False,
    encoding="utf-8",
)

print(f"  CSV guardado: {CSV_LOCAL}")


# ============================================================
# 5. SUBIDA A CLOUD STORAGE
# ============================================================

print("\n[5/6] Subiendo archivo validado a Cloud Storage...")

client_gcs = storage.Client(
    project=PROJECT_ID,
    credentials=credentials,
)

bucket = client_gcs.bucket(BUCKET_NAME)
blob = bucket.blob(GCS_OBJECT)
blob.upload_from_filename(str(CSV_LOCAL))

gcs_uri = f"gs://{BUCKET_NAME}/{GCS_OBJECT}"
print(f"  Subido: {gcs_uri}")


# ============================================================
# 6. CARGA EN BIGQUERY
# ============================================================

print("\n[6/6] Cargando archivo en BigQuery...")

client_bq = bigquery.Client(
    project=PROJECT_ID,
    credentials=credentials,
)

table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=False,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    schema=[
        bigquery.SchemaField("Cliente", "INTEGER"),
        bigquery.SchemaField("cluster", "INTEGER"),
        bigquery.SchemaField("cluster_nombre", "STRING"),
        bigquery.SchemaField("Dias_Mora", "FLOAT"),
        bigquery.SchemaField("Cuotas_Vencidas", "INTEGER"),
        bigquery.SchemaField("Abono_Promedio", "FLOAT"),
        bigquery.SchemaField("Saldo_Desembolsado", "FLOAT"),
        bigquery.SchemaField("Saldo_Vigente", "FLOAT"),
        bigquery.SchemaField("Calificacion_Sbs_Texto", "STRING"),
        bigquery.SchemaField("Calificacion_Sbs_Cod", "INTEGER"),
        bigquery.SchemaField("Peor_Calificacion_Rcc", "INTEGER"),
        bigquery.SchemaField("Estado_Credito", "STRING"),
        bigquery.SchemaField("Tipo_Prestamo", "STRING"),
        bigquery.SchemaField("Numero_Cuotas", "FLOAT"),
        bigquery.SchemaField("Cuotas_Pagadas", "FLOAT"),
        bigquery.SchemaField("Cuotas_Pendientes", "FLOAT"),
        bigquery.SchemaField("Monto_Cuota", "FLOAT"),
        bigquery.SchemaField("Tasa_Interes", "FLOAT"),
        bigquery.SchemaField("Fecha_Apertura", "STRING"),
        bigquery.SchemaField("Sexo", "STRING"),
        bigquery.SchemaField("Desc_Unidad_Ejecutora", "STRING"),
        bigquery.SchemaField("Desc_Lugar_Emision", "STRING"),
    ],
)

load_job = client_bq.load_table_from_uri(
    gcs_uri,
    table_ref,
    job_config=job_config,
)
load_job.result()

table = client_bq.get_table(table_ref)

print(f"  Tabla cargada: {table_ref}")
print(f"  Registros reportados por BigQuery: {table.num_rows:,}")
print(f"  Columnas reportadas por BigQuery : {len(table.schema)}")

if table.num_rows != EXPECTED_ROWS:
    raise AssertionError(
        f"BigQuery contiene {table.num_rows:,} registros; "
        f"se esperaban {EXPECTED_ROWS:,}."
    )

if len(table.schema) != 22:
    raise AssertionError(
        f"BigQuery contiene {len(table.schema)} columnas; se esperaban 22."
    )

# ------------------------------------------------------------
# 6.1 Validación de distribución directamente en BigQuery
# ------------------------------------------------------------

query = f"""
SELECT
    cluster,
    cluster_nombre,
    COUNT(*) AS n_creditos
FROM `{table_ref}`
GROUP BY cluster, cluster_nombre
ORDER BY cluster
"""

df_bq_clusters = client_bq.query(query).to_dataframe()

print("\n  Distribución en BigQuery:")
print(df_bq_clusters.to_string(index=False))

bq_counts = {
    int(row["cluster"]): int(row["n_creditos"])
    for _, row in df_bq_clusters.iterrows()
}

if bq_counts != EXPECTED_CLUSTER_COUNTS:
    raise AssertionError(
        "La distribución de clústeres cargada en BigQuery no coincide con "
        "la salida validada del modelo."
    )

print("  Validación BigQuery por clúster: [OK]")


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 72)
print("PIPELINE GCP COMPLETADO Y VALIDADO")
print("=" * 72)
print(f"  CSV local       : {CSV_LOCAL}")
print(f"  Cloud Storage   : {gcs_uri}")
print(f"  BigQuery        : {table_ref}")
print(f"  Registros       : {table.num_rows:,}")
print(f"  Clientes únicos : {clientes_unicos:,}")
print(f"  Columnas        : {len(table.schema)}")
print("=" * 72)
print("\nSiguiente paso: actualizar Power BI desde BigQuery.")
