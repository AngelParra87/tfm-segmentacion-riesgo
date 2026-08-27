"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 2: preprocesamiento y clustering
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Periodo: noviembre de 2024

Integra la cartera con el RCC mediante Codigo_Cliente_Sbs, prepara las siete
variables del modelo y evalúa MiniBatchKMeans para k=2 a k=10. El modelo final
utiliza k=4. Las métricas internas se calculan sobre una muestra fija de
30.000 operaciones, mientras que la inercia corresponde al ajuste sobre la
cartera completa. Estado_Credito se reserva para la validación externa.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
DATA_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data")
OUTPUT_DIR = DATA_DIR / "outputs" / "clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 10_000
N_INIT = 15
MAX_ITER = 300
K_RANGE = range(2, 11)
METRIC_SAMPLE_SIZE = 30_000
K_FINAL = 4

EXPECTED_COD_ROWS = 597_127
EXPECTED_RCC_ROWS = 1_191_640


def resolver_archivo(base_dir: Path, nombre_base: str) -> Path:
    """Busca primero .txt y luego el mismo nombre sin extensión."""
    candidatos = [base_dir / f"{nombre_base}.txt", base_dir / nombre_base]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró '{nombre_base}.txt' ni '{nombre_base}' en {base_dir}"
    )


def normalizar_clave(serie: pd.Series) -> pd.Series:
    """Limpia claves sin transformar nulos en el literal 'nan'."""
    return serie.astype("string").str.strip().replace("", pd.NA)


def control_filas(nombre: str, actual: int, esperado: int) -> None:
    if actual == esperado:
        print(f"  {nombre}: {actual:,} registros [OK]")
    else:
        print(
            f"  {nombre}: {actual:,} registros "
            f"[ADVERTENCIA: nov-2024 esperaba {esperado:,}]"
        )


COD_FILE = resolver_archivo(DATA_DIR, "cartera_creditos")
RCC_FILE = resolver_archivo(DATA_DIR, "reporte_crediticio_rcc")

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("=" * 72)
print("ETAPA 2 — PREPROCESAMIENTO Y CLUSTERING")
print("=" * 72)
print("\n[1/8] Cargando datasets...")
print(f"  Cartera: {COD_FILE}")
print(f"  RCC    : {RCC_FILE}")

NUMERIC_COLS_COD = [
    "Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio",
    "Saldo_Desembolsado", "Saldo_Vigente", "Capital_Vencido",
    "Capital_Judicial", "Saldo_Mora", "Monto_Cuota", "Tasa_Interes",
    "Saldo_Provision", "Numero_Cuotas", "Cuotas_Pagadas",
]

df_cod = pd.read_csv(
    COD_FILE, sep=";", encoding="latin-1", low_memory=False, dtype=str
)

for col in NUMERIC_COLS_COD:
    if col in df_cod.columns:
        df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

for col in ["Cliente", "Codigo_Cliente_Sbs"]:
    if col not in df_cod.columns:
        raise KeyError(f"Falta la columna obligatoria '{col}' en cartera_creditos.")
    df_cod[col] = normalizar_clave(df_cod[col])

df_rcc = pd.read_csv(
    RCC_FILE, sep=";", encoding="latin-1", low_memory=False, dtype=str
)

COLUMNAS_RCC = [
    "Codigo_Cliente_Sbs", "Calificacion_Entidad", "Saldo", "Entidad_Financiera"
]
faltantes = [c for c in COLUMNAS_RCC if c not in df_rcc.columns]
if faltantes:
    raise KeyError("Faltan columnas en RCC: " + ", ".join(faltantes))

df_rcc["Codigo_Cliente_Sbs"] = normalizar_clave(df_rcc["Codigo_Cliente_Sbs"])
df_rcc["Saldo"] = pd.to_numeric(df_rcc["Saldo"], errors="coerce")
df_rcc["Calificacion_Entidad"] = pd.to_numeric(
    df_rcc["Calificacion_Entidad"], errors="coerce"
)

control_filas("Cartera", len(df_cod), EXPECTED_COD_ROWS)
control_filas("RCC", len(df_rcc), EXPECTED_RCC_ROWS)
print(f"  Clientes únicos en cartera: {df_cod['Cliente'].nunique(dropna=True):,}")
print(
    "  Códigos SBS válidos y únicos en cartera: "
    f"{df_cod['Codigo_Cliente_Sbs'].nunique(dropna=True):,}"
)
print(
    "  Códigos SBS únicos en RCC: "
    f"{df_rcc['Codigo_Cliente_Sbs'].nunique(dropna=True):,}"
)

# ============================================================
# 2. FEATURE ENGINEERING RCC
# ============================================================
print("\n[2/8] Construyendo indicadores RCC...")

rcc_agg = (
    df_rcc
    .dropna(subset=["Codigo_Cliente_Sbs"])
    .groupby("Codigo_Cliente_Sbs", as_index=False)
    .agg(
        Peor_Calificacion_Rcc=("Calificacion_Entidad", "max"),
        Saldo_Rcc_Total=("Saldo", "sum"),
        N_Entidades=("Entidad_Financiera", "nunique"),
    )
)

if rcc_agg["Codigo_Cliente_Sbs"].duplicated().any():
    raise ValueError("La agregación RCC no produjo una clave única por Codigo_Cliente_Sbs.")

print(f"  Códigos SBS agregados en RCC: {len(rcc_agg):,}")

# ============================================================
# 3. MERGE CARTERA + RCC POR Codigo_Cliente_Sbs
# ============================================================
print("\n[3/8] Integrando cartera con RCC mediante Codigo_Cliente_Sbs...")

filas_antes = len(df_cod)
df = df_cod.merge(
    rcc_agg,
    on="Codigo_Cliente_Sbs",
    how="left",
    validate="m:1",
    indicator=True,
)
filas_despues = len(df)

print(f"  Filas antes del merge  : {filas_antes:,}")
print(f"  Filas después del merge: {filas_despues:,}")
if filas_despues != filas_antes:
    raise AssertionError("El merge con RCC modificó el número de operaciones.")

# Cobertura real ANTES de fillna(0)
codigos_validos = df_cod["Codigo_Cliente_Sbs"].nunique(dropna=True)
codigos_con_rcc = df.loc[
    df["_merge"] == "both", "Codigo_Cliente_Sbs"
].nunique(dropna=True)

clientes_unicos = df_cod["Cliente"].nunique(dropna=True)
clientes_con_rcc = df.loc[
    df["_merge"] == "both", "Cliente"
].nunique(dropna=True)

operaciones_con_rcc = int((df["_merge"] == "both").sum())

pct_codigos_rcc = codigos_con_rcc / codigos_validos * 100 if codigos_validos else np.nan
pct_clientes_rcc = clientes_con_rcc / clientes_unicos * 100 if clientes_unicos else np.nan
pct_operaciones_rcc = operaciones_con_rcc / len(df) * 100 if len(df) else np.nan

print("\n  Cobertura RCC:")
print(f"    Códigos SBS válidos en cartera : {codigos_validos:,}")
print(f"    Códigos SBS con RCC             : {codigos_con_rcc:,}")
print(f"    Cobertura por código SBS        : {pct_codigos_rcc:.2f}%")
print(f"    Clientes únicos en cartera      : {clientes_unicos:,}")
print(f"    Clientes únicos con RCC         : {clientes_con_rcc:,}")
print(f"    Cobertura por cliente           : {pct_clientes_rcc:.2f}%")
print(f"    Operaciones con RCC             : {operaciones_con_rcc:,}")
print(f"    Cobertura por operación         : {pct_operaciones_rcc:.2f}%")

pd.DataFrame({
    "metrica": [
        "filas_cartera", "filas_rcc", "clientes_unicos_cartera",
        "codigos_sbs_validos_cartera", "codigos_sbs_con_rcc",
        "pct_cobertura_codigo_sbs", "clientes_unicos_con_rcc",
        "pct_cobertura_cliente", "operaciones_con_rcc",
        "pct_cobertura_operacion",
    ],
    "valor": [
        len(df_cod), len(df_rcc), clientes_unicos, codigos_validos,
        codigos_con_rcc, round(pct_codigos_rcc, 4), clientes_con_rcc,
        round(pct_clientes_rcc, 4), operaciones_con_rcc,
        round(pct_operaciones_rcc, 4),
    ],
}).to_csv(OUTPUT_DIR / "cobertura_rcc.csv", index=False, encoding="utf-8")
print("  → cobertura_rcc.csv guardado")

# Imputación solo después de medir cobertura
df["Peor_Calificacion_Rcc"] = df["Peor_Calificacion_Rcc"].fillna(0)
df["Saldo_Rcc_Total"] = df["Saldo_Rcc_Total"].fillna(0)
df["N_Entidades"] = df["N_Entidades"].fillna(0)
df.drop(columns="_merge", inplace=True)

# ============================================================
# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================
print("\n[4/8] Codificando variables categóricas...")

map_calf = {
    "2. NORMAL": 1,
    "3. CPP": 2,
    "4. DEFICIENTE": 3,
    "5. DUDOSO": 4,
    "6. PERDIDA": 5,
    "1. NO DEFINIDO": 1,
}

if "Calificacion_Sbs" not in df.columns:
    raise KeyError("Falta la columna Calificacion_Sbs.")

df["Calificacion_Sbs"] = (
    df["Calificacion_Sbs"].astype("string").str.strip()
    .map(map_calf).fillna(1).astype(int)
)

# Estado_Credito no entra al modelo: solo validación externa.
if "Estado_Credito" not in df.columns:
    raise KeyError("Falta la columna Estado_Credito.")

estado_limpio = df["Estado_Credito"].astype("string").str.strip().str.upper()
map_estado_cod = {
    "ACT": "ACT", "ACTIVO": "ACT",
    "REF": "REF", "REFINANCIADO": "REF",
    "JUD": "JUD", "JUDICIAL": "JUD",
    "CAS": "CAS", "CASTIGADO": "CAS",
}
map_estado_num = {"ACT": 1, "REF": 2, "JUD": 3, "CAS": 4}

df["Estado_Credito_Cod"] = estado_limpio.map(map_estado_cod)
df["Estado_Credito_Num"] = (
    df["Estado_Credito_Cod"].map(map_estado_num).fillna(1).astype(int)
)
df["Estado_Credito"] = df["Estado_Credito_Cod"].fillna(estado_limpio)

print("  Distribución Estado_Credito:")
print(df["Estado_Credito"].value_counts(dropna=False).to_string())

# ============================================================
# 5. VARIABLES DE CLUSTERING
# ============================================================
print("\n[5/8] Preparando variables de clustering...")

CLUSTER_VARS = [
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Calificacion_Sbs",
    "Peor_Calificacion_Rcc",
]

faltantes_cluster = [c for c in CLUSTER_VARS if c not in df.columns]
if faltantes_cluster:
    raise KeyError("Faltan variables de clustering: " + ", ".join(faltantes_cluster))

df_cluster = df[[
    "Cliente", "Codigo_Cliente_Sbs", "Estado_Credito", "Estado_Credito_Num"
] + CLUSTER_VARS].copy()

for col in CLUSTER_VARS:
    df_cluster[col] = pd.to_numeric(df_cluster[col], errors="coerce")

nulos_pre = df_cluster[CLUSTER_VARS].isnull().sum()
if (nulos_pre > 0).any():
    print("  Nulos antes de imputación:")
    print(nulos_pre[nulos_pre > 0].to_string())
else:
    print("  No se detectaron nulos en variables de clustering.")

for col in CLUSTER_VARS:
    n_nulos = int(df_cluster[col].isnull().sum())
    if n_nulos > 0:
        mediana = df_cluster[col].median()
        if pd.isna(mediana):
            raise ValueError(f"No puede imputarse {col}: todos los valores son nulos.")
        df_cluster[col] = df_cluster[col].fillna(mediana)
        print(f"  {col}: {n_nulos:,} nulos → mediana={mediana:.4f}")

# ============================================================
# 6. WINSORIZACIÓN P1-P99
# ============================================================
print("\n[6/8] Winsorizando variables continuas (P1-P99)...")

VARS_WINSORIZACION = [
    "Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio",
    "Saldo_Desembolsado", "Saldo_Vigente",
]

resumen_win = []
for col in VARS_WINSORIZACION:
    p01 = float(df_cluster[col].quantile(0.01))
    p99 = float(df_cluster[col].quantile(0.99))
    min_original = float(df_cluster[col].min())
    max_original = float(df_cluster[col].max())
    df_cluster[col] = df_cluster[col].clip(lower=p01, upper=p99)
    resumen_win.append({
        "variable": col,
        "p01": p01,
        "p99": p99,
        "min_original": min_original,
        "max_original": max_original,
    })
    print(
        f"  {col}: P1={p01:,.4f} | P99={p99:,.4f} | "
        f"máx. original={max_original:,.4f}"
    )

pd.DataFrame(resumen_win).to_csv(
    OUTPUT_DIR / "resumen_winsorizacion.csv", index=False, encoding="utf-8"
)
print("  → resumen_winsorizacion.csv guardado")

# ============================================================
# 7. ESCALADO — RobustScaler
# ============================================================
print("\n[7/8] Escalando las siete variables con RobustScaler...")

scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df_cluster[CLUSTER_VARS]),
    columns=CLUSTER_VARS,
    index=df_cluster.index,
)

print(f"  Matriz escalada: {X_scaled.shape[0]:,} x {X_scaled.shape[1]}")
print("  Medianas post-escala:")
print(X_scaled.median().round(4).to_string())

# ============================================================
# 8. EVALUACIÓN k=2..10
# ============================================================
print("\n[8/8] Evaluando MiniBatchKMeans para k=2 a k=10...")
print(
    f"  Configuración: batch_size={BATCH_SIZE:,}, n_init={N_INIT}, "
    f"max_iter={MAX_ITER}, random_state={RANDOM_STATE}"
)

np.random.seed(RANDOM_STATE)
idx_sample = np.random.choice(
    len(X_scaled), size=min(METRIC_SAMPLE_SIZE, len(X_scaled)), replace=False
)
X_sample = X_scaled.iloc[idx_sample].to_numpy()
print(f"  Muestra fija para Silhouette/DB/CH: {len(idx_sample):,} operaciones")

inercias, silhouettes, davies_bouldins, calinski_scores = [], [], [], []

for k in K_RANGE:
    print(f"  Probando k={k}...", end=" ", flush=True)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )
    model.fit(X_scaled)

    labels_sample = model.labels_[idx_sample]
    inercia = float(model.inertia_)
    sil = float(silhouette_score(X_sample, labels_sample))
    db = float(davies_bouldin_score(X_sample, labels_sample))
    ch = float(calinski_harabasz_score(X_sample, labels_sample))

    inercias.append(inercia)
    silhouettes.append(sil)
    davies_bouldins.append(db)
    calinski_scores.append(ch)

    print(
        f"inercia={inercia:,.0f} | silhouette={sil:.4f} | "
        f"DB={db:.4f} | CH={ch:,.2f}"
    )

metricas_df = pd.DataFrame({
    "k": list(K_RANGE),
    "inercia": inercias,
    "silhouette": silhouettes,
    "davies_bouldin": davies_bouldins,
    "calinski_harabasz": calinski_scores,
    "muestra_metricas": len(idx_sample),
    "batch_size": BATCH_SIZE,
    "n_init": N_INIT,
    "max_iter": MAX_ITER,
    "random_state": RANDOM_STATE,
})
metricas_df.to_csv(
    OUTPUT_DIR / "metricas_clustering.csv", index=False, encoding="utf-8"
)
print("\nMétricas k=2..10:")
print(metricas_df.to_string(index=False))

# Figura 06 — evolución de las cuatro métricas
# Se marca k=4 como configuración finalmente seleccionada.
# No se denomina "k óptimo", ya que k=2 presenta el mayor silhouette
# y k=3 el mayor Calinski-Harabasz.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.ravel()

series_fig06 = [
    (
        inercias,
        "Inercia por k",
        "Inercia (WCSS)",
    ),
    (
        silhouettes,
        "Índice de silueta por k",
        "Silhouette Score (mayor es mejor)",
    ),
    (
        davies_bouldins,
        "Davies-Bouldin por k",
        "Davies-Bouldin (menor es mejor)",
    ),
    (
        calinski_scores,
        "Calinski-Harabasz por k",
        "Calinski-Harabasz (mayor es mejor)",
    ),
]

for ax, (valores, titulo, ylabel) in zip(axes, series_fig06):
    ax.plot(
        list(K_RANGE),
        valores,
        marker="o",
        linewidth=2,
    )
    ax.axvline(
        x=K_FINAL,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"k seleccionado = {K_FINAL}",
    )
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Número de clústeres (k)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(K_RANGE))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
)
axes[3].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

fig.suptitle(
    "Evolución de métricas de clustering para k = 2 a k = 10",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    OUTPUT_DIR / "fig06_seleccion_k.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

k_mayor_sil = list(K_RANGE)[int(np.argmax(silhouettes))]
k_mayor_ch = list(K_RANGE)[int(np.argmax(calinski_scores))]
k_menor_db = list(K_RANGE)[int(np.argmin(davies_bouldins))]

print("  → fig06_seleccion_k.png guardado")
print(f"  → Mayor silhouette: k={k_mayor_sil}")
print(f"  → Menor Davies-Bouldin: k={k_menor_db}")
print(f"  → Mayor Calinski-Harabasz: k={k_mayor_ch}")
print(f"  → Configuración seleccionada para el modelo final: k={K_FINAL}")

# ============================================================
# 9. MODELO FINAL — k=4
# ============================================================
print("\n[MODELO FINAL] Ajustando MiniBatchKMeans con k=4...")

model_final = MiniBatchKMeans(
    n_clusters=K_FINAL,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    n_init=N_INIT,
    max_iter=MAX_ITER,
)
model_final.fit(X_scaled)
df_cluster["cluster"] = model_final.labels_.astype(int)

dist_clusters = df_cluster["cluster"].value_counts().sort_index()
print("\n  Distribución de créditos activos por clúster:")
print(dist_clusters.to_string())
print(f"  Total asignado: {int(dist_clusters.sum()):,}")

if int(dist_clusters.sum()) != len(df_cluster):
    raise AssertionError("La suma de créditos por clúster no coincide con el dataset.")

# ============================================================
# 10. PERFIL DE CLÚSTERES
# ============================================================
print("\n[PERFILADO] Calculando perfil promedio...")

VARS_PERFIL = CLUSTER_VARS.copy()
perfil = df_cluster.groupby("cluster")[VARS_PERFIL].mean().round(2)
perfil["n_creditos"] = dist_clusters.reindex(perfil.index).astype(int)
perfil["pct_cartera"] = (perfil["n_creditos"] / len(df_cluster) * 100).round(2)
print(perfil.to_string())
perfil.to_csv(OUTPUT_DIR / "perfil_clusters.csv", encoding="utf-8")

# ============================================================
# 10b. VALIDACIÓN EXTERNA CON Estado_Credito
# ============================================================
print("\n[VALIDACIÓN EXTERNA] Estado_Credito por clúster...")

estado_por_cluster = (
    df_cluster.groupby(["cluster", "Estado_Credito"]).size().unstack(fill_value=0)
)
for estado in ["ACT", "CAS", "JUD", "REF"]:
    if estado not in estado_por_cluster.columns:
        estado_por_cluster[estado] = 0
estado_por_cluster = estado_por_cluster[["ACT", "CAS", "JUD", "REF"]]
estado_por_cluster["TOTAL"] = estado_por_cluster.sum(axis=1)
for estado in ["ACT", "CAS", "JUD", "REF"]:
    estado_por_cluster[f"pct_{estado}"] = (
        estado_por_cluster[estado] / estado_por_cluster["TOTAL"] * 100
    ).round(1)

print(estado_por_cluster.to_string())
estado_por_cluster.to_csv(
    OUTPUT_DIR / "estado_credito_por_cluster.csv", encoding="utf-8"
)

print("\n  Pureza por clúster:")
for cluster_id in estado_por_cluster.index:
    fila = estado_por_cluster.loc[cluster_id, ["ACT", "CAS", "JUD", "REF"]]
    dominante = fila.idxmax()
    pct = fila.max() / estado_por_cluster.loc[cluster_id, "TOTAL"] * 100
    print(f"  Clúster {cluster_id}: {dominante} ({pct:.1f}%)")

# ============================================================
# 10c. FIGURA 07 — HEATMAP
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
perfil_norm = perfil[VARS_PERFIL].copy()
for col in perfil_norm.columns:
    rango = perfil_norm[col].max() - perfil_norm[col].min()
    if rango > 0:
        perfil_norm[col] = (perfil_norm[col] - perfil_norm[col].min()) / rango
    else:
        perfil_norm[col] = 0.0

sns.heatmap(
    perfil_norm.T,
    annot=perfil[VARS_PERFIL].T.round(1),
    fmt="g",
    cmap="YlOrRd",
    ax=ax,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
ax.set_title(f"Perfil de clústeres — MiniBatchKMeans (k={K_FINAL})", fontsize=12, pad=12)
ax.set_xlabel("Clúster")
ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig07_perfil_clusters_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig07_perfil_clusters_heatmap.png guardado")

# ============================================================
# 10d. FIGURA 08 — CRÉDITOS ACTIVOS POR CLÚSTER
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2563EB", "#059669", "#DC2626", "#F59E0B"]
bars = ax.bar(
    [f"Clúster {i}" for i in dist_clusters.index],
    dist_clusters.values,
    color=colors[:len(dist_clusters)],
    edgecolor="white",
)
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
ax.set_title(f"Distribución de créditos activos por clúster (k={K_FINAL})", fontsize=12)
ax.set_ylabel("Número de créditos activos")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig08_distribucion_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fig08_distribucion_clusters.png guardado")

# ============================================================
# 11. EXPORTAR RESULTADO FINAL
# ============================================================
# Se conserva Cliente por compatibilidad con 03_carga_gcp.py y se añade
# Codigo_Cliente_Sbs para trazabilidad. Una fila sigue siendo un crédito activo.
resultado_final = df_cluster[["Cliente", "Codigo_Cliente_Sbs", "cluster"]].copy()
resultado_final["cluster"] = resultado_final["cluster"].astype(int)
resultado_final.to_csv(
    OUTPUT_DIR / "resultado_clustering.csv", index=False, encoding="utf-8"
)
print(
    f"\n  → resultado_clustering.csv guardado "
    f"({len(resultado_final):,} operaciones)"
)

# ============================================================
# 12. MÉTRICAS FINALES
# ============================================================
labels_final_sample = model_final.labels_[idx_sample]
sil_final = float(silhouette_score(X_sample, labels_final_sample))
db_final = float(davies_bouldin_score(X_sample, labels_final_sample))
ch_final = float(calinski_harabasz_score(X_sample, labels_final_sample))

pd.DataFrame([{
    "algoritmo": "MiniBatchKMeans",
    "k": K_FINAL,
    "inercia": float(model_final.inertia_),
    "silhouette": sil_final,
    "davies_bouldin": db_final,
    "calinski_harabasz": ch_final,
    "muestra_metricas": len(idx_sample),
    "registros_entrenamiento": len(X_scaled),
    "batch_size": BATCH_SIZE,
    "n_init": N_INIT,
    "max_iter": MAX_ITER,
    "random_state": RANDOM_STATE,
}]).to_csv(
    OUTPUT_DIR / "metricas_modelo_final.csv", index=False, encoding="utf-8"
)

print("\n" + "=" * 72)
print("MÉTRICAS FINALES DEL MODELO")
print("=" * 72)
print("  Algoritmo         : MiniBatchKMeans")
print(f"  k seleccionado    : {K_FINAL}")
print(f"  Registros ajuste  : {len(X_scaled):,}")
print(f"  Muestra métricas  : {len(idx_sample):,}")
print(f"  Inercia (WCSS)    : {model_final.inertia_:,.2f}")
print(f"  Silhouette Score  : {sil_final:.4f} (mayor es mejor)")
print(f"  Davies-Bouldin    : {db_final:.4f} (menor es mejor)")
print(f"  Calinski-Harabasz : {ch_final:,.2f} (mayor es mejor)")
print("  → metricas_modelo_final.csv guardado")

print("\n" + "=" * 72)
print("CONTROLES FINALES")
print("=" * 72)
print(f"  Créditos procesados     : {len(df_cluster):,}")
print(f"  Clústeres generados     : {df_cluster['cluster'].nunique()}")
print(f"  Códigos SBS con RCC     : {codigos_con_rcc:,}")
print(f"  Cobertura código SBS    : {pct_codigos_rcc:.2f}%")
print(f"  Clientes únicos con RCC : {clientes_con_rcc:,}")
print(f"  Cobertura cliente       : {pct_clientes_rcc:.2f}%")
print(f"  Outputs                 : {OUTPUT_DIR.resolve()}")
print("=" * 72)
print("Preprocesamiento y clustering completados.")
