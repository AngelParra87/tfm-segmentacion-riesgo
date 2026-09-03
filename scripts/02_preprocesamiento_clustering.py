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

Al final del script se regeneran las figuras definitivas utilizadas en la
memoria y se documenta visualmente la comparación complementaria con DBSCAN
a partir de los resultados validados del experimento descrito en la sección 5.4.3.
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
print("Preprocesamiento y clustering completados. Generando figuras finales...")


# ============================================================
# 13. FIGURAS FINALES Y COMPARACIÓN DOCUMENTADA CON DBSCAN
# ============================================================
# Durante el desarrollo estas visualizaciones se generaron en un script auxiliar
# de rediseño. Se integran aquí para que el repositorio final conserve únicamente
# tres scripts principales.
#
# Importante: las Figuras 14 y 15 utilizan los resultados validados del
# experimento DBSCAN descrito en la memoria (muestra = 50.000, semilla = 42,
# winsorización P99 y StandardScaler). Este bloque regenera sus visualizaciones;
# no vuelve a ejecutar la búsqueda completa de hiperparámetros DBSCAN.

METRICAS_FILE = OUTPUT_DIR / "metricas_clustering.csv"
PERFIL_FILE = OUTPUT_DIR / "perfil_clusters.csv"

MAP_CLUSTER = {
    0: "Cartera en Riesgo",
    1: "Cartera Vigente",
    2: "Cartera Castigada",
    3: "Cartera Judicial",
}

ORDEN_SEGMENTOS = [
    "Cartera Vigente",
    "Cartera en Riesgo",
    "Cartera Judicial",
    "Cartera Castigada",
]

COLORES_SEGMENTOS = {
    "Cartera Vigente": "#2E7D32",
    "Cartera en Riesgo": "#F59E0B",
    "Cartera Judicial": "#C62828",
    "Cartera Castigada": "#4B5563",
}

COLOR_TEXTO = "#1F2937"
COLOR_NEUTRO = "#94A3B8"
COLOR_ACENTO = "#2563EB"
COLOR_REJILLA = "#E5E7EB"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": COLOR_TEXTO,
        "axes.titlecolor": COLOR_TEXTO,
        "xtick.color": COLOR_TEXTO,
        "ytick.color": COLOR_TEXTO,
        "text.color": COLOR_TEXTO,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
    }
)


def verificar_archivo(ruta: Path) -> None:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecute primero 02_preprocesamiento_clustering.py."
        )


def limpiar_ejes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")

    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=COLOR_REJILLA,
            linewidth=0.8,
            alpha=0.8,
        )
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    fig.savefig(
        OUTPUT_DIR / nombre,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  -> {nombre}")


def nombre_variable(var: str) -> str:
    return {
        "Dias_Mora": "Días de mora",
        "Cuotas_Vencidas": "Cuotas vencidas",
        "Abono_Promedio": "Abono promedio",
        "Saldo_Desembolsado": "Saldo desembolsado",
        "Saldo_Vigente": "Saldo vigente",
        "Calificacion_Sbs": "Calificación SBS",
        "Peor_Calificacion_Rcc": "Peor calificación RCC",
    }.get(var, var.replace("_", " "))


def formatear_valor(var: str, valor: float) -> str:
    if pd.isna(valor):
        return "—"

    if var == "Dias_Mora":
        return f"{valor:,.1f}"
    if var == "Cuotas_Vencidas":
        return f"{valor:,.2f}"
    if var in {"Abono_Promedio", "Saldo_Desembolsado", "Saldo_Vigente"}:
        return f"{valor:,.0f}"
    if var in {"Calificacion_Sbs", "Peor_Calificacion_Rcc"}:
        return f"{valor:,.2f}"
    return f"{valor:,.2f}"


for archivo in [METRICAS_FILE, PERFIL_FILE]:
    verificar_archivo(archivo)

metricas = pd.read_csv(METRICAS_FILE)
perfil = pd.read_csv(PERFIL_FILE)

if "cluster" not in perfil.columns:
    # Cuando el CSV se guardó con el índice, pandas suele recuperarlo como
    # primera columna sin nombre.
    primera = perfil.columns[0]
    perfil = perfil.rename(columns={primera: "cluster"})

perfil["cluster"] = pd.to_numeric(perfil["cluster"], errors="raise").astype(int)
perfil["segmento"] = perfil["cluster"].map(MAP_CLUSTER)

if perfil["segmento"].isna().any():
    raise ValueError("Se encontraron identificadores de clúster no contemplados en MAP_CLUSTER.")


# ============================================================
# FIGURA 06 — EVOLUCIÓN DE MÉTRICAS PARA SELECCIÓN DE k
# ============================================================

columnas_requeridas = [
    "k",
    "inercia",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]
faltantes = [c for c in columnas_requeridas if c not in metricas.columns]
if faltantes:
    raise KeyError("Faltan columnas en metricas_clustering.csv: " + ", ".join(faltantes))

metricas = metricas.sort_values("k").reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
axes = axes.ravel()

series = [
    ("inercia", "Inercia", "Menor indica mayor compactación", "{:,.0f}"),
    ("silhouette", "Índice de silueta", "Mayor es mejor", "{:.4f}"),
    ("davies_bouldin", "Davies-Bouldin", "Menor es mejor", "{:.4f}"),
    ("calinski_harabasz", "Calinski-Harabasz", "Mayor es mejor", "{:,.0f}"),
]

for ax, (col, titulo, subtitulo, formato) in zip(axes, series):
    ax.plot(
        metricas["k"],
        metricas[col],
        color=COLOR_NEUTRO,
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=COLOR_NEUTRO,
    )

    seleccionado = metricas.loc[metricas["k"] == K_FINAL]
    if not seleccionado.empty:
        x = float(seleccionado["k"].iloc[0])
        y = float(seleccionado[col].iloc[0])

        ax.scatter(
            [x],
            [y],
            s=90,
            color=COLOR_ACENTO,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.annotate(
            f"k = {K_FINAL}\n{formato.format(y)}",
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=COLOR_ACENTO,
            ha="left",
            va="bottom",
        )

    ax.set_title(f"{titulo}\n{subtitulo}", loc="left", fontsize=10.5, pad=10)
    ax.set_xlabel("Número de clústeres (k)")
    ax.set_xticks(metricas["k"].astype(int).tolist())
    limpiar_ejes(ax, grid_axis="y")

axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
)
axes[3].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

fig.suptitle(
    "k = 4 equilibra calidad interna y mayor granularidad de segmentación",
    x=0.06,
    ha="left",
    fontsize=13,
    fontweight="semibold",
)
fig.text(
    0.06,
    0.955,
    "El punto azul identifica la configuración seleccionada; no representa el óptimo individual de todas las métricas.",
    ha="left",
    va="top",
    fontsize=9,
    color="#475569",
)

fig.tight_layout(rect=[0, 0, 1, 0.92])
guardar_figura(fig, "fig06_seleccion_k.png")


# ============================================================
# FIGURA 07 — PERFIL NORMALIZADO POR SEGMENTO
# ============================================================

vars_perfil = [
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Calificacion_Sbs",
    "Peor_Calificacion_Rcc",
]
vars_perfil = [v for v in vars_perfil if v in perfil.columns]

perfil_orden = (
    perfil.set_index("segmento")
    .reindex(ORDEN_SEGMENTOS)
)

valores = perfil_orden[vars_perfil].T.astype(float)

normalizado = valores.copy()
for idx in normalizado.index:
    minimo = normalizado.loc[idx].min()
    maximo = normalizado.loc[idx].max()
    rango = maximo - minimo
    if rango > 0:
        normalizado.loc[idx] = (normalizado.loc[idx] - minimo) / rango
    else:
        normalizado.loc[idx] = 0.0

anotaciones = pd.DataFrame(
    index=vars_perfil,
    columns=ORDEN_SEGMENTOS,
    dtype=object,
)
for var in vars_perfil:
    for segmento in ORDEN_SEGMENTOS:
        anotaciones.loc[var, segmento] = formatear_valor(
            var,
            float(valores.loc[var, segmento]),
        )

normalizado.index = [nombre_variable(v) for v in normalizado.index]
anotaciones.index = [nombre_variable(v) for v in anotaciones.index]

fig, ax = plt.subplots(figsize=(10.8, 6.4))

sns.heatmap(
    normalizado,
    annot=anotaciones,
    fmt="",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={
        "shrink": 0.75,
        "label": "Magnitud relativa dentro de cada variable (0–1)",
    },
    annot_kws={"fontsize": 8.5},
    ax=ax,
)

ax.set_title(
    "Los cuatro segmentos presentan perfiles financieros claramente diferenciados",
    loc="left",
    pad=12,
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

fig.text(
    0.125,
    0.025,
    "El color compara magnitudes relativas dentro de cada variable; las cifras muestran los valores promedio originales.",
    fontsize=8.5,
    color="#475569",
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
guardar_figura(fig, "fig07_perfil_clusters_heatmap.png")


# ============================================================
# FIGURA 08 — DISTRIBUCIÓN DE CRÉDITOS ACTIVOS POR SEGMENTO
# ============================================================

if "n_creditos" not in perfil.columns:
    raise KeyError("perfil_clusters.csv no contiene la columna n_creditos.")

dist = (
    perfil[["segmento", "n_creditos"]]
    .copy()
    .set_index("segmento")
    .reindex(ORDEN_SEGMENTOS)
    .reset_index()
)

dist["n_creditos"] = pd.to_numeric(dist["n_creditos"], errors="raise")
dist["pct"] = dist["n_creditos"] / dist["n_creditos"].sum() * 100

fig, ax = plt.subplots(figsize=(10.2, 5.2))

bars = ax.barh(
    dist["segmento"][::-1],
    dist["pct"][::-1],
    color=[COLORES_SEGMENTOS[s] for s in dist["segmento"][::-1]],
    edgecolor="none",
)

max_pct = max(dist["pct"].max(), 1)
ax.set_xlim(0, min(100, max_pct * 1.09))

for bar, pct, n in zip(
    bars,
    dist["pct"][::-1],
    dist["n_creditos"][::-1],
):
    ax.text(
        min(bar.get_width() + 0.7, 99.2),
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.2f} %  ·  {int(n):,}",
        va="center",
        ha="left",
        fontsize=9,
    )

vigente = dist.loc[dist["segmento"] == "Cartera Vigente", "pct"]
if not vigente.empty:
    titulo = f"Cartera Vigente concentra el {float(vigente.iloc[0]):.2f} % de los créditos activos"
else:
    titulo = "Distribución de créditos activos por segmento"

ax.set_title(titulo, loc="left", pad=12)
ax.set_xlabel("Participación en la cartera (%)")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f} %")
)
limpiar_ejes(ax, grid_axis="x")

fig.tight_layout()
guardar_figura(fig, "fig08_distribucion_clusters.png")


# ============================================================
# FIGURA 14 — DISTRIBUCIÓN DE LA SOLUCIÓN DBSCAN
# ============================================================
# Valores documentados en la comparación metodológica del TFM
# para eps = 1.0, min_samples = 50 y muestra = 50,000.
# Se utilizan porcentajes consolidados para comunicar el desbalance
# sin reconstruir el experimento DBSCAN en este script de visualización.

dbscan_dist = pd.DataFrame(
    {
        "grupo": [
            "Clúster principal",
            "Clúster secundario",
            "Ruido",
        ],
        "pct": [97.5, 1.4, 1.1],
    }
)

fig, ax = plt.subplots(figsize=(9.8, 4.8))

colores_dbscan = [
    COLOR_ACENTO,
    COLOR_NEUTRO,
    "#9CA3AF",
]

bars = ax.barh(
    dbscan_dist["grupo"][::-1],
    dbscan_dist["pct"][::-1],
    color=colores_dbscan[::-1],
    edgecolor="none",
)

ax.set_xlim(0, 100)

for bar, pct in zip(bars, dbscan_dist["pct"][::-1]):
    ax.text(
        min(bar.get_width() + 1.0, 99.0),
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f} %",
        va="center",
        ha="left",
        fontsize=9.5,
    )

ax.set_title(
    "DBSCAN concentra el 97.5 % de la muestra en un único clúster",
    loc="left",
    pad=12,
)
ax.set_xlabel("Participación en la muestra (%)")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f} %")
)
limpiar_ejes(ax, grid_axis="x")

fig.text(
    0.125,
    0.015,
    "Configuración: eps = 1.0, min_samples = 50; muestra = 50,000 registros.",
    fontsize=8.4,
    color="#475569",
)

fig.tight_layout(rect=[0, 0.045, 1, 1])
guardar_figura(fig, "fig_dbscan_distribucion.png")


# ============================================================
# FIGURA 15 — COMPARACIÓN DE MÉTRICAS INTERNAS
# ============================================================
# Comparación controlada sobre la misma muestra de 50,000 registros
# y el mismo preprocesamiento. Se muestran los valores originales
# en paneles separados para evitar mezclar escalas incompatibles.

comparacion = {
    "Índice de silueta": {
        "MiniBatchKMeans": 0.479,
        "DBSCAN": 0.735,
        "mejor": "DBSCAN",
        "criterio": "Mayor es mejor",
        "formato": "{:.3f}",
    },
    "Davies-Bouldin": {
        "MiniBatchKMeans": 0.823,
        "DBSCAN": 0.423,
        "mejor": "DBSCAN",
        "criterio": "Menor es mejor",
        "formato": "{:.3f}",
    },
    "Calinski-Harabasz": {
        "MiniBatchKMeans": 30_642,
        "DBSCAN": 12_213,
        "mejor": "MiniBatchKMeans",
        "criterio": "Mayor es mejor",
        "formato": "{:,.0f}",
    },
}

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))

for ax, (metrica, datos) in zip(axes, comparacion.items()):
    algoritmos = ["MiniBatchKMeans", "DBSCAN"]
    valores = [datos[a] for a in algoritmos]

    colores = [
        COLOR_ACENTO if a == datos["mejor"] else COLOR_NEUTRO
        for a in algoritmos
    ]

    bars = ax.barh(
        algoritmos[::-1],
        valores[::-1],
        color=colores[::-1],
        edgecolor="none",
    )

    max_val = max(valores)
    ax.set_xlim(0, max_val * 1.28 if max_val > 0 else 1)

    for bar, algoritmo, valor in zip(
        bars,
        algoritmos[::-1],
        valores[::-1],
    ):
        ax.text(
            bar.get_width() + max_val * 0.025,
            bar.get_y() + bar.get_height() / 2,
            datos["formato"].format(valor),
            va="center",
            ha="left",
            fontsize=9,
            color=COLOR_TEXTO,
        )

    ax.set_title(
        f"{metrica}\n{datos['criterio']}",
        loc="left",
        fontsize=10.5,
        pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    limpiar_ejes(ax, grid_axis="x")

    if metrica == "Calinski-Harabasz":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
        )

fig.suptitle(
    "DBSCAN lidera dos métricas internas; MiniBatchKMeans obtiene mayor Calinski-Harabasz",
    x=0.045,
    ha="left",
    fontsize=13,
    fontweight="semibold",
)

fig.text(
    0.045,
    0.02,
    "Comparación sobre la misma muestra de 50,000 registros y el mismo preprocesamiento. "
    "El color destaca el mejor resultado de cada métrica.",
    fontsize=8.4,
    color="#475569",
)

fig.tight_layout(rect=[0, 0.055, 1, 0.90])
guardar_figura(fig, "fig_comparacion_metricas.png")


print("\nFiguras finales de clustering y comparación DBSCAN completadas.")
print(f"Outputs: {OUTPUT_DIR.resolve()}")
