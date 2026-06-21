"""
02_preprocesamiento_clustering.py - Preprocesamiento y clustering
TFM: Segmentación de Clientes según Perfil de Riesgo
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Dataset: grf10_1124_cod.txt + grf10_1124_rcc.txt
Periodo: Noviembre 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

warnings.filterwarnings("ignore")

# --- Rutas ---
DATA_DIR   = Path(r"C:\ANGEL\UNIR\TFM 2026\Data")
OUTPUT_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026\Data\outputs\clustering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COD_FILE = DATA_DIR / "grf10_1124_cod.txt"
RCC_FILE = DATA_DIR / "grf10_1124_rcc.txt"

RANDOM_STATE = 42

# =============================================
# 1. CARGA DE DATOS
# =============================================
print("=" * 60)
print("ETAPA 2 — PREPROCESAMIENTO Y CLUSTERING")
print("=" * 60)

print("\nCargando datasets...")

NUMERIC_COLS_COD = [
    "DS_MORA", "NCUOTAS_VEN", "SABONO_PROM",
    "SDSBOLSO", "SACTUAL", "SVENCIDO", "SJUDICIAL",
    "SMORA", "SCUOTA", "STASA", "SPROVIS",
    "NRO_CUOTAS", "NCUOTAS_PAG",
]

df_cod = pd.read_csv(
    COD_FILE, sep=";", encoding="latin-1",
    low_memory=False, dtype=str,
)
for col in NUMERIC_COLS_COD:
    if col in df_cod.columns:
        df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

df_cod["CLIENTE"] = df_cod["CLIENTE"].astype(str).str.strip()

df_rcc = pd.read_csv(
    RCC_FILE, sep=";", encoding="latin-1",
    low_memory=False, dtype=str,
)
df_rcc["saldo"]        = pd.to_numeric(df_rcc["saldo"], errors="coerce")
df_rcc["CLASIF_EMP"]   = pd.to_numeric(df_rcc["CLASIF_EMP"], errors="coerce")
df_rcc["cod_cliente_sbs"] = df_rcc["cod_cliente_sbs"].astype(str).str.strip()

print(f"  COD: {df_cod.shape[0]:,} filas")
print(f"  RCC: {df_rcc.shape[0]:,} filas")

# =============================================
# 2. FEATURE ENGINEERING — RCC
# =============================================
print("\nFeature engineering RCC...")

# Agregar por cliente: peor calificación, deuda total, nro entidades
rcc_agg = (
    df_rcc
    .groupby("cod_cliente_sbs")
    .agg(
        clasif_rcc_max  = ("CLASIF_EMP", "max"),
        saldo_rcc_total = ("saldo", "sum"),
        n_entidades     = ("entidad_financiera", "nunique"),
    )
    .reset_index()
    .rename(columns={"cod_cliente_sbs": "CLIENTE"})
)

print(f"  Clientes con RCC: {len(rcc_agg):,}")

# =============================================
# 3. MERGE COD + RCC
# =============================================
print("\nMerge COD + RCC...")

df = df_cod.merge(rcc_agg, on="CLIENTE", how="left")

# Sin RCC → clasif_rcc_max = 0
df["clasif_rcc_max"]   = df["clasif_rcc_max"].fillna(0)
df["saldo_rcc_total"]  = df["saldo_rcc_total"].fillna(0)
df["n_entidades"]      = df["n_entidades"].fillna(0)

print(f"  Dataset integrado: {df.shape[0]:,} filas")
con_rcc = (df["saldo_rcc_total"] > 0).sum()
print(f"  Con deuda externa: {con_rcc:,} ({con_rcc/len(df)*100:.1f}%)")

# =============================================
# 4. CODIFICACIÓN DE CATEGÓRICAS
# =============================================
print("\nCodificando variables categóricas...")

# calf_sbs_nov24 → ordinal 1-5
map_calf = {
    "2. NORMAL"      : 1,
    "3. CPP"         : 2,
    "4. DEFICIENTE"  : 3,
    "5. DUDOSO"      : 4,
    "6. PERDIDA"     : 5,
    "1. NO DEFINIDO" : 1,
}
df["calf_sbs_nov24"] = (
    df["calf_sbs_nov24"]
    .astype(str).str.strip()
    .map(map_calf)
    .fillna(1)   # nulos → Normal (moda)
    .astype(int)
)

# BDSBOLSO → ordinal (solo para validación posterior, no entra al modelo)
map_bdsbolso = {"ACT": 1, "REF": 2, "JUD": 3, "CAS": 4}
df["BDSBOLSO_num"] = (
    df["BDSBOLSO"]
    .astype(str).str.strip()
    .map(map_bdsbolso)
    .fillna(1)
    .astype(int)
)

dist_bdsbolso = df["BDSBOLSO"].value_counts()
print(f"  Distribución BDSBOLSO:\n{dist_bdsbolso.to_string()}")

# =============================================
# 5. VARIABLES DE CLUSTERING
# =============================================
print("\nPreparando variables de clustering...")

# BDSBOLSO no entra al modelo: es la clasificación operativa del banco
# y sesgaría el clustering hacia categorías ya existentes.
# Se usa como validación externa (sección 10b).
CLUSTER_VARS = [
    "DS_MORA",
    "NCUOTAS_VEN",
    "SABONO_PROM",
    "SDSBOLSO",
    "SACTUAL",
    "calf_sbs_nov24",
    "clasif_rcc_max",
]

df_cluster = df[["CLIENTE", "BDSBOLSO", "BDSBOLSO_num"] + CLUSTER_VARS].copy()

# Nulos residuales → mediana
nulos_pre = df_cluster[CLUSTER_VARS].isnull().sum()
print(f"  Nulos por variable:\n{nulos_pre[nulos_pre > 0].to_string()}")

for col in CLUSTER_VARS:
    if df_cluster[col].isnull().sum() > 0:
        mediana = df_cluster[col].median()
        df_cluster[col] = df_cluster[col].fillna(mediana)
        print(f"  {col}: nulos imputados con mediana={mediana:.2f}")

# =============================================
# 6. WINSORIZACIÓN p99
# =============================================
VARS_WINSORIZACION = ["DS_MORA", "NCUOTAS_VEN", "SABONO_PROM", "SDSBOLSO", "SACTUAL"]

for col in VARS_WINSORIZACION:
    p99 = df_cluster[col].quantile(0.99)
    p01 = df_cluster[col].quantile(0.01)
    antes_max = df_cluster[col].max()
    df_cluster[col] = df_cluster[col].clip(lower=p01, upper=p99)
    print(f"  {col}: max {antes_max:,.0f} → {p99:,.0f}")

# =============================================
# 7. ESCALADO
# =============================================
print("\nEscalando variables...")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_cluster[CLUSTER_VARS])
X_scaled = pd.DataFrame(X_scaled, columns=CLUSTER_VARS)

print(f"  Matriz escalada: {X_scaled.shape}")

# =============================================
# 8. SELECCIÓN DE k (k=2 a 10)
# =============================================
print("\nEvaluando k=2 a 10...")
print("  (puede tomar unos minutos)")

K_RANGE = range(2, 11)
inercias        = []
silhouettes     = []
davies_bouldins = []
calinski_scores = []

# Muestra para silhouette (cálculo exacto es O(n²))
SAMPLE_SIZE = 30_000
np.random.seed(RANDOM_STATE)
idx_sample = np.random.choice(len(X_scaled), size=min(SAMPLE_SIZE, len(X_scaled)), replace=False)
X_sample   = X_scaled.iloc[idx_sample].values

for k in K_RANGE:
    print(f"  k={k}...", end=" ", flush=True)
    model = MiniBatchKMeans(
        n_clusters   = k,
        random_state = RANDOM_STATE,
        batch_size   = 10_000,
        n_init       = 10,
    )
    model.fit(X_scaled)
    labels_full   = model.labels_
    labels_sample = labels_full[idx_sample]

    inercia = model.inertia_
    sil     = silhouette_score(X_sample, labels_sample, random_state=RANDOM_STATE)
    db      = davies_bouldin_score(X_sample, labels_sample)
    ch      = calinski_harabasz_score(X_sample, labels_sample)

    inercias.append(inercia)
    silhouettes.append(sil)
    davies_bouldins.append(db)
    calinski_scores.append(ch)
    print(f"sil={sil:.4f} | DB={db:.4f}")

metricas_df = pd.DataFrame({
    "k"                  : list(K_RANGE),
    "inercia"            : inercias,
    "silhouette"         : silhouettes,
    "davies_bouldin"     : davies_bouldins,
    "calinski_harabasz"  : calinski_scores,
})
metricas_df.to_csv(OUTPUT_DIR / "metricas_clustering.csv", index=False)
print(f"\n{metricas_df.to_string(index=False)}")

# Gráfica: codo + silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inercias, marker="o", color="#2563EB", linewidth=2)
axes[0].set_title("Método del Codo — Inercia por k", fontsize=12)
axes[0].set_xlabel("Número de clústeres (k)")
axes[0].set_ylabel("Inercia (WCSS)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].set_xticks(list(K_RANGE))

axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="#059669", linewidth=2)
axes[1].set_title("Silhouette Score por k", fontsize=12)
axes[1].set_xlabel("Número de clústeres (k)")
axes[1].set_ylabel("Silhouette Score (mayor es mejor)")
axes[1].set_xticks(list(K_RANGE))

k_optimo_sil = list(K_RANGE)[np.argmax(silhouettes)]
axes[1].axvline(x=k_optimo_sil, color="red", linestyle="--", alpha=0.7,
                label=f"k óptimo = {k_optimo_sil}")
axes[1].legend()

fig.suptitle("Selección del Número Óptimo de Clústeres", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig06_seleccion_k.png", dpi=150)
plt.close()
print(f"\n  → fig06_seleccion_k.png")
print(f"  k con mayor silhouette: {k_optimo_sil}")

# =============================================
# 9. MODELO FINAL
# =============================================
print("\nEjecutando modelo final...")

# k=4: silhouette 0.947, caída abrupta en k=5 (0.677).
# k=2/k=3 tienen mejores métricas pero no dan granularidad operativa suficiente.
K_FINAL = 4

model_final = MiniBatchKMeans(
    n_clusters   = K_FINAL,
    random_state = RANDOM_STATE,
    batch_size   = 10_000,
    n_init       = 15,
    max_iter     = 300,
)
model_final.fit(X_scaled)
df_cluster["cluster"] = model_final.labels_

dist_clusters = df_cluster["cluster"].value_counts().sort_index()
print(f"\n  Distribución:\n{dist_clusters.to_string()}")

# =============================================
# 10. PERFILADO
# =============================================
print("\nPerfilando clústeres...")

vars_perfil = [
    "DS_MORA", "NCUOTAS_VEN", "SABONO_PROM",
    "SDSBOLSO", "SACTUAL", "calf_sbs_nov24",
    "clasif_rcc_max",
]
perfil = df_cluster.groupby("cluster")[vars_perfil].mean().round(2)
perfil["n_clientes"] = dist_clusters.values
perfil["pct_cartera"] = (dist_clusters.values / len(df_cluster) * 100).round(1)

print(f"\n{perfil.to_string()}")
perfil.to_csv(OUTPUT_DIR / "perfil_clusters.csv")

# --- Validación externa con BDSBOLSO ---
print("\nValidación con BDSBOLSO...")

estado_por_cluster = (
    df_cluster.groupby(["cluster", "BDSBOLSO"])
    .size()
    .unstack(fill_value=0)
)
estado_por_cluster["TOTAL"] = estado_por_cluster.sum(axis=1)
for col in ["ACT", "CAS", "JUD", "REF"]:
    if col in estado_por_cluster.columns:
        estado_por_cluster[f"pct_{col}"] = (
            estado_por_cluster[col] / estado_por_cluster["TOTAL"] * 100
        ).round(1)

print(f"\n{estado_por_cluster.to_string()}")
estado_por_cluster.to_csv(OUTPUT_DIR / "estado_credito_por_cluster.csv")

# Pureza por clúster
print("\nPureza (% estado dominante):")
cols_estado = [c for c in ["ACT","CAS","JUD","REF"] if c in estado_por_cluster.columns]
for idx in estado_por_cluster.index:
    fila = estado_por_cluster.loc[idx, cols_estado]
    estado_dom = fila.idxmax()
    pct_dom = (fila.max() / estado_por_cluster.loc[idx, "TOTAL"] * 100)
    print(f"  Clúster {idx}: {estado_dom} ({pct_dom:.1f}%)")

# --- Heatmap de perfiles ---
fig, ax = plt.subplots(figsize=(11, 5))
perfil_norm = perfil[vars_perfil].copy()
for col in perfil_norm.columns:
    rng = perfil_norm[col].max() - perfil_norm[col].min()
    if rng > 0:
        perfil_norm[col] = (perfil_norm[col] - perfil_norm[col].min()) / rng

sns.heatmap(
    perfil_norm.T, annot=perfil[vars_perfil].T.round(1),
    fmt="g", cmap="YlOrRd", ax=ax,
    linewidths=0.5, cbar_kws={"shrink": 0.8}
)
ax.set_title(f"Perfil de Clústeres — MiniBatchKMeans (k={K_FINAL})", fontsize=12, pad=12)
ax.set_xlabel("Clúster")
ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig07_perfil_clusters_heatmap.png", dpi=150)
plt.close()
print("  → fig07_perfil_clusters_heatmap.png")

# --- Distribución clientes por clúster ---
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2563EB", "#059669", "#DC2626", "#F59E0B"]
bars = ax.bar(
    [f"Clúster {i}" for i in dist_clusters.index],
    dist_clusters.values,
    color=colors[:K_FINAL], edgecolor="white"
)
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
ax.set_title(f"Distribución de Clientes por Clúster (k={K_FINAL})", fontsize=12)
ax.set_ylabel("Número de clientes")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig08_distribucion_clusters.png", dpi=150)
plt.close()
print("  → fig08_distribucion_clusters.png")

# =============================================
# 11. EXPORTAR RESULTADO
# =============================================
resultado_final = df_cluster[["CLIENTE", "cluster"]].copy()
resultado_final["cluster"] = resultado_final["cluster"].astype(int)
resultado_final.to_csv(OUTPUT_DIR / "resultado_clustering.csv", index=False)
print(f"\n  → resultado_clustering.csv ({len(resultado_final):,} registros)")

# =============================================
# MÉTRICAS FINALES
# =============================================
sil_final = silhouette_score(X_sample, model_final.labels_[idx_sample],
                              random_state=RANDOM_STATE)
db_final  = davies_bouldin_score(X_sample, model_final.labels_[idx_sample])
ch_final  = calinski_harabasz_score(X_sample, model_final.labels_[idx_sample])

print("\n" + "=" * 60)
print(f"  Algoritmo        : MiniBatchKMeans (k={K_FINAL})")
print(f"  Silhouette       : {sil_final:.4f}")
print(f"  Davies-Bouldin   : {db_final:.4f}")
print(f"  Calinski-Harabasz: {ch_final:,.2f}")
print(f"  Inercia          : {model_final.inertia_:,.2f}")
print(f"\n  Outputs en: {OUTPUT_DIR.resolve()}")
print("=" * 60)
