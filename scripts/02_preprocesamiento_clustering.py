"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 2: Preprocesamiento y Clustering
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Dataset: cartera_creditos + reporte_crediticio_rcc
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

# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN DE RUTAS
# ─────────────────────────────────────────────
DATA_DIR   = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data")
OUTPUT_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data\outputs\clustering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COD_FILE = DATA_DIR / "cartera_creditos"
RCC_FILE = DATA_DIR / "reporte_crediticio_rcc"

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
print("=" * 60)
print("ETAPA 2 — PREPROCESAMIENTO Y CLUSTERING")
print("=" * 60)

print("\n[1/8] Cargando datasets...")

NUMERIC_COLS_COD = [
    "Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio",
    "Saldo_Desembolsado", "Saldo_Vigente", "Capital_Vencido", "Capital_Judicial",
    "Saldo_Mora", "Monto_Cuota", "Tasa_Interes", "Saldo_Provision",
    "Numero_Cuotas", "Cuotas_Pagadas",
]

df_cod = pd.read_csv(
    COD_FILE, sep=";", encoding="latin-1",
    low_memory=False, dtype=str,
)
for col in NUMERIC_COLS_COD:
    if col in df_cod.columns:
        df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

df_cod["Cliente"] = df_cod["Cliente"].astype(str).str.strip()

df_rcc = pd.read_csv(
    RCC_FILE, sep=";", encoding="latin-1",
    low_memory=False, dtype=str,
)
df_rcc["Saldo"]               = pd.to_numeric(df_rcc["Saldo"], errors="coerce")
df_rcc["Calificacion_Entidad"] = pd.to_numeric(df_rcc["Calificacion_Entidad"], errors="coerce")
df_rcc["Codigo_Cliente_Sbs"]   = df_rcc["Codigo_Cliente_Sbs"].astype(str).str.strip()

print(f"  COD cargado: {df_cod.shape[0]:,} filas")
print(f"  RCC cargado: {df_rcc.shape[0]:,} filas")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING — RCC
#    Peor_Calificacion_Rcc: peor calificación que
#    cualquier entidad externa le asignó al cliente
# ─────────────────────────────────────────────
print("\n[2/8] Feature engineering RCC...")

rcc_agg = (
    df_rcc
    .groupby("Codigo_Cliente_Sbs")
    .agg(
        Peor_Calificacion_Rcc = ("Calificacion_Entidad", "max"),
        Saldo_Rcc_Total       = ("Saldo", "sum"),
        N_Entidades           = ("Entidad_Financiera", "nunique"),
    )
    .reset_index()
    .rename(columns={"Codigo_Cliente_Sbs": "Cliente"})
)

print(f"  Clientes con RCC: {len(rcc_agg):,}")

# ─────────────────────────────────────────────
# 3. MERGE COD + RCC
# ─────────────────────────────────────────────
print("\n[3/8] Merge COD + RCC...")

df = df_cod.merge(rcc_agg, on="Cliente", how="left")

df["Peor_Calificacion_Rcc"] = df["Peor_Calificacion_Rcc"].fillna(0)
df["Saldo_Rcc_Total"]       = df["Saldo_Rcc_Total"].fillna(0)
df["N_Entidades"]           = df["N_Entidades"].fillna(0)

print(f"  Dataset integrado: {df.shape[0]:,} filas")
con_rcc = (df["Saldo_Rcc_Total"] > 0).sum()
print(f"  Clientes con deuda externa: {con_rcc:,} ({con_rcc/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ─────────────────────────────────────────────
print("\n[4/8] Codificando variables categóricas...")

# --- Calificacion_Sbs → ordinal de riesgo ---
map_calf = {
    "2. NORMAL"      : 1,
    "3. CPP"         : 2,
    "4. DEFICIENTE"  : 3,
    "5. DUDOSO"      : 4,
    "6. PERDIDA"     : 5,
    "1. NO DEFINIDO" : 1,
}
df["Calificacion_Sbs"] = (
    df["Calificacion_Sbs"]
    .astype(str).str.strip()
    .map(map_calf)
    .fillna(1)
    .astype(int)
)

# --- Estado_Credito → ordinal de estado del crédito ---
map_estado = {
    "ACT": 1,  # Activo
    "REF": 2,  # Refinanciado
    "JUD": 3,  # Judicial
    "CAS": 4,  # Castigado
}
df["Estado_Credito_Num"] = (
    df["Estado_Credito"]
    .astype(str).str.strip()
    .map(map_estado)
    .fillna(1)
    .astype(int)
)

dist_estado = df["Estado_Credito"].value_counts()
print(f"  Distribución Estado_Credito:\n{dist_estado.to_string()}")

# ─────────────────────────────────────────────
# 5. SELECCIÓN Y LIMPIEZA DE VARIABLES DE CLUSTERING
# ─────────────────────────────────────────────
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

# Estado_Credito se conserva solo para validación posterior
df_cluster = df[["Cliente", "Estado_Credito", "Estado_Credito_Num"] + CLUSTER_VARS].copy()

# Verificar nulos antes de winsorización
nulos_pre = df_cluster[CLUSTER_VARS].isnull().sum()
print(f"  Nulos por variable:\n{nulos_pre[nulos_pre > 0].to_string()}")

# Imputar nulos residuales con la mediana
for col in CLUSTER_VARS:
    if df_cluster[col].isnull().sum() > 0:
        mediana = df_cluster[col].median()
        df_cluster[col] = df_cluster[col].fillna(mediana)
        print(f"  Imputados nulos en {col} con mediana={mediana:.2f}")

# ─────────────────────────────────────────────
# 6. WINSORIZACIÓN AL PERCENTIL 99
# ─────────────────────────────────────────────
VARS_WINSORIZACION = ["Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio", "Saldo_Desembolsado", "Saldo_Vigente"]

for col in VARS_WINSORIZACION:
    p99 = df_cluster[col].quantile(0.99)
    p01 = df_cluster[col].quantile(0.01)
    antes_max = df_cluster[col].max()
    df_cluster[col] = df_cluster[col].clip(lower=p01, upper=p99)
    print(f"  Winsorización {col}: max {antes_max:,.0f} → {p99:,.0f}")

# ─────────────────────────────────────────────
# 7. ESCALADO — RobustScaler
# ─────────────────────────────────────────────
print("\n[6/8] Escalando variables...")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_cluster[CLUSTER_VARS])
X_scaled = pd.DataFrame(X_scaled, columns=CLUSTER_VARS)

print(f"  Shape matriz escalada: {X_scaled.shape}")
print(f"  Media post-escala (debe ser ~0):\n{X_scaled.mean().round(3).to_string()}")

# ─────────────────────────────────────────────
# 8. DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLÚSTERES
# ─────────────────────────────────────────────
print("\n[7/8] Determinando número óptimo de clústeres (k=2 a 10)...")
print("  (esto puede tomar 2-4 minutos con 500k registros)")

K_RANGE = range(2, 11)
inercias        = []
silhouettes     = []
davies_bouldins = []
calinski_scores = []

SAMPLE_SIZE = 30_000
np.random.seed(RANDOM_STATE)
idx_sample = np.random.choice(len(X_scaled), size=min(SAMPLE_SIZE, len(X_scaled)), replace=False)
X_sample   = X_scaled.iloc[idx_sample].values

for k in K_RANGE:
    print(f"  Probando k={k}...", end=" ", flush=True)
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
    print(f"inercia={inercia:,.0f} | silhouette={sil:.4f} | DB={db:.4f}")

metricas_df = pd.DataFrame({
    "k"                  : list(K_RANGE),
    "inercia"            : inercias,
    "silhouette"         : silhouettes,
    "davies_bouldin"     : davies_bouldins,
    "calinski_harabasz"  : calinski_scores,
})
metricas_df.to_csv(OUTPUT_DIR / "metricas_clustering.csv", index=False)
print(f"\n{metricas_df.to_string(index=False)}")

# --- Gráfica: Método del codo + Silhouette ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inercias, marker="o", color="#2563EB", linewidth=2)
axes[0].set_title("Método del codo — Inercia por k", fontsize=12)
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

fig.suptitle("Selección del número óptimo de clústeres", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig06_seleccion_k.png", dpi=150)
plt.close()
print(f"\n  → fig06_seleccion_k.png guardado")
print(f"  → k con mayor silhouette: k={k_optimo_sil}")

# ─────────────────────────────────────────────
# 9. MODELO FINAL — MiniBatchKMeans
# ─────────────────────────────────────────────
print("\n[8/8] Ejecutando modelo final MiniBatchKMeans...")

K_FINAL = 4

print(f"  K_FINAL seleccionado: {K_FINAL}")

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
print(f"\n  Distribución por clúster:\n{dist_clusters.to_string()}")

# ─────────────────────────────────────────────
# 10. PERFILADO DE CLÚSTERES
# ─────────────────────────────────────────────
print("\n  Perfilando clústeres...")

vars_perfil = [
    "Dias_Mora", "Cuotas_Vencidas", "Abono_Promedio",
    "Saldo_Desembolsado", "Saldo_Vigente", "Calificacion_Sbs",
    "Peor_Calificacion_Rcc",
]
perfil = df_cluster.groupby("cluster")[vars_perfil].mean().round(2)
perfil["n_clientes"] = dist_clusters.values
perfil["pct_cartera"] = (dist_clusters.values / len(df_cluster) * 100).round(1)

print(f"\n  Perfil de clústeres:\n{perfil.to_string()}")
perfil.to_csv(OUTPUT_DIR / "perfil_clusters.csv")

# ─────────────────────────────────────────────
# 10b. VALIDACIÓN EXTERNA CON ESTADO_CREDITO
# ─────────────────────────────────────────────
print("\n  Validación externa con Estado_Credito...")

estado_por_cluster = (
    df_cluster.groupby(["cluster", "Estado_Credito"])
    .size()
    .unstack(fill_value=0)
)
estado_por_cluster["TOTAL"] = estado_por_cluster.sum(axis=1)
for col in ["ACT", "CAS", "JUD", "REF"]:
    if col in estado_por_cluster.columns:
        estado_por_cluster[f"pct_{col}"] = (
            estado_por_cluster[col] / estado_por_cluster["TOTAL"] * 100
        ).round(1)

print(f"\n  Estado crédito por clúster (validación):\n{estado_por_cluster.to_string()}")
estado_por_cluster.to_csv(OUTPUT_DIR / "estado_credito_por_cluster.csv")

print("\n  Pureza por clúster (% estado dominante):")
cols_estado = [c for c in ["ACT","CAS","JUD","REF"] if c in estado_por_cluster.columns]
for idx in estado_por_cluster.index:
    fila = estado_por_cluster.loc[idx, cols_estado]
    estado_dom = fila.idxmax()
    pct_dom = (fila.max() / estado_por_cluster.loc[idx, "TOTAL"] * 100)
    print(f"  Clúster {idx}: estado dominante = {estado_dom} ({pct_dom:.1f}%)")

# --- Gráfica: heatmap de perfiles ---
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
ax.set_title(f"Perfil de clústeres — MiniBatchKMeans (k={K_FINAL})", fontsize=12, pad=12)
ax.set_xlabel("Clúster")
ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig07_perfil_clusters_heatmap.png", dpi=150)
plt.close()
print("  → fig07_perfil_clusters_heatmap.png guardado")

# --- Gráfica: distribución clientes por clúster ---
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2563EB", "#059669", "#DC2626", "#F59E0B", "#7C3AED",
          "#0891B2", "#65A30D", "#DB2777", "#EA580C", "#6366F1"]
bars = ax.bar(
    [f"Clúster {i}" for i in dist_clusters.index],
    dist_clusters.values,
    color=colors[:K_FINAL], edgecolor="white"
)
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
ax.set_title(f"Distribución de clientes por clúster (k={K_FINAL})", fontsize=12)
ax.set_ylabel("Número de clientes")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig08_distribucion_clusters.png", dpi=150)
plt.close()
print("  → fig08_distribucion_clusters.png guardado")

# ─────────────────────────────────────────────
# 11. EXPORTAR RESULTADO FINAL
# ─────────────────────────────────────────────
resultado_final = df_cluster[["Cliente", "cluster"]].copy()
resultado_final["cluster"] = resultado_final["cluster"].astype(int)
resultado_final.to_csv(OUTPUT_DIR / "resultado_clustering.csv", index=False)
print(f"\n  → resultado_clustering.csv guardado ({len(resultado_final):,} registros)")

# ─────────────────────────────────────────────
# 12. MÉTRICAS FINALES DEL MODELO
# ─────────────────────────────────────────────
sil_final = silhouette_score(X_sample, model_final.labels_[idx_sample],
                              random_state=RANDOM_STATE)
db_final  = davies_bouldin_score(X_sample, model_final.labels_[idx_sample])
ch_final  = calinski_harabasz_score(X_sample, model_final.labels_[idx_sample])

print("\n" + "=" * 60)
print("MÉTRICAS FINALES DEL MODELO")
print("=" * 60)
print(f"  Algoritmo        : MiniBatchKMeans")
print(f"  k seleccionado   : {K_FINAL}")
print(f"  Inercia (WCSS)   : {model_final.inertia_:,.2f}")
print(f"  Silhouette Score : {sil_final:.4f}  (rango -1 a 1, mayor es mejor)")
print(f"  Davies-Bouldin   : {db_final:.4f}  (menor es mejor)")
print(f"  Calinski-Harabasz: {ch_final:,.2f}  (mayor es mejor)")
print(f"\n  Outputs guardados en: {OUTPUT_DIR.resolve()}")
print("=" * 60)
print("  Preprocesamiento y clustering completados.")
