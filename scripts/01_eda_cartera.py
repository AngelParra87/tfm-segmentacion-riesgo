"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 1: Carga de datos y Análisis Exploratorio (EDA)
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Dataset: cartera_creditos (cartera propia) + reporte_crediticio_rcc (RCC-SBS)
Periodo: Noviembre 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN DE RUTAS
# ─────────────────────────────────────────────
DATA_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data")
OUTPUT_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data\outputs\eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COD_FILE = DATA_DIR / "cartera_creditos"
RCC_FILE = DATA_DIR / "reporte_crediticio_rcc"

# Variables clave para clustering (definidas en el TFM)
CLUSTER_VARS = [
    "Dias_Mora",
    "Capital_Vencido",
    "Cuotas_Vencidas",
    "Capital_Judicial",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Calificacion_Sbs",
]

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
print("=" * 60)
print("ETAPA 1 — CARGA Y EDA")
print("=" * 60)

print("\n[1/6] Cargando datasets...")

df_cod = pd.read_csv(
    COD_FILE,
    sep=";",
    encoding="latin-1",
    low_memory=False,
    dtype=str,
)

df_rcc = pd.read_csv(
    RCC_FILE,
    sep=";",
    encoding="latin-1",
    low_memory=False,
    dtype=str,
)

print(f"  cartera_creditos       → {df_cod.shape[0]:,} filas × {df_cod.shape[1]} columnas")
print(f"  reporte_crediticio_rcc → {df_rcc.shape[0]:,} filas × {df_rcc.shape[1]} columnas")

# ─────────────────────────────────────────────
# 2. TIPOS Y CONVERSIÓN DE COLUMNAS NUMÉRICAS
# ─────────────────────────────────────────────
print("\n[2/6] Convirtiendo columnas numéricas...")

NUMERIC_COLS = [
    "Dias_Mora", "Capital_Vencido", "Cuotas_Vencidas", "Capital_Judicial",
    "Abono_Promedio", "Saldo_Desembolsado", "Saldo_Vigente", "Capital_Vigente",
    "Saldo_Mora", "Saldo_Provision", "Monto_Cuota", "Tasa_Interes", "Numero_Dias",
    "Numero_Cuotas", "Cuotas_Pagadas", "Cuotas_Por_Pagar",
    "Saldo_Amortizacion_Vencida", "Saldo_Interes_Vencido",
    "Abono_Mes1", "Abono_Mes2", "Abono_Mes3",
    "Abono_Mes4", "Abono_Mes5", "Abono_Mes6",
]

for col in NUMERIC_COLS:
    if col in df_cod.columns:
        df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

# RCC numéricos
for col in ["Saldo", "Condicion_Dias"]:
    if col in df_rcc.columns:
        df_rcc[col] = pd.to_numeric(df_rcc[col], errors="coerce")

# Identificador del cliente
df_cod["Cliente"] = df_cod["Cliente"].astype(str).str.strip()
df_rcc["Codigo_Cliente_Sbs"] = df_rcc["Codigo_Cliente_Sbs"].astype(str).str.strip()

# ─────────────────────────────────────────────
# 3. REPORTE DE CALIDAD DE DATOS
# ─────────────────────────────────────────────
print("\n[3/6] Análisis de calidad de datos...")

def reporte_calidad(df, nombre):
    total = len(df)
    reporte = pd.DataFrame({
        "columna": df.columns,
        "tipo": df.dtypes.values,
        "nulos": df.isnull().sum().values,
        "pct_nulos": (df.isnull().sum().values / total * 100).round(2),
        "unicos": df.nunique().values,
    })
    reporte = reporte.sort_values("pct_nulos", ascending=False)
    print(f"\n  Dataset: {nombre} ({total:,} registros)")
    print(reporte[reporte["pct_nulos"] > 0].to_string(index=False))
    return reporte

rep_cod = reporte_calidad(df_cod, "cartera_creditos")
rep_rcc = reporte_calidad(df_rcc, "reporte_crediticio_rcc")

rep_cod.to_csv(OUTPUT_DIR / "calidad_cod.csv", index=False)
rep_rcc.to_csv(OUTPUT_DIR / "calidad_rcc.csv", index=False)

# ─────────────────────────────────────────────
# 4. ESTADÍSTICAS DESCRIPTIVAS — VARIABLES CLAVE
# ─────────────────────────────────────────────
print("\n[4/6] Estadísticas descriptivas de variables clave...")

vars_numericas = [v for v in CLUSTER_VARS if v in df_cod.columns and v != "Calificacion_Sbs"]
desc = df_cod[vars_numericas].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]).T
desc["coef_variacion"] = (desc["std"] / desc["mean"]).round(2)
print(desc.round(2).to_string())
desc.to_csv(OUTPUT_DIR / "estadisticas_descriptivas.csv")

# ─────────────────────────────────────────────
# 5. DISTRIBUCIÓN: CALIFICACIÓN SBS
# ─────────────────────────────────────────────
print("\n[5/6] Distribución de calificación SBS...")

if "Calificacion_Sbs" in df_cod.columns:
    df_cod["Calificacion_Sbs"] = df_cod["Calificacion_Sbs"].astype(str).str.strip()
    calif_counts = df_cod["Calificacion_Sbs"].value_counts().reset_index()
    calif_counts.columns = ["calificacion", "cantidad"]
    calif_counts["pct"] = (calif_counts["cantidad"] / len(df_cod) * 100).round(1)
    print(calif_counts.to_string(index=False))
    calif_counts.to_csv(OUTPUT_DIR / "dist_calificacion_sbs.csv", index=False)

# ─────────────────────────────────────────────
# 6. VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n[6/6] Generando gráficas...")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "#2563EB"

# --- 6a. Distribución calificación SBS (barras) ---
if "Calificacion_Sbs" in df_cod.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    calif_plot = df_cod["Calificacion_Sbs"].value_counts().sort_index()
    bars = ax.bar(calif_plot.index, calif_plot.values, color=PALETTE, edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
    ax.set_title("Distribución de clientes por calificación SBS", fontsize=12, pad=12)
    ax.set_xlabel("Calificación SBS")
    ax.set_ylabel("Número de clientes")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig01_dist_calificacion_sbs.png", dpi=150)
    plt.close()
    print("  → fig01_dist_calificacion_sbs.png")

# --- 6b. Distribución días de mora ---
if "Dias_Mora" in df_cod.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data_mora = df_cod["Dias_Mora"].dropna()
    axes[0].hist(data_mora, bins=60, color=PALETTE, edgecolor="white", alpha=0.85)
    axes[0].set_title("Días de mora — distribución completa")
    axes[0].set_xlabel("Días de mora")
    axes[0].set_ylabel("Frecuencia")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    data_mora_zoom = data_mora[data_mora <= 365]
    axes[1].hist(data_mora_zoom, bins=60, color="#10B981", edgecolor="white", alpha=0.85)
    axes[1].set_title("Días de mora — zoom 0 a 365 días")
    axes[1].set_xlabel("Días de mora")
    axes[1].set_ylabel("Frecuencia")

    fig.suptitle("Distribución de los días de mora", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig02_dist_ds_mora.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig02_dist_ds_mora.png")

# --- 6c. Boxplots variables financieras clave ---
vars_box = [v for v in ["Capital_Vencido", "Saldo_Desembolsado", "Abono_Promedio", "Capital_Judicial"] if v in df_cod.columns]
if vars_box:
    fig, axes = plt.subplots(1, len(vars_box), figsize=(14, 5))
    if len(vars_box) == 1:
        axes = [axes]
    colors = ["#2563EB", "#7C3AED", "#059669", "#DC2626"]
    for ax, var, color in zip(axes, vars_box, colors):
        data_var = df_cod[var].dropna()
        p99 = data_var.quantile(0.99)
        ax.boxplot(data_var[data_var <= p99], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   medianprops=dict(color="black", linewidth=2))
        ax.set_title(var, fontsize=10)
        ax.set_ylabel("Soles (S/.)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xticks([])
    fig.suptitle("Distribución de variables financieras clave (percentil ≤99)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig03_boxplots_vars_financieras.png", dpi=150)
    plt.close()
    print("  → fig03_boxplots_vars_financieras.png")

# --- 6d. Mapa de correlación entre variables de clustering ---
vars_corr = [v for v in vars_numericas if v in df_cod.columns]
if len(vars_corr) >= 2:
    corr_matrix = df_cod[vars_corr].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Matriz de correlación — Variables de clustering", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig04_correlacion_vars_clustering.png", dpi=150)
    plt.close()
    print("  → fig04_correlacion_vars_clustering.png")

# --- 6e. RCC: saldo externo por tipo de crédito ---
if "Tipo_Credito" in df_rcc.columns and "Saldo" in df_rcc.columns:
    rcc_tipo = df_rcc.groupby("Tipo_Credito")["Saldo"].agg(["count", "sum", "mean"])
    rcc_tipo.columns = ["n_registros", "saldo_total", "saldo_promedio"]
    rcc_tipo = rcc_tipo.sort_values("saldo_total", ascending=False).head(10)
    rcc_tipo.to_csv(OUTPUT_DIR / "rcc_por_tipo_credito.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(rcc_tipo.index[::-1], rcc_tipo["saldo_total"][::-1], color="#7C3AED", alpha=0.8)
    ax.set_title("RCC — Saldo total por tipo de crédito (Top 10)", fontsize=12)
    ax.set_xlabel("Saldo total (S/.)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig05_rcc_saldo_tipo_credito.png", dpi=150)
    plt.close()
    print("  → fig05_rcc_saldo_tipo_credito.png")

# ─────────────────────────────────────────────
# 7. RESUMEN EJECUTIVO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN EDA")
print("=" * 60)
print(f"  Total clientes en cartera propia : {len(df_cod):,}")
print(f"  Total registros RCC              : {len(df_rcc):,}")

if "Calificacion_Sbs" in df_cod.columns:
    riesgo_alto = df_cod["Calificacion_Sbs"].astype(str).str.strip().isin(
        ["4. DEFICIENTE", "5. DUDOSO", "6. PERDIDA"]
    ).sum()
    pct_riesgo = riesgo_alto / len(df_cod) * 100
    print(f"  Clientes riesgo alto (Def/Dud/Pérd): {riesgo_alto:,} ({pct_riesgo:.1f}%)")

if "Dias_Mora" in df_cod.columns:
    con_mora = (df_cod["Dias_Mora"] > 0).sum()
    print(f"  Clientes con mora > 0 días       : {con_mora:,} ({con_mora/len(df_cod)*100:.1f}%)")

if "Capital_Judicial" in df_cod.columns:
    judicial = (df_cod["Capital_Judicial"] > 0).sum()
    print(f"  Clientes con capital judicial > 0: {judicial:,} ({judicial/len(df_cod)*100:.1f}%)")

print(f"\n  Outputs guardados en: {OUTPUT_DIR.resolve()}")
print("=" * 60)
print("  EDA completado.")
