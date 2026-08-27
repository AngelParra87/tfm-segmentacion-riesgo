"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 1: carga de datos y análisis exploratorio
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Periodo: noviembre de 2024

Este script realiza los controles iniciales de calidad, estadísticas descriptivas
y visualizaciones utilizadas durante el análisis exploratorio de la cartera y
del Reporte Crediticio Consolidado (RCC).

La unidad de análisis de cartera_creditos es el crédito activo. Por ese motivo,
los conteos de filas se presentan como créditos u operaciones y no como clientes
únicos. Algunas variables incluidas en esta etapa fueron evaluadas como
candidatas y posteriormente descartadas antes del modelo final.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

DATA_DIR = Path(r"C:\ANGEL\UNIR\TFM 2026 - v2\Data")
OUTPUT_DIR = DATA_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 597_127


def resolver_archivo(data_dir: Path, nombre_base: str) -> Path:
    """
    Busca primero el archivo .txt y, si no existe, una versión sin extensión.
    Esto permite conservar compatibilidad con las rutas utilizadas en distintas
    etapas del proyecto.
    """
    candidatos = [
        data_dir / f"{nombre_base}.txt",
        data_dir / nombre_base,
    ]

    for ruta in candidatos:
        if ruta.exists():
            return ruta

    raise FileNotFoundError(
        f"No se encontró '{nombre_base}' en {data_dir}. "
        "Se esperaba un archivo .txt o sin extensión."
    )


COD_FILE = resolver_archivo(DATA_DIR, "cartera_creditos")
RCC_FILE = resolver_archivo(DATA_DIR, "reporte_crediticio_rcc")


# Variables numéricas revisadas en el EDA y utilizadas en las figuras de
# diagnóstico. Capital_Vencido y Capital_Judicial fueron evaluadas en esta
# etapa, pero no forman parte del modelo final.
EDA_VARS_NUMERICAS = [
    "Dias_Mora",
    "Capital_Vencido",
    "Cuotas_Vencidas",
    "Capital_Judicial",
    "Abono_Promedio",
    "Saldo_Desembolsado",
]

# Seis variables procedentes de la cartera que finalmente se utilizan en el
# modelo. Peor_Calificacion_Rcc se incorpora posteriormente en el script 02.
VARS_CARTERA_MODELO = [
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Calificacion_Sbs",
]


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

print("=" * 72)
print("ETAPA 1 — CARGA Y ANÁLISIS EXPLORATORIO")
print("=" * 72)

print("\n[1/6] Cargando datasets...")
print(f"  Cartera: {COD_FILE}")
print(f"  RCC    : {RCC_FILE}")

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

print(
    f"  cartera_creditos       : "
    f"{df_cod.shape[0]:,} filas x {df_cod.shape[1]} columnas"
)
print(
    f"  reporte_crediticio_rcc : "
    f"{df_rcc.shape[0]:,} filas x {df_rcc.shape[1]} columnas"
)

if len(df_cod) != EXPECTED_ROWS:
    print(
        f"  [ADVERTENCIA] El corte contiene {len(df_cod):,} operaciones; "
        f"el corte de noviembre de 2024 utilizado en el TFM contiene "
        f"{EXPECTED_ROWS:,}."
    )


# ============================================================
# 2. CONVERSIÓN DE TIPOS
# ============================================================

print("\n[2/6] Convirtiendo columnas numéricas...")

NUMERIC_COLS = [
    "Dias_Mora",
    "Capital_Vencido",
    "Cuotas_Vencidas",
    "Capital_Judicial",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Capital_Vigente",
    "Saldo_Mora",
    "Saldo_Provision",
    "Monto_Cuota",
    "Tasa_Interes",
    "Numero_Dias",
    "Numero_Cuotas",
    "Cuotas_Pagadas",
    "Cuotas_Por_Pagar",
    "Cuotas_Pendientes",
    "Saldo_Amortizacion_Vencida",
    "Saldo_Interes_Vencido",
    "Abono_Mes1",
    "Abono_Mes2",
    "Abono_Mes3",
    "Abono_Mes4",
    "Abono_Mes5",
    "Abono_Mes6",
]

for col in NUMERIC_COLS:
    if col in df_cod.columns:
        df_cod[col] = pd.to_numeric(df_cod[col], errors="coerce")

for col in ["Saldo", "Condicion_Dias", "Calificacion_Entidad"]:
    if col in df_rcc.columns:
        df_rcc[col] = pd.to_numeric(df_rcc[col], errors="coerce")

# Las claves se normalizan sin convertir los nulos al texto "nan".
if "Cliente" in df_cod.columns:
    df_cod["Cliente"] = (
        df_cod["Cliente"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

if "Codigo_Cliente_Sbs" in df_cod.columns:
    df_cod["Codigo_Cliente_Sbs"] = (
        df_cod["Codigo_Cliente_Sbs"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

if "Codigo_Cliente_Sbs" in df_rcc.columns:
    df_rcc["Codigo_Cliente_Sbs"] = (
        df_rcc["Codigo_Cliente_Sbs"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


# ============================================================
# 3. CALIDAD DE DATOS
# ============================================================

print("\n[3/6] Analizando calidad de datos...")


def reporte_calidad(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    total = len(df)

    reporte = pd.DataFrame(
        {
            "columna": df.columns,
            "tipo": df.dtypes.astype(str).values,
            "nulos": df.isna().sum().values,
            "pct_nulos": (df.isna().sum().values / total * 100).round(4),
            "unicos": df.nunique(dropna=True).values,
        }
    )

    reporte = reporte.sort_values(
        ["pct_nulos", "nulos"],
        ascending=False,
    ).reset_index(drop=True)

    print(f"\n  Dataset: {nombre} ({total:,} registros)")
    con_nulos = reporte[reporte["nulos"] > 0]

    if con_nulos.empty:
        print("  No se detectaron valores nulos.")
    else:
        print(con_nulos.to_string(index=False))

    return reporte


rep_cod = reporte_calidad(df_cod, "cartera_creditos")
rep_rcc = reporte_calidad(df_rcc, "reporte_crediticio_rcc")

rep_cod.to_csv(
    OUTPUT_DIR / "calidad_cod.csv",
    index=False,
    encoding="utf-8",
)
rep_rcc.to_csv(
    OUTPUT_DIR / "calidad_rcc.csv",
    index=False,
    encoding="utf-8",
)


# ============================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

print("\n[4/6] Calculando estadísticas descriptivas...")

vars_eda_disponibles = [
    v for v in EDA_VARS_NUMERICAS
    if v in df_cod.columns
]

if vars_eda_disponibles:
    desc = (
        df_cod[vars_eda_disponibles]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
        .T
    )

    # Evita divisiones indeterminadas en variables con media igual a cero.
    desc["coef_variacion"] = np.where(
        desc["mean"].ne(0),
        desc["std"] / desc["mean"],
        np.nan,
    )

    print("\n  Variables revisadas durante el EDA:")
    print(desc.round(2).to_string())

    desc.to_csv(
        OUTPUT_DIR / "estadisticas_descriptivas.csv",
        encoding="utf-8",
    )

vars_modelo_numericas = [
    v for v in VARS_CARTERA_MODELO
    if v in df_cod.columns and v != "Calificacion_Sbs"
]

if vars_modelo_numericas:
    desc_modelo = (
        df_cod[vars_modelo_numericas]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .T
    )

    desc_modelo.to_csv(
        OUTPUT_DIR / "estadisticas_vars_cartera_modelo.csv",
        encoding="utf-8",
    )


# ============================================================
# 5. DISTRIBUCIÓN DE CALIFICACIÓN SBS
# ============================================================

print("\n[5/6] Revisando distribución de Calificacion_Sbs...")

calif_counts = None

if "Calificacion_Sbs" in df_cod.columns:
    calificacion = (
        df_cod["Calificacion_Sbs"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    df_cod["Calificacion_Sbs"] = calificacion

    calif_counts = (
        calificacion
        .value_counts(dropna=False)
        .rename_axis("calificacion")
        .reset_index(name="cantidad")
    )

    calif_counts["calificacion"] = (
        calif_counts["calificacion"]
        .astype("string")
        .fillna("Sin calificación")
    )

    calif_counts["pct"] = (
        calif_counts["cantidad"] / len(df_cod) * 100
    ).round(2)

    print(calif_counts.to_string(index=False))

    calif_counts.to_csv(
        OUTPUT_DIR / "dist_calificacion_sbs.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# 6. VISUALIZACIONES
# ============================================================

print("\n[6/6] Generando figuras...")

plt.style.use("seaborn-v0_8-whitegrid")

# Los colores se mantienen estables para reproducir las figuras del TFM.
COLOR_AZUL = "#2563EB"
COLOR_VERDE = "#10B981"
COLOR_MORADO = "#7C3AED"


# ------------------------------------------------------------
# 6.1 Distribución de calificación SBS
# ------------------------------------------------------------

if calif_counts is not None:
    calif_plot = calif_counts.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        calif_plot["calificacion"],
        calif_plot["cantidad"],
        color=COLOR_AZUL,
        edgecolor="white",
    )

    ax.bar_label(
        bars,
        fmt="{:,.0f}",
        padding=3,
        fontsize=9,
    )

    ax.set_title(
        "Distribución de créditos activos por calificación SBS",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Calificación SBS")
    ax.set_ylabel("Número de créditos activos")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig01_dist_calificacion_sbs.png",
        dpi=150,
    )
    plt.close()

    print("  -> fig01_dist_calificacion_sbs.png")


# ------------------------------------------------------------
# 6.2 Distribución de días de mora
# ------------------------------------------------------------

if "Dias_Mora" in df_cod.columns:
    data_mora = df_cod["Dias_Mora"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        data_mora,
        bins=60,
        color=COLOR_AZUL,
        edgecolor="white",
        alpha=0.85,
    )
    axes[0].set_title("Días de mora — distribución completa")
    axes[0].set_xlabel("Días de mora")
    axes[0].set_ylabel("Número de créditos activos")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    data_mora_zoom = data_mora[data_mora <= 365]

    axes[1].hist(
        data_mora_zoom,
        bins=60,
        color=COLOR_VERDE,
        edgecolor="white",
        alpha=0.85,
    )
    axes[1].set_title("Días de mora — zoom de 0 a 365 días")
    axes[1].set_xlabel("Días de mora")
    axes[1].set_ylabel("Número de créditos activos")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    fig.suptitle(
        "Distribución de los días de mora",
        fontsize=13,
        y=1.01,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig02_dist_ds_mora.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("  -> fig02_dist_ds_mora.png")


# ------------------------------------------------------------
# 6.3 Variables financieras revisadas durante el EDA
# ------------------------------------------------------------

vars_box = [
    v
    for v in [
        "Capital_Vencido",
        "Saldo_Desembolsado",
        "Abono_Promedio",
        "Capital_Judicial",
    ]
    if v in df_cod.columns
]

if vars_box:
    fig, axes = plt.subplots(
        1,
        len(vars_box),
        figsize=(14, 5),
    )

    if len(vars_box) == 1:
        axes = [axes]

    colores = [
        "#2563EB",
        "#7C3AED",
        "#059669",
        "#DC2626",
    ]

    for ax, var, color in zip(axes, vars_box, colores):
        data_var = df_cod[var].dropna()

        if data_var.empty:
            continue

        p99 = data_var.quantile(0.99)
        data_plot = data_var[data_var <= p99]

        ax.boxplot(
            data_plot,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.5),
            medianprops=dict(color="black", linewidth=2),
        )

        ax.set_title(var, fontsize=10)
        ax.set_ylabel("Soles (S/.)")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        ax.set_xticks([])

    fig.suptitle(
        "Distribución de variables financieras revisadas (percentil <= 99)",
        fontsize=12,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig03_boxplots_vars_financieras.png",
        dpi=150,
    )
    plt.close()

    print("  -> fig03_boxplots_vars_financieras.png")


# ------------------------------------------------------------
# 6.4 Correlación entre variables exploratorias
# ------------------------------------------------------------

vars_corr = [
    v for v in EDA_VARS_NUMERICAS
    if v in df_cod.columns
]

if len(vars_corr) >= 2:
    corr_matrix = df_cod[vars_corr].corr()

    fig, ax = plt.subplots(figsize=(9, 7))

    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        "Matriz de correlación — variables revisadas en el EDA",
        fontsize=12,
        pad=12,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig04_correlacion_vars_clustering.png",
        dpi=150,
    )
    plt.close()

    print("  -> fig04_correlacion_vars_clustering.png")


# ------------------------------------------------------------
# 6.5 RCC: saldo por tipo de crédito
# ------------------------------------------------------------

if "Tipo_Credito" in df_rcc.columns and "Saldo" in df_rcc.columns:
    rcc_tipo = (
        df_rcc
        .groupby("Tipo_Credito", dropna=False)["Saldo"]
        .agg(["count", "sum", "mean"])
    )

    rcc_tipo.columns = [
        "n_registros",
        "saldo_total",
        "saldo_promedio",
    ]

    rcc_tipo = (
        rcc_tipo
        .sort_values("saldo_total", ascending=False)
        .head(10)
    )

    rcc_tipo.to_csv(
        OUTPUT_DIR / "rcc_por_tipo_credito.csv",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        rcc_tipo.index.astype(str)[::-1],
        rcc_tipo["saldo_total"][::-1],
        color=COLOR_MORADO,
        alpha=0.8,
    )

    ax.set_title(
        "RCC — saldo total por tipo de crédito (Top 10)",
        fontsize=12,
    )
    ax.set_xlabel("Saldo total (S/.)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig05_rcc_saldo_tipo_credito.png",
        dpi=150,
    )
    plt.close()

    print("  -> fig05_rcc_saldo_tipo_credito.png")


# ============================================================
# 7. RESUMEN
# ============================================================

print("\n" + "=" * 72)
print("RESUMEN EDA")
print("=" * 72)

print(f"  Créditos activos en cartera : {len(df_cod):,}")
print(f"  Registros RCC               : {len(df_rcc):,}")

if "Cliente" in df_cod.columns:
    clientes_unicos = df_cod["Cliente"].nunique(dropna=True)
    print(f"  Clientes únicos             : {clientes_unicos:,}")

if "Calificacion_Sbs" in df_cod.columns:
    riesgo_alto = (
        df_cod["Calificacion_Sbs"]
        .isin(
            [
                "4. DEFICIENTE",
                "5. DUDOSO",
                "6. PERDIDA",
            ]
        )
        .sum()
    )

    pct_riesgo = riesgo_alto / len(df_cod) * 100

    print(
        f"  Créditos con calificación deteriorada "
        f"(Def./Dud./Pérd.): {riesgo_alto:,} "
        f"({pct_riesgo:.1f}%)"
    )

if "Dias_Mora" in df_cod.columns:
    con_mora = int((df_cod["Dias_Mora"] > 0).sum())

    print(
        f"  Créditos con mora > 0 días  : "
        f"{con_mora:,} "
        f"({con_mora / len(df_cod) * 100:.1f}%)"
    )

if "Capital_Judicial" in df_cod.columns:
    judicial = int((df_cod["Capital_Judicial"] > 0).sum())

    print(
        f"  Créditos con Capital_Judicial > 0: "
        f"{judicial:,} "
        f"({judicial / len(df_cod) * 100:.1f}%)"
    )

print(f"\n  Outputs: {OUTPUT_DIR.resolve()}")
print("=" * 72)
print("EDA completado.")
