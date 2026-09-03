"""
TFM - Segmentación de Clientes con Créditos Activos según Perfil de Riesgo
Etapa 1: carga de datos y análisis exploratorio
Autores: Lourdes Flores Mamani / Angel Parra Florecin
Periodo: noviembre de 2024

Este script realiza los controles de calidad, estadísticas descriptivas y
visualizaciones utilizadas durante el análisis exploratorio de la cartera y
del Reporte Crediticio Consolidado (RCC). También incorpora la caracterización
complementaria por tipo de préstamo, unidad ejecutora, sexo y año de apertura.

La unidad de análisis de cartera_creditos es el crédito activo. Por ese motivo,
los conteos de filas se presentan como créditos u operaciones y no como clientes
únicos. Algunas variables incluidas en esta etapa fueron evaluadas como
candidatas y posteriormente descartadas antes del modelo final.

Revisión visual:
- Figuras orientadas a hallazgos y no solo a la descripción de variables.
- Menor ruido visual y uso restringido del color como elemento de énfasis.
- Etiquetas directas cuando aportan claridad.
- Unidades expresadas de forma legible.
- Exportación a 300 dpi para uso en la memoria.
"""

from pathlib import Path
import re
import textwrap
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

print("\n[6/6] Generando figuras revisadas...")

# Paleta sobria. El color de acento se reserva para el hallazgo principal.
COLOR_TEXTO = "#1F2937"
COLOR_NEUTRO = "#CBD5E1"
COLOR_BASE = "#64748B"
COLOR_ACENTO = "#2563EB"
COLOR_ACENTO_OSCURO = "#1D4ED8"
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


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda una figura con resolución adecuada para la memoria."""
    fig.savefig(
        OUTPUT_DIR / nombre,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  -> {nombre}")


def limpiar_ejes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Reduce elementos decorativos y deja solo ayudas de lectura útiles."""
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


def etiqueta_calificacion(valor: object) -> str:
    """Convierte etiquetas técnicas de Calificacion_Sbs a texto legible."""
    if pd.isna(valor):
        return "Sin calificación"

    texto = str(valor).strip()
    if texto.lower() in {"nan", "<na>", "none", ""}:
        return "Sin calificación"

    texto = re.sub(r"^\s*\d+\.\s*", "", texto)
    clave = texto.upper()

    mapeo = {
        "NO DEFINIDO": "No definido",
        "NORMAL": "Normal",
        "CPP": "CPP",
        "DEFICIENTE": "Deficiente",
        "DUDOSO": "Dudoso",
        "PERDIDA": "Pérdida",
        "PÉRDIDA": "Pérdida",
        "SIN CALIFICACION": "Sin calificación",
        "SIN CALIFICACIÓN": "Sin calificación",
    }

    return mapeo.get(clave, texto)


def etiqueta_variable(var: str) -> str:
    """Nombres legibles para variables usadas en las figuras."""
    mapeo = {
        "Dias_Mora": "Días de mora",
        "Capital_Vencido": "Capital vencido",
        "Cuotas_Vencidas": "Cuotas vencidas",
        "Capital_Judicial": "Capital judicial",
        "Abono_Promedio": "Abono promedio",
        "Saldo_Desembolsado": "Saldo desembolsado",
        "Saldo_Vigente": "Saldo vigente",
    }
    return mapeo.get(var, var.replace("_", " "))


# ------------------------------------------------------------
# 6.1 Distribución de calificación SBS
# ------------------------------------------------------------

if calif_counts is not None:
    calif_plot = calif_counts.copy()
    calif_plot["etiqueta"] = calif_plot["calificacion"].map(etiqueta_calificacion)

    # Reagrupa por si varias etiquetas originales terminan en la misma categoría.
    calif_plot = (
        calif_plot
        .groupby("etiqueta", as_index=False)["cantidad"]
        .sum()
    )
    calif_plot["pct"] = calif_plot["cantidad"] / len(df_cod) * 100

    orden = [
        "Normal",
        "CPP",
        "Deficiente",
        "Dudoso",
        "Pérdida",
        "No definido",
        "Sin calificación",
    ]
    extras = [x for x in calif_plot["etiqueta"] if x not in orden]
    orden_final = orden + extras
    orden_map = {valor: i for i, valor in enumerate(orden_final)}

    calif_plot["orden"] = calif_plot["etiqueta"].map(orden_map)
    calif_plot = calif_plot.sort_values("orden")

    # barh dibuja la primera categoría abajo; se invierte para conservar el orden.
    plot_df = calif_plot.iloc[::-1].copy()

    colores = [
        COLOR_ACENTO if etiqueta == "Normal" else COLOR_NEUTRO
        for etiqueta in plot_df["etiqueta"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.barh(
        plot_df["etiqueta"],
        plot_df["cantidad"],
        color=colores,
        edgecolor="none",
    )

    max_val = max(plot_df["cantidad"].max(), 1)
    ax.set_xlim(0, max_val * 1.25)

    for bar, cantidad, pct in zip(
        bars,
        plot_df["cantidad"],
        plot_df["pct"],
    ):
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{cantidad:,.0f} ({pct:.1f} %)",
            va="center",
            ha="left",
            fontsize=9,
            color=COLOR_TEXTO,
        )

    normal_row = calif_plot[calif_plot["etiqueta"] == "Normal"]
    if not normal_row.empty:
        pct_normal = float(normal_row["pct"].iloc[0])
        titulo = f"La calificación Normal concentra el {pct_normal:.1f} % de los créditos activos"
    else:
        titulo = "Distribución de créditos activos por calificación SBS"

    ax.set_title(titulo, loc="left", pad=12)
    ax.set_xlabel("Número de créditos activos")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    limpiar_ejes(ax, grid_axis="x")

    fig.tight_layout()
    guardar_figura(fig, "fig01_dist_calificacion_sbs.png")


# ------------------------------------------------------------
# 6.2 Distribución de días de mora
# ------------------------------------------------------------

if "Dias_Mora" in df_cod.columns:
    data_mora = df_cod["Dias_Mora"].dropna()
    data_mora = data_mora[data_mora >= 0]

    n_total_mora = len(data_mora)
    n_sin_mora = int((data_mora == 0).sum())
    n_con_mora = int((data_mora > 0).sum())

    pct_sin_mora = n_sin_mora / n_total_mora * 100 if n_total_mora else 0
    pct_con_mora = n_con_mora / n_total_mora * 100 if n_total_mora else 0

    resumen_mora = pd.DataFrame(
        {
            "situacion": ["Sin mora (0 días)", "Con mora (>0 días)"],
            "cantidad": [n_sin_mora, n_con_mora],
            "pct_total": [pct_sin_mora, pct_con_mora],
        }
    )
    resumen_mora.to_csv(
        OUTPUT_DIR / "dist_mora_resumen.csv",
        index=False,
        encoding="utf-8",
    )

    data_mora_pos = data_mora[data_mora > 0]
    bins = [0, 30, 60, 90, 180, 365, np.inf]
    labels = [
        "1–30 días",
        "31–60 días",
        "61–90 días",
        "91–180 días",
        "181–365 días",
        ">365 días",
    ]

    rangos = pd.cut(
        data_mora_pos,
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=False,
    )
    rango_counts = rangos.value_counts(sort=False).reindex(labels, fill_value=0)
    rango_pct = (
        rango_counts / len(data_mora_pos) * 100
        if len(data_mora_pos)
        else rango_counts.astype(float)
    )

    tabla_rangos = pd.DataFrame(
        {
            "rango_mora": labels,
            "cantidad": rango_counts.values,
            "pct_entre_creditos_con_mora": rango_pct.values,
        }
    )
    tabla_rangos.to_csv(
        OUTPUT_DIR / "dist_mora_rangos_positivos.csv",
        index=False,
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # Panel A: dimensión de la cartera sin mora y con mora.
    panel_a = resumen_mora.iloc[::-1]
    colores_a = [
        COLOR_ACENTO if situacion.startswith("Con mora") else COLOR_NEUTRO
        for situacion in panel_a["situacion"]
    ]
    bars_a = axes[0].barh(
        panel_a["situacion"],
        panel_a["cantidad"],
        color=colores_a,
        edgecolor="none",
    )
    max_a = max(panel_a["cantidad"].max(), 1)
    axes[0].set_xlim(0, max_a * 1.23)

    for bar, cantidad, pct in zip(
        bars_a,
        panel_a["cantidad"],
        panel_a["pct_total"],
    ):
        axes[0].text(
            bar.get_width() + max_a * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{cantidad:,.0f} ({pct:.1f} %)",
            va="center",
            ha="left",
            fontsize=9,
        )

    axes[0].set_title("Situación general de mora", loc="left")
    axes[0].set_xlabel("Número de créditos activos")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    limpiar_ejes(axes[0], grid_axis="x")

    # Panel B: distribución interna de los créditos con mora.
    panel_b = tabla_rangos.iloc[::-1]
    bars_b = axes[1].barh(
        panel_b["rango_mora"],
        panel_b["cantidad"],
        color=COLOR_BASE,
        edgecolor="none",
    )
    max_b = max(panel_b["cantidad"].max(), 1)
    axes[1].set_xlim(0, max_b * 1.28)

    for bar, cantidad, pct in zip(
        bars_b,
        panel_b["cantidad"],
        panel_b["pct_entre_creditos_con_mora"],
    ):
        axes[1].text(
            bar.get_width() + max_b * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{cantidad:,.0f} ({pct:.1f} %)",
            va="center",
            ha="left",
            fontsize=9,
        )

    axes[1].set_title("Distribución entre créditos con mora", loc="left")
    axes[1].set_xlabel("Número de créditos activos")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    limpiar_ejes(axes[1], grid_axis="x")

    fig.suptitle(
        f"El {pct_sin_mora:.1f} % de los créditos activos no registra mora",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=COLOR_TEXTO,
    )
    fig.text(
        0.99,
        0.01,
        "Panel derecho: porcentajes calculados sobre créditos con mora (>0 días).",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLOR_BASE,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    guardar_figura(fig, "fig02_dist_ds_mora.png")


# ------------------------------------------------------------
# 6.3 Diagnóstico de variables financieras revisadas durante el EDA
# ------------------------------------------------------------

vars_utiles = [
    v for v in ["Saldo_Desembolsado", "Abono_Promedio"]
    if v in df_cod.columns
]

if vars_utiles:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Panel A: variables con distribución informativa para el análisis.
    datos_box = []
    etiquetas_box = []
    for var in vars_utiles:
        serie = df_cod[var].dropna()
        if serie.empty:
            continue
        p99 = serie.quantile(0.99)
        datos_box.append(serie[serie <= p99])
        etiquetas_box.append(etiqueta_variable(var))

    if datos_box:
        bp = axes[0].boxplot(
            datos_box,
            vert=False,
            patch_artist=True,
            labels=etiquetas_box,
            showfliers=False,
            widths=0.5,
        )

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLOR_ACENTO if i == 0 else COLOR_BASE)
            patch.set_alpha(0.75)
            patch.set_edgecolor("none")

        for median in bp["medians"]:
            median.set_color(COLOR_TEXTO)
            median.set_linewidth(1.8)

        for elemento in bp["whiskers"] + bp["caps"]:
            elemento.set_color(COLOR_BASE)

        axes[0].set_title("Variables con variabilidad útil (hasta P99)", loc="left")
        axes[0].set_xlabel("Soles (S/)")
        axes[0].set_ylabel("")
        axes[0].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        limpiar_ejes(axes[0], grid_axis="x")

    # Panel B: diagnóstico de los campos problemáticos revisados en el EDA.
    ax = axes[1]
    hay_judicial = (
        "Capital_Judicial" in df_cod.columns
        and not df_cod["Capital_Judicial"].dropna().empty
    )

    if hay_judicial:
        serie_j = df_cod["Capital_Judicial"].dropna()
        valores_millones = np.array(
            [
                serie_j.median(),
                serie_j.quantile(0.95),
                serie_j.max(),
            ]
        ) / 1_000_000
        etiquetas_j = ["Mediana", "Percentil 95", "Máximo"]

        bars_j = ax.barh(
            etiquetas_j[::-1],
            valores_millones[::-1],
            color=[COLOR_BASE, COLOR_BASE, COLOR_ACENTO][::-1],
            edgecolor="none",
        )
        max_j = max(valores_millones.max(), 1)
        ax.set_xlim(0, max_j * 1.22)

        for bar, valor in zip(bars_j, valores_millones[::-1]):
            ax.text(
                bar.get_width() + max_j * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"S/ {valor:,.0f} M",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.set_title("Capital judicial: magnitudes atípicas", loc="left")
        ax.set_xlabel("Millones de soles")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        limpiar_ejes(ax, grid_axis="x")
    else:
        ax.axis("off")

    # Anotación de Capital_Vencido: evita dedicarle un boxplot plano.
    if "Capital_Vencido" in df_cod.columns:
        serie_v = df_cod["Capital_Vencido"].dropna()
        if not serie_v.empty:
            n_unicos = serie_v.nunique(dropna=True)
            if n_unicos == 1:
                valor_unico = float(serie_v.iloc[0])
                texto_cv = (
                    f"Capital_Vencido: un único valor observado "
                    f"(S/ {valor_unico:,.0f}); no aporta variabilidad."
                )
            else:
                pct_cero = (serie_v.eq(0).mean() * 100)
                texto_cv = (
                    f"Capital_Vencido: {n_unicos:,} valores distintos; "
                    f"{pct_cero:.1f} % de registros en cero."
                )

            ax.text(
                0.0,
                -0.18,
                textwrap.fill(texto_cv, width=70),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color=COLOR_TEXTO,
            )

    # Archivo auxiliar de trazabilidad del diagnóstico.
    diagnostico_rows = []
    if "Capital_Vencido" in df_cod.columns:
        s = df_cod["Capital_Vencido"].dropna()
        if not s.empty:
            diagnostico_rows.append(
                {
                    "variable": "Capital_Vencido",
                    "mediana": s.median(),
                    "percentil_95": s.quantile(0.95),
                    "maximo": s.max(),
                    "n_unicos": s.nunique(dropna=True),
                }
            )
    if "Capital_Judicial" in df_cod.columns:
        s = df_cod["Capital_Judicial"].dropna()
        if not s.empty:
            diagnostico_rows.append(
                {
                    "variable": "Capital_Judicial",
                    "mediana": s.median(),
                    "percentil_95": s.quantile(0.95),
                    "maximo": s.max(),
                    "n_unicos": s.nunique(dropna=True),
                }
            )

    if diagnostico_rows:
        pd.DataFrame(diagnostico_rows).to_csv(
            OUTPUT_DIR / "diagnostico_vars_financieras.csv",
            index=False,
            encoding="utf-8",
        )

    fig.suptitle(
        "El EDA distinguió variables con variabilidad útil de campos problemáticos",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=COLOR_TEXTO,
    )
    fig.text(
        0.99,
        0.01,
        "Los boxplots del panel izquierdo se limitan al percentil 99 para facilitar la lectura.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLOR_BASE,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    guardar_figura(fig, "fig03_boxplots_vars_financieras.png")


# ------------------------------------------------------------
# 6.4 Correlación entre variables exploratorias
# ------------------------------------------------------------

vars_corr_candidatas = [
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
    "Capital_Judicial",
]

vars_corr = [
    v for v in vars_corr_candidatas
    if v in df_cod.columns
    and df_cod[v].nunique(dropna=True) > 1
]

if len(vars_corr) >= 2:
    corr_matrix = df_cod[vars_corr].corr()

    # Identifica la relación absoluta más alta fuera de la diagonal para usarla
    # como mensaje principal de la figura.
    lower_mask = np.tril(np.ones(corr_matrix.shape, dtype=bool), k=-1)
    corr_lower_abs = corr_matrix.abs().where(lower_mask)
    stacked = corr_lower_abs.stack()

    if not stacked.empty:
        var_a, var_b = stacked.idxmax()
        corr_max = float(corr_matrix.loc[var_a, var_b])
        titulo_corr = (
            f"La mayor relación lineal se observa entre "
            f"{etiqueta_variable(var_a).lower()} y "
            f"{etiqueta_variable(var_b).lower()} (r = {corr_max:.2f})"
        )
    else:
        titulo_corr = "Correlación entre variables revisadas durante el EDA"

    corr_display = corr_matrix.rename(
        index={v: etiqueta_variable(v) for v in vars_corr},
        columns={v: etiqueta_variable(v) for v in vars_corr},
    )

    mask = np.triu(np.ones_like(corr_display, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    sns.heatmap(
        corr_display,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={
            "shrink": 0.78,
            "label": "Coeficiente de correlación",
        },
        annot_kws={"fontsize": 9},
    )

    ax.set_title(titulo_corr, loc="left", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    guardar_figura(fig, "fig04_correlacion_vars_clustering.png")


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

    rcc_tipo = rcc_tipo.sort_values("saldo_total", ascending=False)

    rcc_tipo.to_csv(
        OUTPUT_DIR / "rcc_por_tipo_credito.csv",
        encoding="utf-8",
    )

    plot_rcc = rcc_tipo.reset_index().copy()
    plot_rcc["tipo_credito_etiqueta"] = (
        plot_rcc["Tipo_Credito"]
        .astype("string")
        .fillna("Sin tipo de crédito")
        .str.replace("_", " ", regex=False)
        .str.strip()
        .str.lower()
        .str.capitalize()
        .map(lambda x: textwrap.fill(str(x), width=34))
    )
    plot_rcc["saldo_millones"] = plot_rcc["saldo_total"] / 1_000_000

    total_saldo_rcc = plot_rcc["saldo_total"].sum()
    pct_top = (
        plot_rcc.iloc[0]["saldo_total"] / total_saldo_rcc * 100
        if total_saldo_rcc
        else 0
    )
    top_label = str(plot_rcc.iloc[0]["tipo_credito_etiqueta"]).replace("\n", " ")

    panel = plot_rcc.iloc[::-1].copy()
    top_index = plot_rcc.index[0]
    colores = [
        COLOR_ACENTO if idx == top_index else COLOR_NEUTRO
        for idx in panel.index
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.barh(
        panel["tipo_credito_etiqueta"],
        panel["saldo_millones"],
        color=colores,
        edgecolor="none",
    )

    max_rcc = max(panel["saldo_millones"].max(), 1)
    ax.set_xlim(0, max_rcc * 1.20)

    for bar, valor in zip(bars, panel["saldo_millones"]):
        ax.text(
            bar.get_width() + max_rcc * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"S/ {valor:,.0f} M",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(
        f"{top_label} concentra el {pct_top:.1f} % del saldo del RCC",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Saldo total (millones de S/)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    limpiar_ejes(ax, grid_axis="x")

    fig.tight_layout()
    guardar_figura(fig, "fig05_rcc_saldo_tipo_credito.png")


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
print("EDA principal completado. Continuando con la caracterización ampliada...")


# ============================================================
# 8. CARACTERIZACIÓN AMPLIADA DE LA CARTERA
# ============================================================
# Este bloque integra en el script principal el análisis que durante el
# desarrollo se ejecutó de forma separada en 04_eda_ampliado_actualizado.py.
# Se mantienen las salidas en una subcarpeta distinta para conservar la
# trazabilidad de los archivos utilizados en la memoria.

OUTPUT_DIR_AMPLIADO = DATA_DIR / "outputs" / "eda_ampliado"
OUTPUT_DIR_AMPLIADO.mkdir(parents=True, exist_ok=True)


def limpiar_texto(serie: pd.Series, valor_nulo: str = "Sin identificar") -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna(valor_nulo)
    )


def abreviar_etiqueta(texto: str, ancho: int = 34) -> str:
    texto = re.sub(r"\s+", " ", str(texto)).strip()
    return textwrap.fill(texto, width=ancho)


def guardar_figura_ampliada(fig: plt.Figure, nombre: str) -> None:
    fig.savefig(
        OUTPUT_DIR_AMPLIADO / nombre,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  -> {nombre}")


print("\n" + "=" * 72)
print("CARACTERIZACIÓN AMPLIADA — TIPO DE PRÉSTAMO, UNIDAD Y AÑO")
print("=" * 72)

# Se vuelve a leer la cartera como texto para conservar exactamente la lógica
# de las variables categóricas y de fecha empleada en este bloque.
df_amp = pd.read_csv(
    COD_FILE,
    sep=";",
    encoding="latin-1",
    low_memory=False,
    dtype=str,
)

print(f"Registros: {len(df_amp):,}")
if len(df_amp) != EXPECTED_ROWS:
    print(
        f"[ADVERTENCIA] Para el corte del TFM se esperaban "
        f"{EXPECTED_ROWS:,} registros."
    )

columnas_necesarias = [
    "Tipo_Prestamo",
    "Desc_Unidad_Ejecutora",
    "Fecha_Apertura",
]
faltantes = [c for c in columnas_necesarias if c not in df_amp.columns]
if faltantes:
    raise KeyError(
        "Faltan columnas necesarias para la caracterización ampliada: "
        + ", ".join(faltantes)
    )


# ============================================================
# 1. DISTRIBUCIÓN POR TIPO DE PRÉSTAMO
# ============================================================

tipo = limpiar_texto(df_amp["Tipo_Prestamo"], "Sin identificar")
dist_tipo = (
    tipo.value_counts(dropna=False)
    .rename_axis("tipo_prestamo")
    .reset_index(name="cantidad")
)
dist_tipo["pct"] = dist_tipo["cantidad"] / len(df_amp) * 100

# Observación del director:
# si el porcentaje mostrado con un decimal sería 0.0 %, no se deja como
# categoría aislada. Se agrupa dentro de OTROS.
mask_otros = dist_tipo["pct"].round(1).eq(0.0)

if mask_otros.any():
    otros_cantidad = int(dist_tipo.loc[mask_otros, "cantidad"].sum())
    otros_pct = otros_cantidad / len(df_amp) * 100

    dist_tipo_plot = dist_tipo.loc[~mask_otros].copy()
    dist_tipo_plot = pd.concat(
        [
            dist_tipo_plot,
            pd.DataFrame(
                {
                    "tipo_prestamo": ["OTROS"],
                    "cantidad": [otros_cantidad],
                    "pct": [otros_pct],
                }
            ),
        ],
        ignore_index=True,
    )
else:
    dist_tipo_plot = dist_tipo.copy()

# Orden descendente; OTROS se mantiene al final para que sea fácil de reconocer.
sin_otros = dist_tipo_plot[dist_tipo_plot["tipo_prestamo"] != "OTROS"].copy()
sin_otros = sin_otros.sort_values("cantidad", ascending=False)
otros = dist_tipo_plot[dist_tipo_plot["tipo_prestamo"] == "OTROS"].copy()
dist_tipo_plot = pd.concat([sin_otros, otros], ignore_index=True)

dist_tipo.to_csv(
    OUTPUT_DIR_AMPLIADO / "dist_tipo_prestamo.csv",
    index=False,
    encoding="utf-8",
)

top5_pct = float(dist_tipo.head(5)["pct"].sum()) if len(dist_tipo) >= 5 else float(dist_tipo["pct"].sum())

plot_tipo = dist_tipo_plot.copy()
plot_tipo["etiqueta"] = plot_tipo["tipo_prestamo"].map(
    lambda x: abreviar_etiqueta(x, ancho=35)
)

# Se destacan las cinco modalidades principales; el resto queda neutro.
top5_nombres = set(dist_tipo.head(5)["tipo_prestamo"].tolist())
colores = [
    COLOR_ACENTO if nombre in top5_nombres else COLOR_NEUTRO
    for nombre in plot_tipo["tipo_prestamo"]
]

# Para barh invertimos el orden para que la categoría principal quede arriba.
plot_rev = plot_tipo.iloc[::-1].copy()
colores_rev = colores[::-1]

alto = max(6.0, 0.34 * len(plot_rev) + 2.2)
fig, ax = plt.subplots(figsize=(11.2, alto))

bars = ax.barh(
    plot_rev["etiqueta"],
    plot_rev["cantidad"],
    color=colores_rev,
    edgecolor="none",
)

max_val = max(plot_rev["cantidad"].max(), 1)
ax.set_xlim(0, max_val * 1.28)

for bar, cantidad, pct in zip(
    bars,
    plot_rev["cantidad"],
    plot_rev["pct"],
):
    ax.text(
        bar.get_width() + max_val * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{int(cantidad):,} ({pct:.1f} %)",
        va="center",
        ha="left",
        fontsize=8.6,
    )

ax.set_title(
    f"Las cinco principales modalidades concentran el {top5_pct:.1f} % de los créditos activos",
    loc="left",
    pad=12,
)
ax.set_xlabel("Número de créditos activos")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)
limpiar_ejes(ax, grid_axis="x")

if mask_otros.any():
    fig.text(
        0.125,
        0.012,
        "OTROS agrupa las modalidades cuya participación individual se mostraría como 0.0 % al redondear a un decimal.",
        fontsize=8.2,
        color="#475569",
    )
    rect = [0, 0.04, 1, 1]
else:
    rect = [0, 0, 1, 1]

fig.tight_layout(rect=rect)
guardar_figura_ampliada(fig, "fig_tipo_prestamo.png")


# ============================================================
# 2. TOP 20 UNIDADES EJECUTORAS
# ============================================================

unidad = limpiar_texto(
    df_amp["Desc_Unidad_Ejecutora"],
    "Sin identificar",
)

dist_unidad = (
    unidad.value_counts(dropna=False)
    .rename_axis("unidad_ejecutora")
    .reset_index(name="cantidad")
)
dist_unidad["pct"] = dist_unidad["cantidad"] / len(df_amp) * 100

dist_unidad.to_csv(
    OUTPUT_DIR_AMPLIADO / "dist_ubicacion.csv",
    index=False,
    encoding="utf-8",
)

top20 = dist_unidad.head(20).copy()
top20["etiqueta"] = top20["unidad_ejecutora"].map(
    lambda x: abreviar_etiqueta(x, ancho=35)
)

top2_nombres = set(top20.head(2)["unidad_ejecutora"].tolist())
colores = [
    COLOR_ACENTO if nombre in top2_nombres else COLOR_NEUTRO
    for nombre in top20["unidad_ejecutora"]
]

panel = top20.iloc[::-1].copy()
colores_panel = colores[::-1]

fig, ax = plt.subplots(figsize=(11.0, 7.2))

bars = ax.barh(
    panel["etiqueta"],
    panel["cantidad"],
    color=colores_panel,
    edgecolor="none",
)

max_val = max(panel["cantidad"].max(), 1)
ax.set_xlim(0, max_val * 1.26)

for bar, cantidad, pct in zip(
    bars,
    panel["cantidad"],
    panel["pct"],
):
    ax.text(
        bar.get_width() + max_val * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{int(cantidad):,} ({pct:.1f} %)",
        va="center",
        ha="left",
        fontsize=8.3,
    )

top20_pct = float(top20["pct"].sum())
ax.set_title(
    f"Las 20 principales unidades ejecutoras reúnen el {top20_pct:.1f} % de los créditos activos",
    loc="left",
    pad=12,
)
ax.set_xlabel("Número de créditos activos")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)
limpiar_ejes(ax, grid_axis="x")

fig.tight_layout()
guardar_figura_ampliada(fig, "fig_ubicacion.png")


# ============================================================
# 3. CRÉDITOS ACTIVOS POR AÑO DE APERTURA
# ============================================================

fecha_raw = limpiar_texto(df_amp["Fecha_Apertura"], "")
fecha_parseada = pd.to_datetime(
    fecha_raw.replace("", pd.NA),
    errors="coerce",
    dayfirst=True,
)

anio = fecha_parseada.dt.year

# Respaldo para formatos no reconocidos: extrae un año de cuatro dígitos.
faltan_anios = anio.isna()
if faltan_anios.any():
    extraido = fecha_raw[faltan_anios].str.extract(r"((?:19|20)\d{2})", expand=False)
    anio.loc[faltan_anios] = pd.to_numeric(extraido, errors="coerce")

anio = pd.to_numeric(anio, errors="coerce")
anio = anio[(anio >= 1900) & (anio <= 2024)].astype("Int64")

dist_anio = (
    anio.value_counts(dropna=True)
    .sort_index()
    .rename_axis("anio_apertura")
    .reset_index(name="cantidad")
)
dist_anio["pct"] = dist_anio["cantidad"] / len(df_amp) * 100

dist_anio.to_csv(
    OUTPUT_DIR_AMPLIADO / "dist_anio_apertura.csv",
    index=False,
    encoding="utf-8",
)

periodo_reciente = dist_anio[
    dist_anio["anio_apertura"].between(2021, 2024)
]
pct_reciente = float(periodo_reciente["pct"].sum())

fig, ax = plt.subplots(figsize=(11.2, 5.8))

ax.plot(
    dist_anio["anio_apertura"],
    dist_anio["cantidad"],
    color=COLOR_BASE,
    linewidth=2.2,
    marker="o",
    markersize=4.5,
)

# Destaca los últimos cuatro años sin convertir el gráfico en un arcoíris.
rec = dist_anio[dist_anio["anio_apertura"].between(2021, 2024)]
ax.scatter(
    rec["anio_apertura"],
    rec["cantidad"],
    color=COLOR_ACENTO,
    s=55,
    zorder=4,
)

for _, fila in rec.iterrows():
    ax.annotate(
        f"{int(fila['cantidad']):,}",
        xy=(fila["anio_apertura"], fila["cantidad"]),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=8.2,
        color=COLOR_ACENTO,
    )

ax.set_title(
    f"Los créditos abiertos entre 2021 y 2024 representan el {pct_reciente:.1f} % del stock activo",
    loc="left",
    pad=12,
)
ax.set_xlabel("Año de apertura")
ax.set_ylabel("Número de créditos activos")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)

# Ticks suficientemente espaciados para evitar saturación.
anios = dist_anio["anio_apertura"].astype(int).tolist()
if len(anios) > 15:
    ticks = anios[::2]
    if anios[-1] not in ticks:
        ticks.append(anios[-1])
    ax.set_xticks(ticks)
else:
    ax.set_xticks(anios)

ax.tick_params(axis="x", rotation=45)
limpiar_ejes(ax, grid_axis="y")

fig.text(
    0.125,
    0.01,
    "La serie describe la composición del stock activo a noviembre de 2024 según su año de apertura; no representa colocaciones históricas anuales.",
    fontsize=8.2,
    color="#475569",
)

fig.tight_layout(rect=[0, 0.045, 1, 1])
guardar_figura_ampliada(fig, "fig_anio_apertura.png")


# ============================================================
# 4. SALIDAS COMPLEMENTARIAS
# ============================================================

if "Sexo" in df.columns:
    sexo = limpiar_texto(df_amp["Sexo"], "Sin identificar")
    dist_sexo = (
        sexo.value_counts(dropna=False)
        .rename_axis("sexo")
        .reset_index(name="cantidad")
    )
    dist_sexo["pct"] = dist_sexo["cantidad"] / len(df_amp) * 100
    dist_sexo.to_csv(
        OUTPUT_DIR_AMPLIADO / "dist_sexo.csv",
        index=False,
        encoding="utf-8",
    )

# Se conserva un resumen simple de las variables financieras disponibles.
vars_estadisticas = [
    "Dias_Mora",
    "Cuotas_Vencidas",
    "Abono_Promedio",
    "Saldo_Desembolsado",
    "Saldo_Vigente",
]
vars_disp = [v for v in vars_estadisticas if v in df.columns]

if vars_disp:
    datos_est = df_amp[vars_disp].copy()
    for col in vars_disp:
        datos_est[col] = pd.to_numeric(datos_est[col], errors="coerce")
    datos_est.describe().T.to_csv(
        OUTPUT_DIR_AMPLIADO / "estadisticas_7vars.csv",
        encoding="utf-8",
    )

print("\nCaracterización ampliada completada.")
print(f"Outputs ampliados: {OUTPUT_DIR_AMPLIADO.resolve()}")
