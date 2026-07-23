# -*- coding: utf-8 -*-
"""Genera las figuras verificables usadas por el informe final.

Las visualizaciones se derivan de la base analítica y de salidas persistidas
de los experimentos 02 y 03. No reentrena modelos ni modifica resultados.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figuras"
FIG.mkdir(exist_ok=True)

BASE = ROOT / "020_Isolation_Forest" / "datos" / "V_BASE_OFERTA_ITEM.parquet"
EXP02 = (
    ROOT
    / "020_Isolation_Forest"
    / "exp_02_refinamiento"
    / "resultados"
    / "anomalias_exp02.parquet"
)
EXP03 = ROOT / "020_Isolation_Forest" / "exp_03_explicabilidad" / "resultados"

BLUE = "#1f4e79"
LIGHT_BLUE = "#9dc3e6"
ORANGE = "#c55a11"
GREEN = "#548235"
GRAY = "#666666"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 140,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_base() -> pd.DataFrame:
    columns = [
        "NRO_SICOP",
        "NRO_OFERTA",
        "NRO_LINEA",
        "OFERTA_SEGMENTO",
        "OFERTA_PRECIO_UNITARIO_CRC",
        "CARTEL_PRECIO_UNITARIO_CRC",
        "FECHA_PRESENTA_OFERTA",
        "FECHA_CIERRE_RECEPCION",
    ]
    return pd.read_parquet(BASE, columns=columns)


def eda_y_calidad(base: pd.DataFrame) -> None:
    price = base["OFERTA_PRECIO_UNITARIO_CRC"]
    positive = price[price.gt(0)].dropna()
    rng = np.random.default_rng(42)
    if len(positive) > 250_000:
        positive = positive.iloc[rng.choice(len(positive), 250_000, replace=False)]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].hist(np.log10(positive), bins=70, color=BLUE, alpha=0.9)
    axes[0].axvline(np.log10(15_820), color=ORANGE, linestyle="--", label="Mediana: ₡15 820")
    axes[0].set_title("Distribución del precio unitario ofertado")
    axes[0].set_xlabel("log10(precio unitario en CRC)")
    axes[0].set_ylabel("Ofertas en muestra reproducible")
    axes[0].legend(frameon=False)

    seg = base.loc[price.gt(0), ["OFERTA_SEGMENTO", "OFERTA_PRECIO_UNITARIO_CRC"]].copy()
    seg["log_precio"] = np.log10(seg["OFERTA_PRECIO_UNITARIO_CRC"])
    top = seg["OFERTA_SEGMENTO"].value_counts().head(12).index
    quant = (
        seg[seg["OFERTA_SEGMENTO"].isin(top)]
        .groupby("OFERTA_SEGMENTO")["log_precio"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .sort_values(0.5)
    )
    y = np.arange(len(quant))
    axes[1].hlines(y, quant[0.25], quant[0.75], color=LIGHT_BLUE, linewidth=7)
    axes[1].scatter(quant[0.5], y, color=BLUE, s=28, zorder=3)
    axes[1].set_yticks(y, [str(value) for value in quant.index])
    axes[1].set_title("Heterogeneidad entre segmentos UNSPSC")
    axes[1].set_xlabel("Mediana e intervalo intercuartílico de log10(precio)")
    axes[1].set_ylabel("Segmento")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, "eda_precios_segmentos.png")

    keys = ["NRO_SICOP", "NRO_OFERTA", "NRO_LINEA"]
    dates_offer = pd.to_datetime(base["FECHA_PRESENTA_OFERTA"], errors="coerce")
    dates_close = pd.to_datetime(base["FECHA_CIERRE_RECEPCION"], errors="coerce")
    missing_dates = dates_offer.isna() | dates_close.isna()
    invalid_order = (~missing_dates) & dates_offer.gt(dates_close)
    valid_order = (~missing_dates) & (~invalid_order)

    metrics = pd.DataFrame(
        [
            ("Filas totales", len(base)),
            ("Duplicados por llave", int(base.duplicated(keys, keep=False).sum())),
            ("Precio ofertado nulo", int(price.isna().sum())),
            ("Precio ofertado igual a cero", int(price.eq(0).sum())),
            ("Precio ofertado negativo", int(price.lt(0).sum())),
            (
                "Precio estimado nulo",
                int(base["CARTEL_PRECIO_UNITARIO_CRC"].isna().sum()),
            ),
            ("Secuencia temporal válida", int(valid_order.sum())),
            ("Fechas incompletas", int(missing_dates.sum())),
            ("Presentación posterior al cierre", int(invalid_order.sum())),
        ],
        columns=["control", "n"],
    )
    metrics["pct"] = metrics["n"] / len(base) * 100
    metrics.to_csv(OUT / "metricas_calidad_datos.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    quality_labels = [
        "Duplicados",
        "Precio nulo",
        "Precio cero",
        "Precio negativo",
        "Estimado nulo",
    ]
    quality_values = [
        metrics.loc[metrics["control"].eq("Duplicados por llave"), "n"].iat[0],
        metrics.loc[metrics["control"].eq("Precio ofertado nulo"), "n"].iat[0],
        metrics.loc[metrics["control"].eq("Precio ofertado igual a cero"), "n"].iat[0],
        metrics.loc[metrics["control"].eq("Precio ofertado negativo"), "n"].iat[0],
        metrics.loc[metrics["control"].eq("Precio estimado nulo"), "n"].iat[0],
    ]
    bars = axes[0].bar(quality_labels, quality_values, color=[GREEN, BLUE, BLUE, GREEN, BLUE])
    axes[0].set_title("Controles de integridad y completitud")
    axes[0].set_ylabel("Número de filas")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].bar_label(bars, labels=[f"{v:,}".replace(",", " ") for v in quality_values], fontsize=8)

    temporal_values = [int(valid_order.sum()), int(missing_dates.sum()), int(invalid_order.sum())]
    bars = axes[1].bar(
        ["Secuencia válida", "Fechas incompletas", "Posterior al cierre"],
        temporal_values,
        color=[GREEN, LIGHT_BLUE, ORANGE],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validación de secuencia temporal (escala log)")
    axes[1].set_ylabel("Número de filas, escala logarítmica")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].bar_label(bars, labels=[f"{v:,}".replace(",", " ") for v in temporal_values], fontsize=8)
    fig.tight_layout()
    save(fig, "calidad_validaciones_dominio.png")


def perfil_anomalias() -> None:
    columns = [
        "OFERTA_PRECIO_UNITARIO_CRC",
        "RATIO_OFERTADO_VS_ESTIMADO",
        "N_OFERENTES_ITEM",
        "ANOMALY_FLAG",
    ]
    df = pd.read_parquet(EXP02, columns=columns)
    anom = df[df["ANOMALY_FLAG"].eq(1)]
    normal = df[df["ANOMALY_FLAG"].eq(0)].sample(n=150_000, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    groups = [(normal, GRAY, "Normal"), (anom, ORANGE, "Priorizada")]
    price_values = [
        np.log10(group.loc[group["OFERTA_PRECIO_UNITARIO_CRC"].gt(0), "OFERTA_PRECIO_UNITARIO_CRC"])
        for group, _, _ in groups
    ]
    ratio_values = [
        np.log10(
            1
            + group["RATIO_OFERTADO_VS_ESTIMADO"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .clip(lower=0)
        )
        for group, _, _ in groups
    ]
    price_all = pd.concat(price_values)
    ratio_all = pd.concat(ratio_values)
    price_bins = np.linspace(price_all.quantile(0.001), price_all.quantile(0.999), 60)
    ratio_bins = np.linspace(ratio_all.quantile(0.001), ratio_all.quantile(0.99), 60)
    for index, (group, color, label) in enumerate(groups):
        axes[0].hist(price_values[index], bins=price_bins, density=True, histtype="step", linewidth=1.6, color=color, label=label)
        axes[1].hist(ratio_values[index], bins=ratio_bins, density=True, histtype="step", linewidth=1.6, color=color, label=label)
        offerers = group["N_OFERENTES_ITEM"].dropna().clip(upper=20)
        axes[2].hist(offerers, bins=np.arange(0.5, 21.5), density=True, histtype="step", linewidth=1.6, color=color, label=label)
    axes[0].set_title("Precio unitario")
    axes[0].set_xlabel("log10(CRC)")
    axes[1].set_title("Ratio oferta/estimado")
    axes[1].set_xlabel("log10(1 + ratio), mostrado hasta P99")
    axes[2].set_title("Número de oferentes")
    axes[2].set_xlabel("Oferentes (20 agrupa la cola)")
    for ax in axes:
        ax.set_ylabel("Densidad")
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "features_anom_vs_normal.png")


def shap_global() -> None:
    seg = pd.read_csv(EXP03 / "shap_importancia_segmento.csv").head(12)
    cls = pd.read_csv(EXP03 / "shap_importancia_clase.csv").head(12)
    merged = seg[["feature", "importancia_seg"]].merge(
        cls[["feature", "importancia_cls"]], on="feature", how="outer"
    ).fillna(0)
    merged["max"] = merged[["importancia_seg", "importancia_cls"]].max(axis=1)
    merged = merged.nlargest(12, "max").sort_values("max")

    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.barh(y - 0.18, merged["importancia_seg"], 0.34, label="Segmento", color=BLUE)
    ax.barh(y + 0.18, merged["importancia_cls"], 0.34, label="Clase", color=ORANGE)
    ax.set_yticks(y, [value.replace("_", " ") for value in merged["feature"]])
    ax.set_xlabel("Media de |SHAP|")
    ax.set_title("Importancia global SHAP por nivel de la arquitectura")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, "shap_importancia_global.png")


def estabilidad_y_metodos() -> None:
    counts = pd.Series(
        [2_820_383, 32_255, 20_677, 18_589, 21_201, 108_231],
        index=["0/5", "1/5", "2/5", "3/5", "4/5", "5/5"],
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.bar(counts.index, counts.values, color=[GRAY, ORANGE, LIGHT_BLUE, LIGHT_BLUE, BLUE, GREEN])
    ax.set_yscale("log")
    ax.set_ylabel("Ofertas (escala logarítmica)")
    ax.set_xlabel("Número de semillas en que la oferta fue priorizada")
    ax.set_title("Persistencia de las alertas en cinco semillas")
    ax.bar_label(bars, labels=[f"{v:,}".replace(",", " ") for v in counts], fontsize=8)
    fig.tight_layout()
    save(fig, "estabilidad_semillas.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    specs = [
        ((4.0, 3.9), 2.35, BLUE, "IF"),
        ((6.0, 3.9), 2.35, ORANGE, "HBOS"),
        ((5.0, 2.5), 2.35, GREEN, "ECOD"),
    ]
    for (center, radius, color, label) in specs:
        ax.add_patch(Circle(center, radius, facecolor=color, edgecolor=color, alpha=0.25, linewidth=2))
        ax.text(center[0], center[1] + radius - 0.35, label, color=color, weight="bold", ha="center")
    ax.text(5.0, 3.45, "17 526", ha="center", va="center", weight="bold", fontsize=12)
    ax.text(2.75, 4.15, "90 172\nsolo IF", ha="center", va="center", fontsize=8)
    ax.text(7.25, 4.15, "100 319\nsolo HBOS", ha="center", va="center", fontsize=8)
    ax.text(5.0, 1.05, "70 800\nsolo ECOD", ha="center", va="center", fontsize=8)
    ax.text(5.0, 5.15, "6 926", ha="center", fontsize=8)
    ax.text(3.95, 2.35, "36 443", ha="center", fontsize=8)
    ax.text(6.05, 2.35, "26 298", ha="center", fontsize=8)
    ax.set_title("Intersecciones al percentil 5")
    save(fig, "venn_metodos.png")


def temporal() -> None:
    yearly = pd.read_csv(EXP03 / "validacion_temporal_por_anio.csv")
    metrics = pd.read_csv(EXP03 / "validacion_temporal_metricas.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    axes[0].plot(yearly.iloc[:, 0], yearly["pct"], marker="o", color=BLUE)
    axes[0].axvline(2023.5, color=ORANGE, linestyle="--")
    axes[0].set_title("Tasa anual con umbral fijo")
    axes[0].set_xlabel("Año")
    axes[0].set_ylabel("Ofertas priorizadas (%)")
    axes[0].grid(alpha=0.2)
    values = [metrics["pct_anom_train_umbral_fijo"], metrics["pct_anom_test_umbral_fijo"]]
    bars = axes[1].bar(["Entrenamiento\n2020–2023", "Prueba\n2024–2025"], values, color=[BLUE, ORANGE])
    axes[1].set_title(f"Cambio fuera de muestra · Jaccard {metrics['jaccard_oot_vs_full']:.3f}")
    axes[1].set_ylabel("Ofertas priorizadas (%)")
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in values])
    fig.tight_layout()
    save(fig, "validacion_temporal.png")


def box(ax, x, y, w, h, text, color=BLUE, fontsize=8) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=GRAY))


def diagramas() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, 0.3, 2.5, 2.0, 1.2, "SICOP\n12 archivos semestrales")
    box(ax, 3.0, 2.5, 2.0, 1.2, "Oracle\ncarga y persistencia")
    box(ax, 5.7, 2.5, 2.0, 1.2, "Vistas SQL\nestandarización")
    box(ax, 8.4, 2.5, 2.0, 1.2, "Tabla analítica\n3 021 336 ofertas")
    box(ax, 11.1, 4.7, 2.0, 1.0, "57 modelos IF\npor segmento", ORANGE)
    box(ax, 11.1, 0.5, 2.0, 1.0, "656 modelos IF\npor clase", ORANGE)
    box(ax, 11.1, 2.6, 2.0, 1.0, "Contrastes\nHBOS · ECOD", "#9673a6")
    box(ax, 13.8, 2.5, 1.9, 1.2, "Lista priorizada\n+ explicaciones", GREEN)
    for x in [2.3, 5.0, 7.7]:
        arrow(ax, x, 3.1, x + 0.7, 3.1)
    arrow(ax, 10.4, 3.1, 11.1, 5.2)
    arrow(ax, 10.4, 3.1, 11.1, 1.0)
    arrow(ax, 10.4, 3.1, 11.1, 3.1)
    arrow(ax, 13.1, 5.2, 13.8, 3.5)
    arrow(ax, 13.1, 1.0, 13.8, 2.7)
    arrow(ax, 13.1, 3.1, 13.8, 3.1)
    ax.text(4.0, 1.9, "SQL", color=GRAY, ha="center")
    ax.text(9.4, 1.9, "Python · pandas", color=GRAY, ha="center")
    ax.text(12.1, 6.0, "scikit-learn", color=GRAY, ha="center")
    ax.text(12.1, 2.25, "PyOD", color=GRAY, ha="center")
    ax.text(14.75, 1.9, "SHAP · reportes", color=GRAY, ha="center")
    ax.set_title("Arquitectura de datos, modelado y salida", fontsize=12, weight="bold")
    save(fig, "diagrama_arquitectura.png")

    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    entities = {
        "INSTITUCIONES": (0.2, 6.4, "PK CEDULA"),
        "V_CARTELES": (3.2, 6.4, "PK NRO_SICOP\nFK CEDULA_INSTITUCION"),
        "V_LINEA_CARTELES": (6.4, 6.4, "PK NRO_SICOP + NUMERO_LINEA\nCODIGO_IDENTIFICACION"),
        "JERARQUÍA UNSPSC DERIVADA": (10.3, 6.4, "no es tabla física\nprefijos: segmento · familia · clase"),
        "PROVEEDORES": (0.2, 2.7, "PK CEDULA_PROVEEDOR"),
        "V_OFERTAS": (3.2, 2.7, "PK NRO_SICOP + IDENTIFICADOR\nFK CEDULA_PROVEEDOR"),
        "V_LINEAS_OFERTAS": (6.4, 2.7, "PK NRO_SICOP + NRO_OFERTA + NRO_LINEA\n+ CODIGO_PRODUCTO"),
        "V_BASE_OFERTA_ITEM": (10.3, 2.7, "oferta–ítem\nprecios · competencia · tiempo"),
        "LINEA_CONTRATOS": (6.4, 0.3, "FK NRO_SICOP + NRO_LINEA_CARTEL\nFK CEDULA_PROVEEDOR · uso post-hoc"),
    }
    for title, (x, y, fields) in entities.items():
        color = GREEN if title == "V_BASE_OFERTA_ITEM" else (GRAY if title == "LINEA_CONTRATOS" else BLUE)
        width = 3.1 if title in {"V_LINEA_CARTELES", "V_LINEAS_OFERTAS", "LINEA_CONTRATOS"} else 2.55
        box(ax, x, y, width, 1.35, f"{title}\n{fields}", color, 7.2)
    arrow(ax, 2.75, 7.05, 3.2, 7.05)
    arrow(ax, 5.75, 7.05, 6.4, 7.05)
    arrow(ax, 9.5, 7.05, 10.3, 7.05)
    arrow(ax, 2.75, 3.35, 3.2, 3.35)
    arrow(ax, 5.75, 3.35, 6.4, 3.35)
    arrow(ax, 9.5, 3.35, 10.3, 3.35)
    arrow(ax, 7.95, 6.4, 7.95, 4.05)
    arrow(ax, 7.95, 2.7, 11.0, 4.05)
    arrow(ax, 8.0, 1.65, 10.3, 2.7)
    ax.text(2.9, 7.35, "1:N", fontsize=7, color=GRAY)
    ax.text(6.0, 7.35, "1:N", fontsize=7, color=GRAY)
    ax.text(9.85, 7.35, "N:1", fontsize=7, color=GRAY)
    ax.text(2.9, 3.65, "1:N", fontsize=7, color=GRAY)
    ax.text(6.0, 3.65, "1:N", fontsize=7, color=GRAY)
    ax.set_title("Modelo lógico de las entidades utilizadas", fontsize=12, weight="bold")
    save(fig, "diagrama_entidad_relacion.png")


def main() -> None:
    base = load_base()
    eda_y_calidad(base)
    del base
    perfil_anomalias()
    shap_global()
    estabilidad_y_metodos()
    temporal()
    diagramas()
    print(f"OK -> {FIG}")
    print(f"OK -> {OUT / 'metricas_calidad_datos.csv'}")


if __name__ == "__main__":
    main()
