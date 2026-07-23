# -*- coding: utf-8 -*-
"""Calcula evidencia complementaria para el informe final.

Los resultados se derivan únicamente de los archivos persistidos por los
experimentos 02 y 03. El script no reentrena modelos ni modifica sus salidas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
EXP02 = (
    ROOT
    / "020_Isolation_Forest"
    / "exp_02_refinamiento"
    / "resultados"
    / "anomalias_exp02.parquet"
)
BASE = ROOT / "020_Isolation_Forest" / "datos" / "V_BASE_OFERTA_ITEM.parquet"
OUT = Path(__file__).resolve().parent / "resultados_complementarios.csv"
CORR_OUT = Path(__file__).resolve().parent / "correlaciones_features.csv"


def rank_biserial(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Devuelve U y correlación biserial de rangos para x frente a y."""
    result = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    effect = 2.0 * result.statistic / (len(x) * len(y)) - 1.0
    return float(result.statistic), float(effect)


exp = pd.read_parquet(
    EXP02,
    columns=[
        "NRO_SICOP",
        "NRO_OFERTA",
        "NRO_LINEA",
        "OFERTA_PRECIO_UNITARIO_CRC",
        "RATIO_OFERTADO_VS_ESTIMADO",
        "N_OFERTAS_ITEM",
        "N_OFERENTES_ITEM",
        "SINGLE_BID_FLAG",
        "ANOMALY_FLAG",
    ],
)

# Las variables temporales y de dispersión no se exportaron en el parquet del
# experimento 02. Se recuperan de la base analítica mediante la llave original.
base_columns = pd.read_parquet(BASE).columns
candidate_columns = [
    "NRO_SICOP",
    "NRO_OFERTA",
    "NRO_LINEA",
    "FECHA_CIERRE_RECEPCION",
    "FECHA_PRESENTA_OFERTA",
    "PRECIO_PROM_ITEM_CRC",
    "PRECIO_STDDEV_ITEM_CRC",
]
available = [column for column in candidate_columns if column in base_columns]
base = pd.read_parquet(BASE, columns=available)

keys = ["NRO_SICOP", "NRO_OFERTA", "NRO_LINEA"]
extra = [column for column in available if column not in keys]
if extra:
    exp = exp.merge(base, on=keys, how="left", validate="one_to_one")

for column in ["FECHA_CIERRE_RECEPCION", "FECHA_PRESENTA_OFERTA"]:
    exp[column] = pd.to_datetime(exp[column], errors="coerce")
exp["DIAS_PARA_CIERRE"] = (
    exp["FECHA_CIERRE_RECEPCION"] - exp["FECHA_PRESENTA_OFERTA"]
).dt.total_seconds() / 86_400
exp["CV_PRECIO_ITEM"] = np.where(
    exp["PRECIO_PROM_ITEM_CRC"].isna() | exp["PRECIO_PROM_ITEM_CRC"].eq(0),
    np.nan,
    exp["PRECIO_STDDEV_ITEM_CRC"] / exp["PRECIO_PROM_ITEM_CRC"],
)

numeric_candidates = [
    "OFERTA_PRECIO_UNITARIO_CRC",
    "RATIO_OFERTADO_VS_ESTIMADO",
    "N_OFERTAS_ITEM",
    "N_OFERENTES_ITEM",
    "DIAS_PARA_CIERRE",
    "CV_PRECIO_ITEM",
]

rows = []
for column in numeric_candidates:
    if column not in exp.columns:
        continue
    anomaly = exp.loc[exp["ANOMALY_FLAG"].eq(1), column].dropna()
    normal = exp.loc[exp["ANOMALY_FLAG"].eq(0), column].dropna()
    u_stat, effect = rank_biserial(anomaly, normal)
    ks_result = ks_2samp(anomaly, normal, alternative="two-sided", method="asymp")
    rows.append(
        {
            "variable": column,
            "n_anomalia": len(anomaly),
            "n_normal": len(normal),
            "mediana_anomalia": anomaly.median(),
            "mediana_normal": normal.median(),
            "u_mann_whitney": u_stat,
            "rank_biserial": effect,
            "ks_d": float(ks_result.statistic),
            "ks_p": float(ks_result.pvalue),
        }
    )

pd.DataFrame(rows).to_csv(OUT, index=False)

# La correlación se estima sobre una muestra reproducible para evitar que el
# tamaño del conjunto convierta diferencias irrelevantes en resultados
# aparentemente precisos. Se incluyen transformaciones derivadas cuando están
# disponibles; si no lo están, se reconstruyen con la misma definición.
corr = exp[["OFERTA_PRECIO_UNITARIO_CRC", "RATIO_OFERTADO_VS_ESTIMADO"]].copy()
corr["LOG_PRECIO_OFERTADO"] = np.where(
    corr["OFERTA_PRECIO_UNITARIO_CRC"].gt(0),
    np.log10(corr["OFERTA_PRECIO_UNITARIO_CRC"]),
    np.nan,
)
corr["LOG_RATIO_VS_ESTIMADO"] = np.where(
    corr["RATIO_OFERTADO_VS_ESTIMADO"].gt(0),
    np.log10(corr["RATIO_OFERTADO_VS_ESTIMADO"]),
    np.nan,
)
corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
sample = corr.sample(n=min(200_000, len(corr)), random_state=42)
sample.corr(method="spearman").to_csv(CORR_OUT)

print(f"OK -> {OUT.name}")
print(f"OK -> {CORR_OUT.name}")
