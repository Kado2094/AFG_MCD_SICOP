"""
Replica V_BASE_OFERTA_ITEM desde los CSV locales (000_Datos).
Granularidad: una fila por oferta por item.
PK: (NRO_SICOP, NRO_OFERTA, NRO_LINEA)
Precios colonizados a CRC.
Output: V_BASE_OFERTA_ITEM.parquet
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r'c:\Users\Federico Chan\OneDrive - McKinsey & Company'
            r'\Documents\Personal\PUC\999 - Trabajo de graduacion\000_Datos')
OUT  = Path(__file__).parent

# =====================================================================
# PASO 1: CARGA DE DATOS
# =====================================================================
print("=" * 70)
print("PASO 1: CARGA DE DATOS")
print("=" * 70)
lineas_ofertas  = pd.read_csv(BASE / 'Ofertas' / 'lineas_ofertas.csv',
                              sep='|', low_memory=False)
ofertas_raw     = pd.read_csv(BASE / 'Ofertas' / 'ofertas.csv',
                              sep='|', low_memory=False)
linea_carteles  = pd.read_csv(BASE / 'Detalle de Carteles' / 'linea_carteles.csv',
                              sep='|', low_memory=False)
carteles        = pd.read_csv(BASE / 'Detalle de Carteles' / 'carteles.csv',
                              sep='|', low_memory=False)
linea_contratos = pd.read_csv(BASE / 'Contratos' / 'linea_contratos.csv',
                              sep='|', low_memory=False)
instituciones   = pd.read_csv(BASE / 'Instituciones' / 'instituciones.csv',
                              sep='|', low_memory=False)
proveedores     = pd.read_csv(BASE / 'Proveedores' / 'proveedores.csv',
                              sep='|', low_memory=False)
for name, df in [('lineas_ofertas', lineas_ofertas),
                 ('ofertas', ofertas_raw),
                 ('linea_carteles', linea_carteles),
                 ('carteles', carteles),
                 ('linea_contratos', linea_contratos),
                 ('instituciones', instituciones),
                 ('proveedores', proveedores)]:
    print(f"  {name:<20s}: {len(df):>10,} registros, {df.shape[1]} cols")

# =====================================================================
# PASO 2: CTE ofertas_base (V_LINEAS_OFERTAS LEFT JOIN V_OFERTAS)
# =====================================================================
print("\n" + "=" * 70)
print("PASO 2: ofertas_base")
print("=" * 70)
ob = lineas_ofertas.merge(
    ofertas_raw[['NRO_SICOP', 'IDENTIFICADOR', 'CEDULA_PROVEEDOR',
                 'ELEGIBLE', 'TIPO_OFERTA', 'ESTADO',
                 'FECHA_PRESENTA_OFERTA', 'ID_CONSORCIO']],
    left_on=['NRO_SICOP', 'NRO_OFERTA'],
    right_on=['NRO_SICOP', 'IDENTIFICADOR'],
    how='left'
)
ob.drop(columns=['IDENTIFICADOR'], inplace=True, errors='ignore')
ob.rename(columns={
    'CODIGO_PRODUCTO': 'OFERTA_CODIGO_PRODUCTO',
    'TIPO_MONEDA':     'OFERTA_TIPO_MONEDA',
    'IVA':             'OFERTA_IVA',
    'DESCUENTO':       'OFERTA_DESCUENTO',
    'OTROS_IMPUESTOS': 'OFERTA_OTROS_IMPUESTOS',
    'ACARREOS':        'OFERTA_ACARREOS',
    'TIPO_CAMBIO_CRC': 'OFERTA_TIPO_CAMBIO_CRC',
    'ELEGIBLE':        'OFERTA_ELEGIBLE',
    'ESTADO':          'OFERTA_ESTADO',
}, inplace=True)
prod = ob['OFERTA_CODIGO_PRODUCTO'].astype(str)
ob['OFERTA_SEGMENTO']   = prod.str[:2]
ob['OFERTA_FAMILIA']    = prod.str[:4]
ob['OFERTA_CLASE']      = prod.str[:6]
ob['OFERTA_MERCADERIA'] = prod.str[:8]
# Colonizacion de precios de oferta
mask_fx = (ob['OFERTA_TIPO_MONEDA'] != 'CRC') & ob['OFERTA_TIPO_CAMBIO_CRC'].notna() & (ob['OFERTA_TIPO_CAMBIO_CRC'] > 0)
ob['OFERTA_PRECIO_UNITARIO_CRC'] = ob['PRECIO_UNITARIO_OFERTADO'].copy()
ob.loc[mask_fx, 'OFERTA_PRECIO_UNITARIO_CRC'] = (
    ob.loc[mask_fx, 'PRECIO_UNITARIO_OFERTADO'] * ob.loc[mask_fx, 'OFERTA_TIPO_CAMBIO_CRC']
)
n_fx = mask_fx.sum()
print(f"  Ofertas base: {len(ob):,} filas")
pct = n_fx / len(ob) * 100
print(f"  Colonizacion: {n_fx:,} registros ({pct:.1f}%) convertidos a CRC")

# =====================================================================
# PASO 3: CTE cartel_info (V_LINEA_CARTELES LEFT JOIN V_CARTELES)
# =====================================================================
print("\n" + "=" * 70)
print("PASO 3: cartel_info")
print("=" * 70)
ci = linea_carteles.merge(
    carteles[['NRO_SICOP', 'NRO_PROCEDIMIENTO', 'CEDULA_INSTITUCION',
              'COD_UNIDAD_COMPRA', 'NOMBRE_UNIDAD_COMPRA',
              'TIPO_PROCEDIMIENTO', 'MODALIDAD_PROCEDIMIENTO',
              'EXCEPCION_CD', 'DESCRIPCION', 'PAGO_ADELANTADO_PYMES',
              'FECHA_PUBLICACION', 'FECHA_CIERRE_RECEPCION', 'FECHA_APERTURA']],
    on='NRO_SICOP', how='left'
)
ci.rename(columns={
    'NUMERO_PARTIDA':          'CARTEL_NUMERO_PARTIDA',
    'CODIGO_IDENTIFICACION':   'CARTEL_CODIGO_PRODUCTO',
    'TIPO_MONEDA':             'CARTEL_TIPO_MONEDA',
    'PRECIO_UNITARIO_ESTIMADO':'CARTEL_PRECIO_UNITARIO_ESTIMADO',
    'CANTIDAD_SOLICITADA':     'CARTEL_CANTIDAD_SOLICITADA',
    'TIPO_CAMBIO_USD':         'CARTEL_TIPO_CAMBIO_USD',
    'DESCRIPCION':             'CARTEL_DESCRIPCION',
}, inplace=True)
cprod = ci['CARTEL_CODIGO_PRODUCTO'].astype(str)
ci['CARTEL_SEGMENTO']   = cprod.str[:2]
ci['CARTEL_FAMILIA']    = cprod.str[:4]
ci['CARTEL_CLASE']      = cprod.str[:6]
ci['CARTEL_MERCADERIA'] = cprod.str[:8]
# Colonizacion de precios estimados del cartel
mask_ci = (ci['CARTEL_TIPO_MONEDA'] != 'CRC') & ci['CARTEL_TIPO_CAMBIO_USD'].notna() & (ci['CARTEL_TIPO_CAMBIO_USD'] > 0)
ci['CARTEL_PRECIO_UNITARIO_CRC'] = ci['CARTEL_PRECIO_UNITARIO_ESTIMADO'].copy()
ci.loc[mask_ci, 'CARTEL_PRECIO_UNITARIO_CRC'] = (
    ci.loc[mask_ci, 'CARTEL_PRECIO_UNITARIO_ESTIMADO'] * ci.loc[mask_ci, 'CARTEL_TIPO_CAMBIO_USD']
)
ci['CARTEL_MONTO_TOTAL_CRC'] = ci['CARTEL_CANTIDAD_SOLICITADA'].fillna(0) * ci['CARTEL_PRECIO_UNITARIO_CRC'].fillna(0)
print(f"  cartel_info: {len(ci):,} filas")

# =====================================================================
# PASO 4: CTE item_competencia (agregar por NRO_SICOP, NRO_LINEA)
# =====================================================================
print("\n" + "=" * 70)
print("PASO 4: item_competencia")
print("=" * 70)
grp_ic = ob.groupby(['NRO_SICOP', 'NRO_LINEA'])
ic_counts = grp_ic.agg(
    N_OFERTAS_ITEM=('NRO_OFERTA', 'count'),
    N_OFERENTES_ITEM=('CEDULA_PROVEEDOR', 'nunique'),
).reset_index()
elegibles = ob[ob['OFERTA_ELEGIBLE'] == 'Sí'].groupby(
    ['NRO_SICOP', 'NRO_LINEA']
)['CEDULA_PROVEEDOR'].nunique().reset_index()
elegibles.columns = ['NRO_SICOP', 'NRO_LINEA', 'N_OFERENTES_ELEGIBLES_ITEM']
ic_counts = ic_counts.merge(elegibles, on=['NRO_SICOP', 'NRO_LINEA'], how='left')
ic_counts['N_OFERENTES_ELEGIBLES_ITEM'] = ic_counts['N_OFERENTES_ELEGIBLES_ITEM'].fillna(0).astype(int)
print(f"  item_competencia: {len(ic_counts):,} grupos")

# =====================================================================
# PASO 5: CTE proveedores_adjudicados
# =====================================================================
print("\n" + "=" * 70)
print("PASO 5: proveedores_adjudicados")
print("=" * 70)
pa = linea_contratos[linea_contratos['CEDULA_PROVEEDOR'].notna()][
    ['NRO_SICOP', 'NRO_LINEA_CARTEL', 'CEDULA_PROVEEDOR']
].drop_duplicates()
pa.rename(columns={
    'NRO_LINEA_CARTEL': 'NRO_LINEA',
    'CEDULA_PROVEEDOR': 'PROVEEDOR_CONTRATADO',
}, inplace=True)
print(f"  proveedores_adjudicados: {len(pa):,} combinaciones unicas")

# =====================================================================
# PASO 6: CTE contrato_item (GROUP BY NRO_SICOP, NRO_LINEA_CARTEL)
# =====================================================================
print("\n" + "=" * 70)
print("PASO 6: contrato_item")
print("=" * 70)
grp_ct = linea_contratos.groupby(['NRO_SICOP', 'NRO_LINEA_CARTEL'])
ct_num = grp_ct.agg(
    CONTRATO_PRECIO_UNITARIO=('PRECIO_UNITARIO', 'mean'),
    CONTRATO_CANTIDAD=('CANTIDAD_CONTRATADA', lambda x: x.fillna(0).sum()),
).reset_index()
ct_cat = grp_ct.agg(
    CONTRATO_TIPO_MONEDA=('TIPO_MONEDA', 'first'),
    CONTRATO_TIPO_CAMBIO_CRC=('TIPO_CAMBIO_CRC', 'first'),
).reset_index()
ct = ct_num.merge(ct_cat, on=['NRO_SICOP', 'NRO_LINEA_CARTEL'])
# Colonizacion de precios de contrato
mask_ct = (ct['CONTRATO_TIPO_MONEDA'] != 'CRC') & ct['CONTRATO_TIPO_CAMBIO_CRC'].notna() & (ct['CONTRATO_TIPO_CAMBIO_CRC'] > 0)
ct['CONTRATO_PRECIO_UNITARIO_CRC'] = ct['CONTRATO_PRECIO_UNITARIO'].copy()
ct.loc[mask_ct, 'CONTRATO_PRECIO_UNITARIO_CRC'] = (
    ct.loc[mask_ct, 'CONTRATO_PRECIO_UNITARIO'] * ct.loc[mask_ct, 'CONTRATO_TIPO_CAMBIO_CRC']
)
ct.rename(columns={'NRO_LINEA_CARTEL': 'NRO_LINEA'}, inplace=True)
print(f"  contrato_item: {len(ct):,} grupos")

# =====================================================================
# PASO 7: JOIN FINAL
# =====================================================================
print("\n" + "=" * 70)
print("PASO 7: JOIN FINAL")
print("=" * 70)
# ob LEFT JOIN cartel_info
result = ob.merge(
    ci[['NRO_SICOP', 'NUMERO_LINEA', 'NRO_PROCEDIMIENTO',
        'CEDULA_INSTITUCION', 'COD_UNIDAD_COMPRA', 'NOMBRE_UNIDAD_COMPRA',
        'TIPO_PROCEDIMIENTO', 'MODALIDAD_PROCEDIMIENTO', 'EXCEPCION_CD',
        'CARTEL_DESCRIPCION', 'PAGO_ADELANTADO_PYMES',
        'FECHA_PUBLICACION', 'FECHA_CIERRE_RECEPCION', 'FECHA_APERTURA',
        'CARTEL_NUMERO_PARTIDA', 'CARTEL_CODIGO_PRODUCTO',
        'CARTEL_SEGMENTO', 'CARTEL_FAMILIA', 'CARTEL_CLASE', 'CARTEL_MERCADERIA',
        'CARTEL_TIPO_MONEDA', 'CARTEL_PRECIO_UNITARIO_ESTIMADO',
        'CARTEL_TIPO_CAMBIO_USD', 'CARTEL_PRECIO_UNITARIO_CRC',
        'CARTEL_CANTIDAD_SOLICITADA', 'CARTEL_MONTO_TOTAL_CRC']],
    left_on=['NRO_SICOP', 'NRO_LINEA'],
    right_on=['NRO_SICOP', 'NUMERO_LINEA'],
    how='left'
)
result.drop(columns=['NUMERO_LINEA'], inplace=True, errors='ignore')
print(f"  Despues de cartel_info: {len(result):,}")
# LEFT JOIN item_competencia
result = result.merge(ic_counts, on=['NRO_SICOP', 'NRO_LINEA'], how='left')
print(f"  Despues de item_competencia: {len(result):,}")
# LEFT JOIN proveedores_adjudicados (para flag FUE_ADJUDICADO)
result = result.merge(
    pa, left_on=['NRO_SICOP', 'NRO_LINEA', 'CEDULA_PROVEEDOR'],
    right_on=['NRO_SICOP', 'NRO_LINEA', 'PROVEEDOR_CONTRATADO'],
    how='left', indicator='_adj'
)
result['FUE_ADJUDICADO'] = (result['_adj'] == 'both').astype(int)
result.drop(columns=['PROVEEDOR_CONTRATADO', '_adj'], inplace=True)
print(f"  Despues de proveedores_adjudicados: {len(result):,}")
# LEFT JOIN contrato_item
result = result.merge(ct, on=['NRO_SICOP', 'NRO_LINEA'], how='left')
print(f"  Despues de contrato_item: {len(result):,}")
# LEFT JOIN instituciones
result = result.merge(
    instituciones[['CEDULA', 'NOMBRE_INSTITUCION']],
    left_on='CEDULA_INSTITUCION', right_on='CEDULA', how='left'
)
result.drop(columns=['CEDULA'], inplace=True, errors='ignore')
print(f"  Despues de instituciones: {len(result):,}")
# LEFT JOIN proveedores
proveedores_cols = proveedores.rename(columns={'TAMAÑO_PROVEEDOR': 'TAMANO_PROVEEDOR'})
result = result.merge(
    proveedores_cols[['CEDULA_PROVEEDOR', 'NOMBRE_PROVEEDOR',
                      'TIPO_PROVEEDOR', 'TAMANO_PROVEEDOR']],
    on='CEDULA_PROVEEDOR', how='left'
)
print(f"  Despues de proveedores: {len(result):,}")

# =====================================================================
# PASO 8: CAMPOS CALCULADOS (window functions equivalentes)
# =====================================================================
print("\n" + "=" * 70)
print("PASO 8: CAMPOS CALCULADOS")
print("=" * 70)
# Monto total ofertado
result['OFERTA_MONTO_TOTAL_CRC'] = (
    result['OFERTA_PRECIO_UNITARIO_CRC'] * result['CANTIDAD_OFERTADA'].fillna(0)
)
# SINGLE_BID_FLAG
result['SINGLE_BID_FLAG'] = (result['N_OFERENTES_ITEM'] == 1).astype(int)
# Window functions por item: min, max, avg, stddev, rank
grp_w = result.groupby(['NRO_SICOP', 'NRO_LINEA'])['OFERTA_PRECIO_UNITARIO_CRC']
result['PRECIO_MIN_ITEM_CRC']    = grp_w.transform('min')
result['PRECIO_MAX_ITEM_CRC']    = grp_w.transform('max')
result['PRECIO_PROM_ITEM_CRC']   = grp_w.transform('mean')
result['PRECIO_STDDEV_ITEM_CRC'] = grp_w.transform('std')
# Ratios
result['RATIO_OFERTADO_VS_ESTIMADO'] = np.where(
    result['CARTEL_PRECIO_UNITARIO_CRC'].isna() | (result['CARTEL_PRECIO_UNITARIO_CRC'] == 0),
    np.nan,
    result['OFERTA_PRECIO_UNITARIO_CRC'] / result['CARTEL_PRECIO_UNITARIO_CRC']
)
result['RATIO_OFERTADO_VS_PROM_ITEM'] = np.where(
    result['PRECIO_PROM_ITEM_CRC'].isna() | (result['PRECIO_PROM_ITEM_CRC'] == 0),
    np.nan,
    result['OFERTA_PRECIO_UNITARIO_CRC'] / result['PRECIO_PROM_ITEM_CRC']
)
result['RATIO_OFERTADO_VS_MIN_ITEM'] = np.where(
    result['PRECIO_MIN_ITEM_CRC'].isna() | (result['PRECIO_MIN_ITEM_CRC'] == 0),
    np.nan,
    result['OFERTA_PRECIO_UNITARIO_CRC'] / result['PRECIO_MIN_ITEM_CRC']
)
# Rank por precio ascendente dentro del item
result['RANK_PRECIO_ASC'] = result.groupby(
    ['NRO_SICOP', 'NRO_LINEA']
)['OFERTA_PRECIO_UNITARIO_CRC'].rank(method='min', ascending=True).astype('Int64')
print("  Campos calculados completados")

# =====================================================================
# PASO 9: ORDENAR COLUMNAS
# =====================================================================
col_order = [
    # Identificadores
    'NRO_SICOP', 'NRO_PROCEDIMIENTO', 'NRO_OFERTA', 'NRO_LINEA',
    # Institucion
    'CEDULA_INSTITUCION', 'NOMBRE_INSTITUCION',
    'COD_UNIDAD_COMPRA', 'NOMBRE_UNIDAD_COMPRA',
    # Procedimiento
    'TIPO_PROCEDIMIENTO', 'MODALIDAD_PROCEDIMIENTO', 'EXCEPCION_CD',
    'CARTEL_DESCRIPCION', 'PAGO_ADELANTADO_PYMES',
    # Fechas
    'FECHA_PUBLICACION', 'FECHA_CIERRE_RECEPCION', 'FECHA_APERTURA',
    'FECHA_PRESENTA_OFERTA',
    # Cartel: producto estimado
    'CARTEL_NUMERO_PARTIDA', 'CARTEL_CODIGO_PRODUCTO',
    'CARTEL_SEGMENTO', 'CARTEL_FAMILIA', 'CARTEL_CLASE', 'CARTEL_MERCADERIA',
    'CARTEL_TIPO_MONEDA', 'CARTEL_PRECIO_UNITARIO_ESTIMADO',
    'CARTEL_TIPO_CAMBIO_USD', 'CARTEL_PRECIO_UNITARIO_CRC',
    'CARTEL_CANTIDAD_SOLICITADA', 'CARTEL_MONTO_TOTAL_CRC',
    # Oferta: proveedor
    'CEDULA_PROVEEDOR', 'NOMBRE_PROVEEDOR', 'TIPO_PROVEEDOR', 'TAMANO_PROVEEDOR',
    'OFERTA_ELEGIBLE', 'TIPO_OFERTA', 'OFERTA_ESTADO', 'ID_CONSORCIO',
    # Oferta: producto
    'OFERTA_CODIGO_PRODUCTO',
    'OFERTA_SEGMENTO', 'OFERTA_FAMILIA', 'OFERTA_CLASE', 'OFERTA_MERCADERIA',
    # Oferta: montos
    'OFERTA_TIPO_MONEDA', 'PRECIO_UNITARIO_OFERTADO', 'OFERTA_TIPO_CAMBIO_CRC',
    'OFERTA_PRECIO_UNITARIO_CRC', 'CANTIDAD_OFERTADA', 'OFERTA_MONTO_TOTAL_CRC',
    'OFERTA_IVA', 'OFERTA_DESCUENTO', 'OFERTA_OTROS_IMPUESTOS', 'OFERTA_ACARREOS',
    # Competencia
    'N_OFERTAS_ITEM', 'N_OFERENTES_ITEM', 'N_OFERENTES_ELEGIBLES_ITEM',
    'SINGLE_BID_FLAG',
    # Estadisticas de precio del item
    'PRECIO_MIN_ITEM_CRC', 'PRECIO_MAX_ITEM_CRC',
    'PRECIO_PROM_ITEM_CRC', 'PRECIO_STDDEV_ITEM_CRC',
    # Ratios
    'RATIO_OFERTADO_VS_ESTIMADO', 'RATIO_OFERTADO_VS_PROM_ITEM',
    'RATIO_OFERTADO_VS_MIN_ITEM',
    # Rank y adjudicacion
    'RANK_PRECIO_ASC', 'FUE_ADJUDICADO',
    # Contrato
    'CONTRATO_TIPO_MONEDA', 'CONTRATO_PRECIO_UNITARIO',
    'CONTRATO_PRECIO_UNITARIO_CRC', 'CONTRATO_CANTIDAD',
]
existing = [c for c in col_order if c in result.columns]
extra = [c for c in result.columns if c not in col_order]
if extra:
    print(f"\n  Columnas extra (no en orden definido): {extra}")
result = result[existing + extra]

# =====================================================================
# PASO 10: GUARDAR
# =====================================================================
print("\n" + "=" * 70)
print("PASO 10: GUARDAR")
print("=" * 70)
out_path = OUT / 'V_BASE_OFERTA_ITEM.parquet'
try:
    result.to_parquet(out_path, index=False, engine='pyarrow')
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Guardado: {out_path}")
    print(f"  Formato:  Parquet")
    print(f"  Tamano:   {size_mb:.1f} MB")
except ImportError:
    out_csv = OUT / 'V_BASE_OFERTA_ITEM.csv'
    result.to_csv(out_csv, index=False)
    size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"  Guardado: {out_csv}")
    print(f"  Formato:  CSV (pyarrow no disponible)")
    print(f"  Tamano:   {size_mb:.1f} MB")

# =====================================================================
# VERIFICACION
# =====================================================================
print("\n" + "=" * 70)
print("VERIFICACION FINAL")
print("=" * 70)
print(f"  Filas:              {len(result):,}")
print(f"  Columnas:           {result.shape[1]}")
print(f"  NRO_SICOP unicos:   {result['NRO_SICOP'].nunique():,}")
pk_dupes = result.duplicated(subset=['NRO_SICOP', 'NRO_OFERTA', 'NRO_LINEA']).sum()
print(f"  Duplicados en PK:   {pk_dupes:,}")
print(f"  Ofertas adjudicadas:{(result['FUE_ADJUDICADO']==1).sum():,} ({(result['FUE_ADJUDICADO']==1).mean()*100:.1f}%)")
print(f"  Single-bid:         {(result['SINGLE_BID_FLAG']==1).sum():,} ({(result['SINGLE_BID_FLAG']==1).mean()*100:.1f}%)")
has_cartel = result['CARTEL_PRECIO_UNITARIO_CRC'].notna().sum()
print(f"  Con precio cartel:  {has_cartel:,} ({has_cartel/len(result)*100:.1f}%)")
has_contrato = result['CONTRATO_PRECIO_UNITARIO_CRC'].notna().sum()
print(f"  Con precio contrato:{has_contrato:,} ({has_contrato/len(result)*100:.1f}%)")
print(f"\n  Columnas finales:")
for c in result.columns:
    nn = result[c].notna().sum()
    print(f"    {c:<42s} {str(result[c].dtype):<12s} ({nn:>10,} no-null)")
print("\nListo.")
