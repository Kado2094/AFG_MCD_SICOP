# Detección no supervisada de anomalías en la contratación pública de Costa Rica

Este repositorio reúne los principales artefactos computacionales del Trabajo Final de Graduación sobre detección y priorización de ofertas atípicas en datos del Sistema Integrado de Compras Públicas de Costa Rica (SICOP).

El flujo parte de una base analítica con granularidad oferta–ítem, construida a partir de información de carteles, ofertas, contratos, instituciones y proveedores. Sobre esta base se generan variables relacionadas con precios, competencia, temporalidad y clasificación UNSPSC, que posteriormente se utilizan para ajustar modelos no supervisados.

El método principal es Isolation Forest, aplicado por segmento y clase de producto. La interpretación de los resultados se realiza mediante SHAP, mientras que la robustez del enfoque se contrasta con otros métodos de detección de anomalías y con reglas heurísticas. Los resultados deben interpretarse como un mecanismo de priorización para revisión, no como evidencia concluyente de irregularidad o fraude.

## Lógica computacional y archivos principales

| Etapa | Lógica principal | Archivos base |
|---|---|---|
| Base analítica | Unir oferta, ítem, cartel, proveedor e institución; convertir precios a CRC; calcular competencia, estadísticas y ratios. | `_build_oferta_item.py` |
| Modelo principal | Transformar variables, ajustar modelos por segmento y clase UNSPSC, combinar puntajes y aplicar percentiles. | `IF_SICOP_v2.ipynb` |
| Explicabilidad | Calcular valores SHAP, agrupar contribuciones y caracterizar tipologías dominantes de anomalía. | `IF_SICOP_shap.ipynb` |
| Contrastes | Comparar Isolation Forest con HBOS, ECOD y LOF; calcular intersecciones, consenso entre métodos, reglas heurísticas y validación temporal. | `benchmark_metodos.ipynb` |
| Evidencia del informe | Calcular tamaños de efecto, correlaciones y análisis estadísticos complementarios. | `analisis_complementario.py` |
| Figuras del informe | Generar las visualizaciones verificables utilizadas en el informe final. | `generar_figuras_informe.py` |

## Flujo general

```text
Datos abiertos de SICOP
        │
        ▼
Construcción de la base oferta–ítem
        │
        ▼
Ingeniería y transformación de variables
        │
        ▼
Isolation Forest por segmento y clase UNSPSC
        │
        ├──► Explicabilidad con SHAP
        │
        ├──► Comparación con HBOS, ECOD, LOF y reglas heurísticas
        │
        └──► Evidencia estadística y figuras del informe
```

## Estructura

```text
AFG_MCD_SICOP/
├── README.md
└── AFG III/
    └── Anexos/
        ├── _build_oferta_item.py
        ├── IF_SICOP_v2.ipynb
        ├── IF_SICOP_shap.ipynb
        ├── benchmark_metodos.ipynb
        ├── analisis_complementario.py
        └── generar_figuras_informe.py
```

## Alcance

Este repositorio publica el código y los notebooks principales del análisis. Los datos originales y algunos artefactos intermedios no se incluyen, por lo que la reproducción completa requiere reconstruir la estructura de datos esperada por cada archivo.
