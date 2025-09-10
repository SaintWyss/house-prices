# Proyecto: House Prices (Kaggle) — EDA completo y datos listos para modelado

## 1) Objetivo del trabajo

Dejar un **dataset limpio, reproducible y sin fugas** para entrenar modelos de precios de viviendas. El foco es la **Exploratory Data Analysis (EDA)** con decisiones documentadas: inspección, nulos, target, relaciones, outliers, asimetría y exportación de datos procesados.

**Entrega principal de esta fase**

* `data/processed/train_clean.csv` (1458 × 81, sin nulos)
* `data/processed/test_clean.csv`  (1459 × 80, sin nulos)
* Figuras clave en `reports/figs/`
* Reportes auxiliares en `reports/`

---

## 2) Estructura del proyecto

```
house-prices/
  data/
    raw/           # CSV originales de Kaggle: train.csv, test.csv
    processed/     # CSV limpios exportados
  notebooks/
    eda-house-prices.ipynb
  reports/
    figs/          # imágenes generadas durante el EDA
    skew_report.csv
    skewed_feats.txt
    bitacora.md    # decisiones y hallazgos
  models/
  src/
```

**Criterios de organización**

* **Rutas relativas** dentro del repo (evita mezclar Windows/WSL).
* **Idempotencia**: crear carpetas antes de guardar artefactos.
* **Trazabilidad**: todo gráfico/decisión importante queda registrado.

---

## 3) Configuración de entorno (VSCode + WSL)

* Proyecto abierto en **VSCode – Remote WSL** (Ubuntu), para trabajar en Linux desde Windows.
* **Jupyter Server**: fijado en **Local** (se limpiaron URIs guardadas de servidores anteriores).
* **Kernel único**: notebook apuntando a un único entorno conda de WSL (evitando duplicados de “Existing Jupyter Server”).
* Extensiones: Python + Jupyter (Copilot opcional).

**Motivo**: evitar inconsistencias de intérprete/servidor y asegurar reproducibilidad de rutas y dependencias.

---

## 4) Carga de datos y verificación inicial

* CSV colocados en `data/raw/`.
* Lectura de `train.csv` (1460 × 81) y `test.csv` (1459 × 80).
* Chequeos básicos: `Id` único, `SalePrice` solo en `train`.

**Motivo**: validar integridad de entradas y precondiciones de la competencia.

---

## 5) Inspección inicial (tipos, cardinalidad, num→cat)

* Conteo por tipo, separación **numéricas** vs **categóricas**.
* Detección de columnas constantes y duplicados (no encontrados).
* **MSSubClass** marcada como **categórica** (aunque venga numérica) por su significado semántico (clase de vivienda).

**Motivo**: tener una radiografía clara para definir estrategias de imputación y encoding.

---

## 6) Nulos: cuantificación y nulos estructurales

* Reporte de % de nulos en train/test y visual Top-20.
* Agrupación por familias para detectar **nulos estructurales** (si toda la familia está NA ⇒ la propiedad *no tiene* esa característica):

  * Garage\*, Bsmt\*, FireplaceQu, Pool\*, MasVnr\*
  * Columnas unitarias: Alley, Fence, MiscFeature

**Decisión de imputación (sin leakage):**

* **Estructurales**: categóricas → `"None"` (guardado como `None_` en CSV), numéricas → `0`.
* **LotFrontage**: **mediana por Neighborhood** (fallback mediana global).
* **Resto numéricas**: mediana **global** (ajustada en `train_wo_outliers`).
* **Resto categóricas**: **moda** global (ajustada en `train_wo_outliers`).

**Motivo**: coherencia semántica y robustez; todo se aprende en train y se aplica igual en test (anti-fugas).

---

## 7) Variable objetivo (SalePrice)

* Distribución original con **sesgo alto** (asimetría positiva).
* Transformación **log1p(SalePrice)** redujo notablemente el sesgo (histogramas y QQ-plots opcionales).

**Decisión**: **modelar el target en log** y aplicar `expm1` para interpretar/enviar a Kaggle.

**Motivo**: mejorar supuestos/estabilidad de modelos y la métrica (equivalente a trabajar con RMSLE).

---

## 8) Correlaciones numéricas y multicolinealidad

* Top correlaciones con `SalePrice` (y con `SalePrice_log`).
* Heatmap de las top-N con el target.
* Observaciones típicas: `OverallQual`, `GrLivArea`, `GarageCars/Area`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `YearBuilt`, `TotRmsAbvGrd`.
* **Pares colineales** esperables (p. ej. `GarageCars`~~`GarageArea`, `1stFlrSF`~~`TotalBsmtSF`).

**Motivo**: priorizar variables fuertes y anticipar regularización/selección para lineales.

---

## 9) Categóricas: ordenamientos y efectos

* Orden por **mediana del precio** (Neighborhood, KitchenQual).
* **Boxplots** de KitchenQual/OverallQual mostraron relación monotónica con el target.
* **MSSubClass** tratada como categórica.

**Notas**

* Categóricas **ordinales** (p. ej. ExterQual, BsmtQual, KitchenQual, FireplaceQu, GarageFinish) conservarán orden en el encoding.
* Categóricas **nominales** con muchas categorías: considerar **agrupar raras** o usar **Target/WOE encoding** con *k-fold* (más adelante).

**Motivo**: guiar decisiones de codificación para el pipeline de modelado.

---

## 10) Outliers

* Scatter `GrLivArea` vs target: 2 outliers clásicos.
* **Regla aplicada**: `GrLivArea > 4000` & `SalePrice < 300000` ⇒ **removidos 2 casos** (índices: **\[523, 1298]**).
* `train` pasó a **`train_wo_outliers` (1458 × 82)**.
* Chequeo IQR en variables clave **solo diagnóstico**, sin remover masivamente.

**Motivo**: esos puntos distorsionan fuertemente modelos lineales; se documentan y se elimina **solo en train**.

---

## 11) Skew (asimetría) en features numéricas

* Cálculo de asimetría (base: `train_wo_outliers`), reporte ordenado.
* Umbral adoptado: **0.75**.
* Se generó `reports/skewed_feats.txt` con **20 columnas** candidatas a **`log1p`** (p. ej., `LotArea`, `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `MasVnrArea`, porches/decks, etc.).
* **Exclusiones**: ordinales fuertes (p. ej., `OverallQual`, `OverallCond`) y no-numéricas (`MSSubClass` tratada como string).

**Motivo**: reducir colas largas, mejorar linealidad y estabilidad de entrenamiento.

---

## 12) Imputación y transformaciones (aplicación sin fugas)

**Ajuste** (solo con `train_wo_outliers`):

* Mapas para estructurales: cat→`None` (luego guardado como `None_`), num→0.
* `LotFrontage`: medianas por `Neighborhood` (25 vecindarios) + fallback global.
* Medianas globales para numéricas restantes; modas globales para categóricas restantes.

**Aplicación**

* Se aplicaron a una copia de `train` (previa remoción de outliers) y a `test`.
* Transformación **`log1p`** a las columnas numéricas listadas en `skewed_feats.txt` (solo si son numéricas; se omiten no-numéricas y se informa).

**Verificaciones**

* Nulos totales: **0** en `train_clean` y **0** en `test_clean`.
* Columnas alineadas (salvo `SalePrice`, presente solo en train).

**Motivo**: dejar datos consistentes y comparables para cualquier modelo, evitando **data leakage**.

---

## 13) Exportación robusta y lectura segura

* `train_clean.csv` y `test_clean.csv` guardados en `data/processed/`.
* **Blindaje**: reemplazo literal `"None" → "None_"` en columnas de texto **antes de guardar** para que ningún `read_csv` vuelva a interpretarlo como NA.
* Verificación re-leyendo desde disco con `keep_default_na=False` (nulos = 0), shapes esperadas y columnas alineadas.

**Motivo**: asegurar interoperabilidad futura con cualquier lector y evitar falsos nulos.

---

## 14) Artefactos generados

* **Datos**: `data/processed/train_clean.csv`, `data/processed/test_clean.csv`.
* **Figuras** (ejemplos):

  * `hist_saleprice.png`, `hist_saleprice_log.png`
  * `heatmap_top_corr.png`
  * `bar_median_by_neighborhood.png`
  * `boxplot_kitchenqual.png`, `boxplot_overallqual.png`
  * `scatter_grlivarea_vs_target.png`
* **Reportes**: `reports/skew_report.csv`, `reports/skewed_feats.txt`.
* **Bitácora**: `reports/bitacora.md` (decisiones y justificaciones).

---

## 15) Principios que guiaron el trabajo

* **Anti-fugas**: todo ajuste (imputación, mapeos, etc.) se aprende **solo** con `train` y se aplica en test.
* **Idempotencia**: celdas que crean carpetas/archivos no fallan si existen.
* **Reproducibilidad**: rutas relativas y un **kernel único** en WSL.
* **Trazabilidad**: índices removidos, listas de columnas transformadas y figuras guardadas.

---

## 16) Próximos pasos (baseline profesional)

1. **Notebook** `notebooks/modeling-baseline.ipynb`.
2. Definir **X** = todas las columnas de `train_clean` menos `SalePrice`; **y** = `log1p(SalePrice)` en memoria.
3. **Preprocesamiento** con `ColumnTransformer`:

   * Numéricas: `passthrough` (opcional `StandardScaler` para lineales).
   * **Ordinales** con mapeos explícitos (p. ej., `Ex > Gd > TA > Fa > Po > None_`).
   * **Nominales**: `OneHotEncoder(handle_unknown='ignore')`.
4. **Modelos** (CV=5): `LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`; luego `RandomForestRegressor` y `GradientBoostingRegressor` (XGB/LGBM opcional).
5. **Métrica**: **RMSE sobre log** (≈ RMSLE). Reportar también RMSE en escala original (`expm1`).
6. **Entrega**: tabla de resultados CV + `submissions/submission_baseline.csv` (predicciones `expm1` sobre `test_clean`).

---

## 17) Resumen ejecutivo

* EDA completo con decisiones fundamentadas.
* Datos **sin nulos**, target y features **transformados** donde corresponde.
* **Outliers** documentados y removidos en `train`.
* Exportables listos para **modelado reproducible**.
* Próximo paso: **pipeline + CV** para baselines confiables.

# Proyecto: House Prices (Kaggle) — Modelado Baseline (PASO 14–17)

## Resumen de esta fase

Objetivo: pasar de datos ya limpios a un **baseline reproducible** con **métrica y validación confiables**, generar una **submission** y dejar **trazabilidad** (modelo y metadatos guardados).

**Entregables principales**

* `notebooks/modeling-baseline.ipynb` (pipeline, CV, resultados y predicciones)
* `submissions/submission_baseline.csv` (+ variante con timestamp)
* (sugerido) `models/baseline_{Modelo}.joblib`, `models/metadata_{Modelo}.json`, `models/feature_names_{Modelo}.csv`, `reports/model_card_{Modelo}.md`

---

## 14) Setup del notebook de baseline

**Qué hicimos**

* Creamos `notebooks/modeling-baseline.ipynb` y cargamos `data/processed/train_clean.csv` y `test_clean.csv` (ambos sin nulos).
* Definimos **X** (todas las columnas salvo `SalePrice`) y **y\_log = log1p(SalePrice)\`** para modelar y evaluar en escala log.
* Fijamos la **métrica** como **RMSE en log** (≈ RMSLE) y la **validación** como **KFold=5** con shuffle y semilla fija (42).

**Por qué**

* Usar el target en log estabiliza la distribución (decisión tomada en el EDA) y mejora la comparabilidad del error.
* CV=5 con shuffle da una estimación honesta del rendimiento esperado fuera de la muestra.

---

## 15) Preprocesamiento y baselines con CV

**Diseño del preprocesamiento (dentro del Pipeline)**

* **Numéricas**: ya imputadas/transformadas en el EDA; se pasan tal cual. Se incluyó **StandardScaler** para estabilizar los modelos lineales.
* **Ordinales textuales**: (p. ej., `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, `GarageFinish`) con **OrdinalEncoder** y orden explícito (`Ex > Gd > TA > Fa > Po > None_`).
* **Nominales**: resto de columnas `object` con \*\*OneHotEncoder(handle\_unknown='ignore')\`.
* **MSSubClass**: tratada como **categórica** (string), no como numérica.

**Métrica y validación**

* Se evaluaron **LinearRegression**, **RidgeCV**, **LassoCV** y **ElasticNetCV** con **CV externo (5 folds)**.
* Para Lasso y ElasticNet se usó **CV interna** para elegir hiperparámetros (proceso anidado y sin fugas). Ridge se evaluó con conjunto de alphas log-espaciados.

**Compatibilidad (incidencias y resolución)**

* Cambio de API en scikit‑learn:

  * `OneHotEncoder`: se reemplazó `sparse=False` por `sparse_output=False` en versiones nuevas (detección automática por firma).
  * `RidgeCV`: se eliminó el argumento `store_cv_values`; se removió para compatibilidad.

**Resultado esperado**

* Tabla con **RMSE\_log (media ± std)** por modelo. Ridge/Lasso/ElasticNet suelen mejorar o empatar a Linear. Se elige el **mejor provisional** por menor RMSE\_log medio.

**Por qué**

* Encapsular preprocesamiento en el **Pipeline** evita fugas y asegura que CV mida exactamente el flujo que se usará al desplegar.

---

## 16) Entrenamiento final y submission

**Qué hicimos**

* Reentrenamos el **mejor pipeline** en **todo `train_clean`** (con y en log).
* Generamos predicciones en log sobre **`test_clean`** y las **volvimos a escala original** con `expm1`.
* Armamos la **submission** con columnas `Id` y `SalePrice`; guardamos en:

  * `submissions/submission_baseline.csv`
  * `submissions/submission_baseline_{Modelo}_{timestamp}.csv`
* Chequeos: sin NaN, 1459 filas, valores positivos razonables.

**Por qué**

* Entrenar con todos los datos maximiza la información disponible antes de predecir el set de test de Kaggle.

---

## 17) Persistencia y trazabilidad (sugerido y recomendado)

**Qué dejamos listo para guardar**

* **Pipeline entrenado**: `models/baseline_{Modelo}.joblib` (prepro + modelo en un único artefacto).
* **Metadatos**: `models/metadata_{Modelo}.json` (versiones de Python/sklearn/numpy, seed, fecha/hora, métrica CV, hiperparámetros elegidos, columnas usadas, mapping ordinal, etc.).
* **Nombres de features** pos‑transformación: `models/feature_names_{Modelo}.csv` (útil para inspección de coeficientes/importancias).
* **Model Card**: `reports/model_card_{Modelo}.md` (objetivo, datos, prepro, métrica, limitaciones, uso previsto, riesgos conocidos).

**Por qué**

* Garantiza **reproducibilidad** y facilita depuración, auditoría y comunicación del modelo.

---

## Checklist de calidad (baseline)

* [x] **Anti‑fugas**: todo preprocesamiento dentro del Pipeline; la CV evalúa el flujo real.
* [x] **Reproducibilidad**: kernel único (WSL), semilla fija, rutas relativas.
* [x] **Datos**: `train_clean`/`test_clean` sin nulos; `Id` preservado.
* [x] **Métrica**: RMSE en log; comparable con notebooks de referencia.
* [x] **Submission**: `Id`, `SalePrice`, 1459 filas, sin NaN, rango positivo.
* [ ] **Persistencia**: artefactos del modelo guardados y verificados (cargar y predecir un batch pequeño).

---

## Próximos pasos sugeridos

1. **Regularización y búsqueda fina**: ampliar grillas/espacios para Ridge/Lasso/EN y comparar.
2. **Árboles/Boosting**: probar `RandomForest` y `GradientBoosting` (y luego XGBoost/LightGBM) con el mismo preprocesamiento.
3. **Análisis de errores**: estudiar residuales por barrio/calidad/superficies; detectar sesgos sistemáticos.
4. **Stacking simple**: combinar lineales + árboles para reducir varianza.
5. **Explainability**: coeficientes (lineales) e importancias (árboles), con nombres de features pos‑transformación.
6. **Governance**: completar Model Card y versionar artefactos en `models/`.
