# Model Card — ElasticNetCV
Fecha: 2025-09-10 11:38

## Objetivo
Predecir SalePrice (House Prices — Kaggle). Entrenado en log1p(SalePrice).

## Datos
- Train limpio: data/processed/train_clean.csv (sin nulos)
- Test limpio : data/processed/test_clean.csv (sin nulos)
- Outliers removidos en train (2 casos en GrLivArea)

## Preprocesamiento
- OrdinalEncoder en: ['ExterQual', 'BsmtQual', 'KitchenQual', 'FireplaceQu', 'GarageFinish']
- OneHotEncoder en nominales; StandardScaler en numéricas
- Log1p en features sesgadas (según reports/skewed_feats.txt)
- MSSubClass como categórica (string)

## Validación y métrica
- CV=5 (shuffle=True, seed=42)
- Métrica: RMSE en log (≈ RMSLE)
- Resultados CV (resumen): [{'modelo': 'ElasticNetCV', 'rmse_log_mean': 0.11207245054497705, 'rmse_log_std': 0.0069430207362930745}, {'modelo': 'LassoCV', 'rmse_log_mean': 0.11273930810269772, 'rmse_log_std': 0.007374203726917119}, {'modelo': 'RidgeCV', 'rmse_log_mean': 0.11314633994117793, 'rmse_log_std': 0.007695939420668351}, {'modelo': 'LinearRegression', 'rmse_log_mean': 0.13500278731186446, 'rmse_log_std': 0.018568671809526656}]

## Modelo
- ElasticNetCV (ElasticNetCV)
- Parámetros: ver models/metadata_ElasticNetCV.json

## Limitaciones y supuestos
- Distribución y relaciones condicionadas por el dataset original
- Transformaciones log1p pueden afectar interpretabilidad directa
- One-Hot puede generar alta dimensionalidad

## Uso previsto
- Benchmark / baseline reproducible para la competencia
- Base para mejorar con regularización/end-to-end/boosting
