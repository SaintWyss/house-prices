# House Prices: EDA y Baseline Model

Repositorio para el proyecto de predicción de precios de casas del concurso **House Prices - Kaggle**. Contiene los pasos para limpiar los datos, explorar el conjunto, entrenar un modelo base y generar una submission reproducible.

## Estructura del repositorio

```
house-prices/
├── data/
│   └── processed/            # CSV limpios listos para modelar
├── notebooks/                # Análisis exploratorio y modelo baseline
├── reports/
│   ├── figures/              # Gráficos generados durante el EDA
│   └── model_card_ElasticNetCV.md
├── submissions/              # Predicción preparada para Kaggle
├── environment.yml           # Entorno conda recomendado
├── requirements.txt          # Alternativa con dependencias mínimas
└── LICENSE
```

## Cómo reproducir el entorno

Con conda:

```bash
conda env create -f environment.yml
conda activate house-prices
```

O bien con pip:

```bash
pip install -r requirements.txt
```

## Datos

Los archivos originales de Kaggle (**train.csv** y **test.csv**) no se versionan. Descárgalos y colócalos en `data/raw/` si quieres rehacer todo el flujo. El repositorio ya incluye versiones limpias listas para modelar:

- `data/processed/train_clean.csv`
- `data/processed/test_clean.csv`

## Notebooks principales

1. **`notebooks/eda-house-prices.ipynb`** – análisis exploratorio, imputación de nulos y tratamiento de variables.
2. **`notebooks/modeling-baseline.ipynb`** – pipeline con preprocesamiento y modelos lineales con validación cruzada.

## Resultados

El mejor modelo baseline fue **ElasticNetCV**, con un **RMSE log de ~0.112** en validación cruzada. El archivo `submissions/submission_baseline.csv` contiene la predicción generada a partir de ese modelo.

Gráficos relevantes del EDA se encuentran en `reports/figures/`.

## Cómo contribuir

1. Crea un fork del proyecto.
2. Trabaja en una rama basada en `main`.
3. Asegúrate de que los notebooks y artefactos generados se guarden dentro de las carpetas existentes.

## Licencia

Este proyecto se distribuye bajo los términos de la licencia MIT. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.
