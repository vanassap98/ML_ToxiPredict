# ToxiPredict  
## Anticipating Acute Aquatic Toxicity Using Machine Learning  
## Predicción de Toxicidad Acuática Aguda mediante Aprendizaje Automático

> 📄 This document is available in both **English** and **Spanish**.  
> 📄 Este documento está disponible en **inglés** y **español**.

---

## 🇬🇧 English Version

### Project Overview

Chemical manufacturers must assess the ecological impact of their compounds before regulatory approval. One of the key requirements is evaluating **acute aquatic toxicity**, typically through costly and ethically sensitive bioassays.

**ToxiPredict** is a machine learning project designed to predict the acute toxicity of chemical substances on aquatic species — particularly fish — based on their chemical, taxonomic, and experimental properties. This helps:

- Prioritize which compounds should undergo lab testing  
- Reduce regulatory costs and approval delays  
- Minimize the use of animal testing and enhance sustainability

This project was developed as part of the **Data Science Bootcamp - The Bridge (2025)**.



### Dataset

The dataset comes from **ADORE** (A Data-driven benchmark fOR Ecotoxicology), published by Schür et al. (2023) in *Scientific Data* (Nature). It includes:

- Molecular descriptors and chemical fingerprints  
- Exposure conditions (media, duration)  
- Taxonomic data  
- Binary labels indicating toxic (1) or non-toxic (0) outcomes  

For this project, we use the `a-F2F_mortality.csv` subset: a benchmark challenge focused on predicting toxicity in fish based on historical fish test data.

📚 Reference:  
Schür C, et al. (2023). *A benchmark dataset for machine learning in ecotoxicology.* Scientific Data. https://doi.org/10.1038/s41597-023-02612-2

⚠️ Due to GitHub limitations, the complete raw dataset (t-F2F_mortality.csv) is not included in this repository, as it exceeds the 100 MB file size limit.
A sample dataset is provided instead in src/data_sample/sample_t-F2F_mortality.csv to ensure reproducibility and structure demonstration.

### Objectives

- Perform a structured EDA on ecotoxicological features  
- Train classification models to predict toxicity  
- Apply validation techniques and evaluate performance  
- Interpret the model using feature importance and SHAP  
- Deliver a clean notebook and business-oriented presentation



### Solution Summary

Several supervised classification models were evaluated, including Random Forest and Logistic Regression. After comparison and cross-validation, the final model selected was a **Logistic Regression classifier**, which provided robust performance and interpretability. The model achieved an **F1-score of 0.715** on the held-out test set and was further interpreted using SHAP values.




### How to Run

1. Clone this repo and create a virtual environment  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the pipeline in main.ipynb




### Repository Structure / Estructura del Repositorio

├── main.ipynb            # Final notebook: complete pipeline and model execution
├── README.md             # Project summary and instructions (EN/ES)
├── requirements.txt      # Dependencies
├── ML_ToxiPredict_Presentacion.pdf  # Presentation file (PDF)

├── data/                 # Raw dataset (t-F2F_mortality.csv)
├── models/               # Trained model files (.joblib/.pkl)
├── src/
│   ├── data_sample/      # Public data sample (<5MB)
│   ├── img/              # Visual assets used in notebooks
│   ├── models/           # Saved models during training
│   ├── notebooks/        # Exploratory notebooks and modular workflow
│   ├── utils/            # Python modules (eda.py, toolbox_ML.py)


### License and Acknowledgments
This project reuses data under CC BY 4.0.
Dataset authors: Schür et al. (2023) – ADORE Benchmark.

---

## 🇪🇸 Versión en Español
### Descripción del Proyecto
Los fabricantes de sustancias químicas deben evaluar el impacto ecológico de sus compuestos antes de su aprobación regulatoria. Una de las pruebas más críticas es la de toxicidad acuática aguda, que suele implicar ensayos costosos y éticamente delicados con organismos vivos.

ToxiPredict es un proyecto de aprendizaje automático cuyo objetivo es predecir la toxicidad aguda de sustancias químicas sobre especies acuáticas —especialmente peces— a partir de sus propiedades químicas, taxonómicas y experimentales. Esto permite:

- Priorizar qué compuestos deben testearse en laboratorio
- Reducir costes regulatorios y tiempos de aprobación
- Minimizar el uso de ensayos con animales y mejorar la sostenibilidad

Este proyecto ha sido desarrollado como parte del Bootcamp de Data Science de The Bridge (2025).

### Origen del Dataset
Los datos utilizados provienen de ADORE (A Data-driven benchmark fOR Ecotoxicology), publicado por Schür et al. (2023) en la revista Scientific Data (Nature). El benchmark incluye:

- Descriptores moleculares y huellas químicas
- Condiciones de exposición (medio, duración)
- Información taxonómica
- Etiquetas binarias indicando si el compuesto es tóxico (1) o no tóxico (0)

Para este proyecto se ha utilizado el subset a-F2F_mortality.csv, centrado en predecir toxicidad en peces a partir de datos históricos en la misma clase taxonómica.

📚 Referencia:
Schür C, et al. (2023). A benchmark dataset for machine learning in ecotoxicology. Scientific Data. https://doi.org/10.1038/s41597-023-02612-2

⚠️ Debido a las limitaciones de GitHub, el dataset completo (t-F2F_mortality.csv) no está incluido en este repositorio, ya que excede el límite de tamaño de archivo (100 MB).
En su lugar, se proporciona una muestra representativa en src/data_sample/sample_t-F2F_mortality.csv, útil para replicar la estructura y el flujo del proyecto.

### Objetivos
- Realizar un EDA estructurado sobre las variables ecotoxicológicas
- Entrenar modelos de clasificación para predecir toxicidad
- Aplicar técnicas de validación y evaluación
- Interpretar el modelo usando importancia de variables y SHAP
- Entregar un notebook limpio y una presentación orientada a negocio

### Cómo ejecutar
1. Clonar el repositorio y crear un entorno virtual
2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar todo el pipeline desde main.ipynb (en la raíz del proyecto)


### Resumen de la Solución

Se evaluaron varios modelos de clasificación supervisada, incluyendo Random Forest, XGBoost y Regresión Logística. Tras comparar su rendimiento con validación cruzada, se seleccionó como modelo final una Regresión Logística, por su equilibrio entre rendimiento y facilidad de interpretación. El modelo alcanzó una F1-score de 0.715 sobre el conjunto de test, y fue interpretado utilizando valores SHAP.


