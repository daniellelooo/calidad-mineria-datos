# Práctica 4 — Calidad y Minería de Datos en Python
**Curso:** Analítica de Datos | **Universidad:** UPB Medellín | **2026**

Dataset trabajado: **Escuela de Talentos** (Indeportes/IDER) — inversión por deportista en programas olímpicos y paralímpicos entre 2022 y 2024.

---

## Archivos del repositorio

| Archivo | Descripción |
|---|---|
| `Escuela_Talento_limpio.arff` | Dataset original en formato Weka |
| `escuela_talento_limpio.csv` | Dataset post-limpieza generado por el Notebook 1 |
| `01_calidad_datos.ipynb` | Notebook de calidad de datos |
| `02_modelo_predictivo.ipynb` | Notebook de modelado y GridSearch |
| `03_despliegue_gui.ipynb` | Notebook de despliegue con interfaz gráfica |
| `perfilado_original.html` | Reporte de perfilado automático (ydata-profiling) |
| `modelo_final.joblib` | Modelo serializado listo para producción |
| `app_tkinter.py` | Aplicación de escritorio standalone (Tkinter) |

---

## Notebooks

### 01 — Calidad de Datos
Perfilado automático del dataset original con `ydata-profiling` y diagnóstico de las seis dimensiones de calidad (completitud, unicidad, validez, consistencia, exactitud y oportunidad). Aplica los pasos de limpieza: strip de espacios, unificación de categorías sinónimas (`NINGUNA → NO APLICA`) y conversión de tipo en `VIGENCIA`. Exporta el CSV limpio.

### 02 — Modelo Predictivo
Predice el **Sector** (OLIMPICO vs PARALIMPICO) de un deportista. Compara cinco algoritmos — Árbol de Decisión, KNN, Red Neuronal (MLP), SVM y Random Forest — usando F1 sobre la clase minoritaria como métrica principal (el dataset tiene 95/5 de desbalance). El mejor modelo (Árbol de Decisión, F1 = 0.80) se hiperparametriza con `GridSearchCV` y se serializa como `modelo_final.joblib`.

### 03 — Despliegue con GUI
Carga el modelo serializado y expone una ventana de escritorio con Tkinter donde el usuario ingresa vigencia, grupo etario e inversión, y obtiene la predicción del sector junto con las probabilidades. También exporta el script `app_tkinter.py` para ejecución standalone.

---

## Resultados del modelo

| Modelo | Accuracy | F1 (PARALIMPICO) | ROC AUC |
|---|---|---|---|
| **Árbol de Decisión** ✅ | 0.9836 | **0.8021** | 0.9520 |
| KNN | 0.9832 | 0.7958 | 0.8420 |
| Red Neuronal (MLP) | 0.9832 | 0.7958 | 0.9429 |
| Random Forest | 0.9832 | 0.7958 | 0.9492 |
| SVM | 0.9828 | 0.7917 | 0.9078 |

---

## Cómo ejecutar la app

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
pip install joblib scikit-learn pandas numpy
python3 app_tkinter.py
```

> En Mac con Python 3.14+ instalar `brew install python-tk@3.13` y usar `python3.13`.

---

## Dependencias principales
`pandas` · `scikit-learn` · `ydata-profiling` · `scipy` · `matplotlib` · `seaborn` · `joblib`
