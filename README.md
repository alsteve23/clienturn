# Clien↩urn — retención predictiva de clientes

*Antes de que crucen la puerta.*

Solución de Machine Learning para que el equipo comercial de un **operador regional de internet y cable** sepa **a quién contactar, a cuántos, por qué está en riesgo cada cliente y qué ofrecerle**, a partir del dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

🔗 **App desplegada:** [CLIENTURN](https://clienturn.streamlit.app/)

## Resultados principales (test: 1.409 clientes nunca vistos)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.843 |
| Recall / Precision (umbral 0.58) | 0.717 / 0.555 |
| F1 | 0.625 (CV: 0.636) |
| Precision en el 10 % de mayor riesgo | 74 % (lift 2.8×) |

- **Modelo:** Random Forest dentro de un `Pipeline` (imputación, escalado y one-hot), comparado con Regresión Logística y XGBoost.
- **Optimización:** Grid Search vs Random Search vs TPE (Optuna), con el mismo presupuesto (48 configuraciones) y validación cruzada estratificada de 5 folds.
- **Multiobjetivo:** NSGA-II maximizando recall y precision (con el umbral como variable); se eligió la solución que contacta al 34 % de los clientes conservando el 95 % del valor neto máximo.
- **Explicabilidad:** SHAP (global y local), LIME y comparación con la feature importance del Random Forest y permutation importance.

> Los valores monetarios del proyecto (costo de la promoción, tasa de éxito, precios) son **supuestos hipotéticos** declarados como tales.

## Estructura

```
├── Churn_Datos.ipynb   # notebook: EDA, modelos, optimización, NSGA-II, test, SHAP/LIME, propuesta de negocio
├── app.py              # app Streamlit (solo inferencia, no reentrena)
├── test_app.py         # prueba de humo de la app
├── models/             # pipelines exportados (.joblib)
├── artifacts/          # resultados del notebook que consume la app
├── data/               # dataset original
└── requirements.txt    # versiones fijadas (las mismas del entrenamiento)
```

## Ejecución local

Requiere **Python 3.12**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Para reproducir el notebook: `pip install jupyter` y ejecutar `Churn_Datos.ipynb` completo (≈ 15 min). Regenera `models/` y `artifacts/`.

Prueba de la app: `python test_app.py`

## Secciones de la app

1. **Exploración de datos:** filtros por segmento, KPIs y gráficos interactivos.
2. **Desempeño del modelo:** comparación de modelos, optimizadores, frente de Pareto, métricas de test y matriz de confusión.
3. **Predicción y explicación:** formulario de cliente → score de riesgo, decisión, explicación SHAP/LIME en lenguaje simple y acción de retención sugerida.
4. **Aplicación de negocio:** planificador de campaña (costo y efectividad de la promoción, capacidad de contacto), valor neto frente a contactar al azar o a todos, ROI por tamaño de operador, lista descargable y análisis de una cartera propia (CSV).
