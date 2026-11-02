"""Clien↩urn — retención predictiva para operadores regionales.

Solo inferencia: carga el pipeline exportado desde el notebook (models/) y los
resultados de los experimentos (artifacts/). No reentrena nada.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title='Clien↩urn', page_icon='↩', layout='wide')

ART = 'artifacts/'
COSTO_CLIENTURN_ANIO1 = 1500 + 3000 + 9 * 800  # piloto + implementación + 9 meses de suscripción (hipotético)
ETIQUETA = {0: 'Se queda', 1: 'Se va'}
COLOR = {'Se queda': '#4C78A8', 'Se va': '#E45756'}

VAR_ES = {
    'Contract': 'Tipo de contrato', 'InternetService': 'Servicio de internet', 'tenure': 'Antigüedad',
    'PaymentMethod': 'Método de pago', 'MonthlyCharges': 'Cargo mensual', 'TotalCharges': 'Total facturado',
    'PaperlessBilling': 'Factura electrónica', 'OnlineSecurity': 'Seguridad online', 'TechSupport': 'Soporte técnico',
    'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming películas', 'Dependents': 'Dependientes',
    'Partner': 'Pareja', 'OnlineBackup': 'Respaldo online', 'SeniorCitizen': 'Adulto mayor',
    'MultipleLines': 'Múltiples líneas', 'PhoneService': 'Servicio telefónico',
    'DeviceProtection': 'Protección de equipo', 'gender': 'Género',
}


def prepare(df):
    """Misma limpieza que el notebook (sección 1): TotalCharges numérico y categorías redundantes a 'No'."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df.replace({'No internet service': 'No', 'No phone service': 'No'})


def valor_cliente(row, var):
    v = row[var]
    if var == 'tenure':
        return f'{int(v)} meses'
    if var in ('MonthlyCharges', 'TotalCharges'):
        return f'{v:,.2f} USD'
    if var == 'SeniorCitizen':
        return 'Sí' if v == 1 else 'No'
    return str(v)


# ----------------------------------------------------------------------------- carga
@st.cache_resource
def load_model():
    model = joblib.load('models/rf_final.joblib')
    pre, rf = model.named_steps['preprocess'], model.named_steps['model_instance']
    num_cols, cat_cols = list(pre.transformers_[0][2]), list(pre.transformers_[1][2])
    feat_names = [n.split('__', 1)[1] for n in pre.get_feature_names_out()]
    var_of = {n: next(c for c in num_cols + cat_cols if n == c or n.startswith(c + '_')) for n in feat_names}
    return model, pre, num_cols, cat_cols, feat_names, var_of, shap.TreeExplainer(rf)


@st.cache_data
def load_data():
    df = prepare(pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    test = pd.read_csv(ART + 'test_clients.csv')
    arts = {
        'cv': pd.read_csv(ART + 'model_comparison_cv.csv', index_col=0),
        'opt': pd.read_csv(ART + 'optimization_results.csv', index_col=0),
        'hist': pd.read_csv(ART + 'optimization_history.csv', index_col=0),
        'test': pd.read_csv(ART + 'test_metrics.csv', index_col=0),
        'cm': pd.read_csv(ART + 'confusion_matrix_test.csv', index_col=0),
        'trials': pd.read_csv(ART + 'pareto_trials.csv'),
        'lift': pd.read_csv(ART + 'lift_test.csv'),
        'imp': pd.read_csv(ART + 'feature_importance_comparison.csv', index_col=0),
    }
    selected = json.load(open(ART + 'selected_config.json'))
    return df, test, arts, selected


model, pre, NUM, CAT, FEAT, VAR_OF, explainer = load_model()
df, test, A, SEL = load_data()
FEATURES = NUM + CAT
THRESHOLD = SEL['threshold']


def shap_explain(X):
    """Valores SHAP (clase churn) por columna transformada y agrupados por variable original."""
    Xt = pd.DataFrame(pre.transform(X[FEATURES]), columns=FEAT)
    sv = explainer(Xt)[:, :, 1]
    disp = Xt.copy()
    disp[NUM] = X[NUM].to_numpy()
    sv.display_data = disp.to_numpy()
    grouped = pd.DataFrame(sv.values, columns=FEAT).T.groupby(VAR_OF).sum().T
    return sv, grouped


def acciones(row, g):
    """Acción de retención sugerida para cada factor de riesgo que SHAP marca como positivo."""
    reglas = [
        ('Contract', row['Contract'] == 'Month-to-month', 'Descuento por migrar a contrato anual'),
        ('TechSupport', row['InternetService'] != 'No' and row['TechSupport'] == 'No', '3 meses de soporte técnico sin costo'),
        ('OnlineSecurity', row['InternetService'] != 'No' and row['OnlineSecurity'] == 'No', '3 meses de seguridad online sin costo'),
        ('tenure', row['tenure'] <= 12, 'Programa de bienvenida: contacto a los 30, 60 y 90 días'),
        ('MonthlyCharges', row['MonthlyCharges'] > 70, 'Revisión de plan: paquete con mejor relación precio/servicios'),
        ('PaymentMethod', row['PaymentMethod'] == 'Electronic check', 'Incentivo por domiciliar el pago'),
    ]
    activas = sorted([(g[v], a) for v, cond, a in reglas if cond and g[v] > 0], reverse=True)
    return [a for _, a in activas] or ['Seguimiento estándar (sin factor de riesgo accionable)']


def score_clientes(X):
    """Ranking con score, razón principal y acción sugerida para un conjunto de clientes."""
    X = X.reset_index(drop=True)
    scores = model.predict_proba(X[FEATURES])[:, 1]
    _, g = shap_explain(X)
    out = X[['customerID']].copy() if 'customerID' in X else pd.DataFrame(index=X.index)
    out['score_riesgo'] = scores.round(3)
    out['contactar'] = np.where(scores >= THRESHOLD, 'Sí', 'No')
    out['razón principal'] = [f'{VAR_ES[g.loc[i].idxmax()]}: {valor_cliente(X.loc[i], g.loc[i].idxmax())}' for i in X.index]
    out['acción sugerida'] = [acciones(X.loc[i], g.loc[i])[0] for i in X.index]
    for c in ['Contract', 'tenure', 'MonthlyCharges']:
        out[c] = X[c]
    if 'Churn' in X:
        out['churn real (demo)'] = X['Churn'].map(ETIQUETA)
    return out.sort_values('score_riesgo', ascending=False).reset_index(drop=True)


@st.cache_data
def ranking_test():
    return score_clientes(test)


@st.cache_resource
def lime_explainer():
    sample = pd.read_csv(ART + 'X_train_sample.csv')[FEATURES]
    cat_vals = {c: sorted(sample[c].astype(str).unique()) for c in CAT}
    dtypes = sample.dtypes

    def encode(d):
        d = d[FEATURES].copy()
        for c in CAT:
            d[c] = d[c].astype(str).map({v: i for i, v in enumerate(cat_vals[c])})
        return d.to_numpy(dtype=float)

    def predict(arr):
        d = pd.DataFrame(arr, columns=FEATURES)
        for c in CAT:
            d[c] = d[c].round().astype(int).map(dict(enumerate(cat_vals[c]))).astype(dtypes[c])
        return model.predict_proba(d)

    exp = LimeTabularExplainer(
        encode(sample), feature_names=FEATURES, class_names=['No churn', 'Churn'],
        categorical_features=[FEATURES.index(c) for c in CAT],
        categorical_names={FEATURES.index(c): cat_vals[c] for c in CAT},
        mode='classification', random_state=42)
    return exp, encode, predict


# ----------------------------------------------------------------------------- encabezado
st.title('Clien↩urn')
st.markdown('**Retención predictiva para operadores regionales** — *antes de que crucen la puerta.*  \n'
            'Datos: Telco Customer Churn (IBM). Modelo: Random Forest optimizado (Optuna + NSGA-II). '
            'Los valores monetarios son **hipotéticos** y se declaran como supuestos.')

tab_eda, tab_perf, tab_pred, tab_biz = st.tabs(
    ['📊 Exploración de datos', '🎯 Desempeño del modelo', '🔍 Predicción y explicación', '💼 Aplicación de negocio'])

# ----------------------------------------------------------------------------- A. EDA
with tab_eda:
    st.subheader('¿Quiénes abandonan?')
    f1, f2, f3, f4 = st.columns(4)
    contratos = f1.multiselect('Contrato', sorted(df['Contract'].unique()), default=sorted(df['Contract'].unique()))
    internet = f2.multiselect('Internet', sorted(df['InternetService'].unique()), default=sorted(df['InternetService'].unique()))
    pagos = f3.multiselect('Método de pago', sorted(df['PaymentMethod'].unique()), default=sorted(df['PaymentMethod'].unique()))
    ten = f4.slider('Antigüedad (meses)', 0, 72, (0, 72))
    d = df[df['Contract'].isin(contratos) & df['InternetService'].isin(internet)
           & df['PaymentMethod'].isin(pagos) & df['tenure'].between(*ten)]

    if d.empty:
        st.warning('No hay clientes con esos filtros.')
    else:
        global_rate = df['Churn'].mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Clientes en el segmento', f'{len(d):,}')
        k2.metric('Tasa de abandono', f"{d['Churn'].mean():.1%}",
                  delta=f"{(d['Churn'].mean() - global_rate) * 100:+.1f} pp vs total", delta_color='inverse')
        k3.metric('Ingresos mensuales perdidos', f"{d.loc[d['Churn'] == 1, 'MonthlyCharges'].sum():,.0f} USD")
        k4.metric('Antigüedad mediana', f"{d['tenure'].median():.0f} meses")

        d = d.assign(Estado=d['Churn'].map(ETIQUETA))
        c1, c2 = st.columns(2)
        var = c1.selectbox('Tasa de abandono por…', ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport',
                                                      'OnlineSecurity', 'PaperlessBilling', 'SeniorCitizen', 'Partner'],
                           format_func=VAR_ES.get)
        tasa = d.groupby(var)['Churn'].agg(['mean', 'size']).reset_index().sort_values('mean', ascending=False)
        fig = px.bar(tasa, x=var, y='mean', text=tasa['mean'].map('{:.1%}'.format), hover_data={'size': True},
                     labels={'mean': 'Tasa de abandono', var: VAR_ES[var], 'size': 'Clientes'})
        fig.add_hline(y=global_rate, line_dash='dash', line_color='red', annotation_text=f'Global {global_rate:.1%}')
        fig.update_yaxes(tickformat='.0%')
        c1.plotly_chart(fig)

        fig = px.histogram(d, x='tenure', color='Estado', barmode='overlay', nbins=36, color_discrete_map=COLOR,
                           labels={'tenure': 'Antigüedad (meses)'}, title='Antigüedad: los clientes nuevos se van más')
        c2.plotly_chart(fig)

        c3, c4 = st.columns(2)
        fig = px.box(d, x='Estado', y='MonthlyCharges', color='Estado', color_discrete_map=COLOR, points=False,
                     labels={'MonthlyCharges': 'Cargo mensual (USD)'}, title='Cargo mensual por estado')
        c3.plotly_chart(fig)
        cruce = d.pivot_table(index='Contract', columns='InternetService', values='Churn', aggfunc='mean')
        fig = px.imshow(cruce, text_auto='.0%', color_continuous_scale='Reds', aspect='auto',
                        labels={'color': 'Abandono'}, title='Abandono: contrato × internet')
        c4.plotly_chart(fig)

# ----------------------------------------------------------------------------- B. desempeño
with tab_perf:
    st.subheader('1. Comparación de modelos (validación cruzada 5-fold, umbral 0.5)')
    st.dataframe(A['cv'])
    st.caption('Los tres modelos se evaluaron con los mismos folds. La Regresión Logística fue el baseline a superar.')

    st.subheader('2. Optimización de hiperparámetros del Random Forest (48 configuraciones cada una)')
    opt = A['opt'].drop(columns='Parámetros').round(3)
    st.dataframe(opt)
    estr = [e for e in A['opt'].index if 'Baseline' not in e]
    c1, c2, c3 = st.columns(3)
    fig = go.Figure(go.Bar(x=estr, y=A['opt'].loc[estr, 'F1'], error_y=dict(array=A['opt'].loc[estr, 'F1 std'])))
    fig.add_hline(y=A['opt'].loc['Baseline: Logistic Regression', 'F1'], line_dash='dash', line_color='red',
                  annotation_text='Baseline LR')
    fig.update_layout(title='Mejor F1 (± std)', yaxis_range=[0.55, 0.68])
    c1.plotly_chart(fig)
    c2.plotly_chart(px.bar(x=estr, y=A['opt'].loc[estr, 'Tiempo (s)'], title='Tiempo de optimización (s)',
                           labels={'x': '', 'y': 'segundos'}))
    conv = A['hist'].cummax().reset_index().melt(id_vars='config', var_name='Estrategia', value_name='Mejor F1')
    conv['config'] += 1
    c3.plotly_chart(px.line(conv, x='config', y='Mejor F1', color='Estrategia', title='Convergencia',
                            labels={'config': 'Configuraciones evaluadas'}))
    st.caption('Las tres estrategias empatan dentro de la desviación entre folds; se eligió TPE por escalabilidad.')

    st.subheader('3. Frente de Pareto: Recall vs Precision (NSGA-II)')
    tr = A['trials'].assign(Tipo=np.where(A['trials']['pareto'], 'No dominada', 'Dominada'))
    fig = px.scatter(tr, x='recall', y='precision', color='Tipo', hover_data=['threshold', 'f1'],
                     color_discrete_map={'No dominada': '#E45756', 'Dominada': '#BBBBBB'})
    fig.add_trace(go.Scatter(x=[SEL['cv_recall']], y=[SEL['cv_precision']], mode='markers', name='Solución elegida',
                             marker=dict(symbol='star', size=18, color='gold', line=dict(width=1, color='black'))))
    fig.update_layout(xaxis_title='Recall (churners detectados)', yaxis_title='Precision (contactados que se iban)')
    st.plotly_chart(fig)
    st.caption(f"Solución elegida: umbral {THRESHOLD:.2f}, recall {SEL['cv_recall']:.2f}, precision {SEL['cv_precision']:.2f}. "
               'Es la que contacta a menos clientes conservando el 95 % del valor neto máximo.')

    st.subheader('4. Evaluación final en test (1.409 clientes nunca vistos)')
    c1, c2 = st.columns([3, 2])
    c1.dataframe(A['test'].round(3))
    c1.dataframe(A['lift'], hide_index=True)
    cm = A['cm'].copy()
    cm.index, cm.columns = ['Real: se queda', 'Real: se va'], ['Pred: se queda', 'Pred: se va']
    c2.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                              title=f'Matriz de confusión (umbral {THRESHOLD:.2f})'))

# ----------------------------------------------------------------------------- C. predicción
with tab_pred:
    st.subheader('¿Qué mira el modelo en general?')
    imp = A['imp'].sort_values('SHAP mean|valor|').tail(10)
    fig = px.bar(imp, x='SHAP mean|valor|', y=imp.index.map(VAR_ES.get), orientation='h',
                 labels={'y': '', 'SHAP mean|valor|': 'Impacto promedio en el riesgo (SHAP)'})
    st.plotly_chart(fig)
    with st.expander('Comparación SHAP vs Feature Importance (RF) vs Permutation'):
        st.dataframe(A['imp'].rename(index=VAR_ES).round(4))
        st.caption('Importancia predictiva no implica causalidad.')

    st.subheader('Evaluar un cliente')
    churners = test[test['Churn'] == 1]
    scores_test = model.predict_proba(test[FEATURES])[:, 1]
    s_churn = model.predict_proba(churners[FEATURES])[:, 1]
    presets = {'Cliente de alto riesgo (ejemplo)': churners.iloc[int(np.argmax(s_churn))],
               'Cliente leal que igual se fue (ejemplo)': churners.iloc[int(np.argmin(s_churn))],
               'Personalizado': test.iloc[0]}
    preset = st.selectbox('Cargar datos', list(presets))
    base = presets[preset]

    def opt_idx(col, val):
        opciones = sorted(df[col].unique())
        return opciones, opciones.index(val)

    with st.form('cliente'):
        c1, c2, c3 = st.columns(3)
        entrada = {}
        c1.markdown('**Perfil**')
        for col in ['gender', 'Partner', 'Dependents']:
            o, i = opt_idx(col, base[col])
            entrada[col] = c1.selectbox(VAR_ES[col], o, index=i, key=f'{col}_{preset}')
        entrada['SeniorCitizen'] = c1.selectbox('Adulto mayor', [0, 1], index=int(base['SeniorCitizen']),
                                                format_func=lambda x: 'Sí' if x else 'No', key=f'sen_{preset}')
        c2.markdown('**Cuenta**')
        entrada['tenure'] = c2.slider('Antigüedad (meses)', 0, 72, int(base['tenure']), key=f'ten_{preset}')
        for col in ['Contract', 'PaymentMethod', 'PaperlessBilling']:
            o, i = opt_idx(col, base[col])
            entrada[col] = c2.selectbox(VAR_ES[col], o, index=i, key=f'{col}_{preset}')
        entrada['MonthlyCharges'] = c2.number_input('Cargo mensual (USD)', 18.0, 120.0, float(base['MonthlyCharges']),
                                                    key=f'mc_{preset}')
        entrada['TotalCharges'] = c2.number_input('Total facturado (USD)', 0.0, 9000.0, float(base['TotalCharges']),
                                                  key=f'tc_{preset}')
        c3.markdown('**Servicios**')
        for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            o, i = opt_idx(col, base[col])
            entrada[col] = c3.selectbox(VAR_ES[col], o, index=i, key=f'{col}_{preset}')
        enviado = st.form_submit_button('Predecir riesgo', type='primary')

    if enviado:
        # Coherencia: sin internet no hay servicios de internet; sin teléfono no hay múltiples líneas
        if entrada['InternetService'] == 'No':
            for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                entrada[col] = 'No'
        if entrada['PhoneService'] == 'No':
            entrada['MultipleLines'] = 'No'
        X = pd.DataFrame([entrada])[FEATURES]
        for c in FEATURES:
            X[c] = X[c].astype(test[c].dtype)
        p = model.predict_proba(X)[0, 1]
        sv, g = shap_explain(X)
        g = g.iloc[0]
        row = X.iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric('Score de riesgo', f'{p:.2f}')
        m2.metric('Decisión', '🔴 Contactar' if p >= THRESHOLD else '🟢 No contactar',
                  help=f'Se contacta si el score ≥ {THRESHOLD:.2f} (punto de operación elegido).')
        m3.metric('Más riesgoso que', f'{(scores_test < p).mean():.0%} de los clientes')
        st.caption('El score ordena a los clientes por riesgo; no es una probabilidad calibrada '
                   '(el modelo se entrenó con clases balanceadas).')

        riesgo = g[g > 0].sort_values(ascending=False).head(3)
        protege = g[g < 0].sort_values().head(2)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('#### ¿Por qué? (en simple)')
            for v, val in riesgo.items():
                st.markdown(f'🔺 **{VAR_ES[v]}: {valor_cliente(row, v)}** — sube el riesgo ({val:+.2f})')
            for v, val in protege.items():
                st.markdown(f'🔻 **{VAR_ES[v]}: {valor_cliente(row, v)}** — lo protege ({val:+.2f})')
            st.markdown('#### Acción sugerida')
            for a in acciones(row, g)[:3]:
                st.markdown(f'✅ {a}')
            st.caption('Las acciones son hipótesis basadas en asociaciones del modelo (no causales): validar con grupo de control.')
        with c2:
            st.markdown('#### SHAP: aporte de cada variable')
            fig = plt.figure()
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close(fig)

        st.markdown('#### LIME: segunda opinión local')
        exp, encode, predict = lime_explainer()
        with st.spinner('Calculando LIME…'):
            e = exp.explain_instance(encode(X)[0], predict, num_features=8, labels=(1,), num_samples=2000)
        lime_df = pd.DataFrame(e.as_list(label=1), columns=['Regla', 'Peso']).iloc[::-1]
        lime_df['Efecto'] = np.where(lime_df['Peso'] > 0, 'Hacia abandono', 'Hacia quedarse')
        st.plotly_chart(px.bar(lime_df, x='Peso', y='Regla', color='Efecto', orientation='h',
                               color_discrete_map={'Hacia abandono': '#E45756', 'Hacia quedarse': '#4C78A8'}))
        st.caption('LIME ajusta un modelo lineal alrededor del cliente con vecinos aleatorios: sus pesos son relativos '
                   'y pueden variar levemente entre ejecuciones. SHAP es exacto y suma la predicción.')

# ----------------------------------------------------------------------------- D. negocio
with tab_biz:
    st.subheader('Planificador de campaña de retención')
    st.markdown('Ajusta los supuestos de **tu** campaña: Clien↩urn recomienda a cuántos clientes contactar y a quiénes. '
                'Demostración sobre los 1.409 clientes de test (con su resultado real conocido).')
    c1, c2, c3, c4 = st.columns(4)
    costo = c1.number_input('Costo de la promoción por cliente (USD)', 0, 500, 50, 10)
    exito = c2.slider('Tasa de éxito de la promoción', 0.0, 1.0, 0.30, 0.05,
                      help='Proporción de churners contactados que aceptan y se quedan.')
    meses = c3.slider('Meses de facturación retenidos', 1, 36, 12)
    capacidad = c4.slider('Capacidad máxima de contacto (% de clientes)', 5, 100, 100, 5)

    y = test['Churn'].to_numpy()
    mc = test['MonthlyCharges'].to_numpy()
    umbrales = np.round(np.arange(0.05, 0.96, 0.01), 2)
    filas = []
    for t in umbrales:
        c = scores_test >= t
        tp = c & (y == 1)
        filas.append({'umbral': t, '% contactado': c.mean(), 'contactados': int(c.sum()),
                      'recall': tp.sum() / y.sum(), 'precision': tp.sum() / max(c.sum(), 1),
                      'valor': exito * meses * mc[tp].sum() - costo * c.sum()})
    curva = pd.DataFrame(filas)
    factible = curva[curva['% contactado'] <= capacidad / 100]
    if factible.empty:
        factible = curva.iloc[[-1]]
    # Misma regla que el notebook (5.5): el menor nº de contactos que conserva el 95 % del valor máximo
    vmax = factible['valor'].max()
    candidatos = factible[factible['valor'] >= 0.95 * vmax] if vmax > 0 else factible
    rec = candidatos.loc[candidatos['contactados'].idxmin()] if vmax > 0 else factible.loc[factible['valor'].idxmax()]

    manual = st.checkbox('Elegir el umbral manualmente')
    t_op = st.slider('Umbral de operación', 0.05, 0.95, float(rec['umbral']), 0.01) if manual else float(rec['umbral'])
    op = curva.loc[(curva['umbral'] - t_op).abs().idxmin()]
    st.caption('Recomendación: contactar al menor número de clientes que conserva el 95 % del valor neto máximo '
               'dentro de tu capacidad.')

    if vmax <= 0 or rec['% contactado'] < 0.02:
        st.error('Con estos supuestos **la campaña no es rentable** (o solo lo es para un puñado de clientes): '
                 'el costo de la promoción supera lo que se recupera. Recomendación: no lanzar la campaña o rediseñar la promoción.')

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Clientes a contactar', f"{int(op['contactados'])} ({op['% contactado']:.0%})")
    k2.metric('Churners detectados', f"{op['recall']:.0%}")
    k3.metric('Aciertos por llamada', f"{op['precision']:.0%}", help='Precision: de los contactados, cuántos se iban.')
    k4.metric('Valor neto estimado', f"{op['valor']:,.0f} USD")

    rng = np.random.default_rng(42)
    azar = np.zeros(len(y), dtype=bool)
    azar[rng.choice(len(y), int(op['contactados']), replace=False)] = True
    v_azar = exito * meses * mc[azar & (y == 1)].sum() - costo * azar.sum()
    v_todos = exito * meses * mc[y == 1].sum() - costo * len(y)

    c1, c2 = st.columns([3, 2])
    fig = px.line(curva, x='% contactado', y='valor', hover_data=['umbral', 'recall', 'precision'],
                  labels={'valor': 'Valor neto (USD)', '% contactado': '% de clientes contactados'},
                  title='Valor neto según cuántos clientes contactas')
    fig.add_trace(go.Scatter(x=[op['% contactado']], y=[op['valor']], mode='markers', name='Operación',
                             marker=dict(symbol='star', size=16, color='gold', line=dict(width=1, color='black'))))
    if capacidad < 100:
        fig.add_vline(x=capacidad / 100, line_dash='dot', annotation_text='Capacidad')
    fig.update_xaxes(tickformat='.0%')
    c1.plotly_chart(fig)
    comp = pd.DataFrame({'Estrategia': ['Clien↩urn', 'Al azar (mismo nº)', 'Contactar a todos'],
                         'Valor neto (USD)': [op['valor'], v_azar, v_todos]})
    c2.plotly_chart(px.bar(comp, x='Estrategia', y='Valor neto (USD)', color='Estrategia', title='Comparación',
                           color_discrete_sequence=['#F2B701', '#BBBBBB', '#4C78A8']))

    st.markdown('#### ¿Cuánto vale para tu operador?')
    c1, c2 = st.columns([1, 3])
    cartera = c1.number_input('Clientes de tu operador', 1000, 200000, 10000, 1000)
    valor_anual = op['valor'] / len(test) * cartera
    with c2:
        k1, k2, k3 = st.columns(3)
        k1.metric('Valor neto estimado / año', f'{valor_anual:,.0f} USD')
        k2.metric('Costo Clien↩urn año 1', f'{COSTO_CLIENTURN_ANIO1:,.0f} USD',
                  help='Piloto 1.500 + implementación 3.000 + 9 meses × 800 (precios hipotéticos).')
        k3.metric('Retorno', f'{valor_anual / COSTO_CLIENTURN_ANIO1:.1f}×' if valor_anual > 0 else '—')
    st.caption('⚠️ Estimación con supuestos hipotéticos escalada desde la muestra de test. '
               'El piloto con grupo de control reemplaza estos supuestos por datos reales del operador.')

    st.markdown('#### Lista de clientes a contactar')
    rk = ranking_test()
    lista = rk[rk['score_riesgo'] >= op['umbral']]
    st.dataframe(lista, hide_index=True)
    st.download_button('Descargar lista (CSV)', lista.to_csv(index=False).encode('utf-8'),
                       'clienturn_contactar.csv', 'text/csv')

    st.markdown('#### Analiza tu propia cartera')
    archivo = st.file_uploader('CSV con las mismas columnas del dataset Telco (customerID opcional)', type='csv')
    if archivo is not None:
        nuevos = prepare(pd.read_csv(archivo))
        faltan = [c for c in FEATURES if c not in nuevos]
        if faltan:
            st.error(f'Faltan columnas: {faltan}')
        else:
            for c in FEATURES:
                nuevos[c] = nuevos[c].astype(test[c].dtype)
            res = score_clientes(nuevos.drop(columns='Churn', errors='ignore'))
            res = res[res['score_riesgo'] >= op['umbral']]
            st.success(f'{len(res)} de {len(nuevos)} clientes superan el umbral {op["umbral"]:.2f}.')
            st.dataframe(res, hide_index=True)
            st.download_button('Descargar resultado (CSV)', res.to_csv(index=False).encode('utf-8'),
                               'clienturn_cartera.csv', 'text/csv')
