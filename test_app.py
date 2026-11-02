"""Prueba de humo de la app: arranca sin errores y predice un cliente.

Ejecutar: python test_app.py
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file('app.py', default_timeout=180).run()
assert not at.exception, at.exception

next(b for b in at.button if b.label == 'Predecir riesgo').click()
at.run()
assert not at.exception, at.exception

labels = [m.label for m in at.metric]
assert 'Score de riesgo' in labels, labels
print('OK -', len(at.metric), 'métricas,', len(at.get('plotly_chart')), 'gráficos plotly')
