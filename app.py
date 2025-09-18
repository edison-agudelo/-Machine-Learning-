from flask import Flask, render_template, request
import modelo
from data.busqueda import research_info, ml_use_cases, team_info, study_metrics

app = Flask(__name__)

# ===============================
# RUTAS PRINCIPALES DEL PROYECTO
# ===============================

@app.route('/')
def index():
    return render_template(
        'index.html',
        title='Inicio',
        research=research_info,
        use_cases=ml_use_cases[:2],
        metrics=study_metrics
    )

@app.route('/casos')
def casos():
    return render_template(
        'casos.html',
        title='Casos de Uso',
        research=research_info,
        use_cases=ml_use_cases,
        metrics=study_metrics
    )

@app.route('/caso/<int:caso_id>')
def caso_detalle(caso_id):
    caso = next((uc for uc in ml_use_cases if uc['id'] == caso_id), None)
    if caso is None:
        return "Caso no encontrado", 404
    return render_template(
        'caso_detalle.html',
        title=f'Caso {caso_id}',
        caso=caso,
        research=research_info
    )

@app.route('/metodologia')
def metodologia():
    return render_template(
        'metodologia.html',
        title='Metodología',
        research=research_info,
        team=team_info,
        metrics=study_metrics
    )

# ===============================
# RUTAS DE REGRESIÓN LINEAL
# ===============================

@app.route('/conceptos')
def conceptos():
    datos = modelo.generar_conceptos()
    return render_template(
        'conceptos.html', 
        title="Conceptos básicos de Regresión Lineal", 
        plot_conceptos=datos["plot_conceptos"]
    )

@app.route('/RL', methods=['GET', 'POST'])
def RL():
    prediccion = None
    metrics = None
    plot_url = None
    data = None

    if request.method == 'POST':
        tiempo = float(request.form['tiempo'])
        calidad = float(request.form['calidad'])
        resultado = modelo.ejecutar_modelo(tiempo, calidad)
        prediccion = resultado["prediccion"]
        metrics = resultado["metrics"]
        plot_url = resultado["plot_url"]
        data = resultado["data"]
    else:
        resultado = modelo.ejecutar_modelo()
        metrics = resultado["metrics"]
        plot_url = resultado["plot_url"]
        data = resultado["data"]

    return render_template(
        'rl.html',
        title="Regresión Lineal",
        prediccion=prediccion,
        metrics=metrics,
        plot_url=plot_url,
        data=data
    )

# ===============================
# MANEJO DE ERRORES
# ===============================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', title='Página no encontrada'), 404

# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
