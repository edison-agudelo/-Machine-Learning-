from flask import Flask, render_template
from data.busqueda import research_info, ml_use_cases, team_info, study_metrics

app = Flask(__name__)

@app.route('/')
def index():
    """Página principal con resumen de la investigación"""
    return render_template('index.html', 
                         title='Inicio',
                         research=research_info,
                         use_cases=ml_use_cases[:2],  # Solo mostramos 2 casos en el inicio
                         metrics=study_metrics)

@app.route('/casos')
def casos():
    """Página con todos los casos de uso detallados"""
    return render_template('casos.html',
                         title='Casos de Uso',
                         research=research_info,
                         use_cases=ml_use_cases,
                         metrics=study_metrics)

@app.route('/caso/<int:caso_id>')
def caso_detalle(caso_id):
    """Página de detalle de un caso específico"""
    caso = None
    for uc in ml_use_cases:
        if uc['id'] == caso_id:
            caso = uc
            break
    
    if caso is None:
        return "Caso no encontrado", 404
    
    return render_template('caso_detalle.html',
                         title=f'Caso {caso_id}',
                         caso=caso,
                         research=research_info)

@app.route('/metodologia')
def metodologia():
    """Página sobre la metodología de investigación"""
    return render_template('metodologia.html',
                         title='Metodología',
                         research=research_info,
                         team=team_info,
                         metrics=study_metrics)

@app.errorhandler(404)
def not_found(error):
    """Manejo de error 404"""
    return render_template('404.html', title='Página no encontrada'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)