from flask import Flask, render_template, request, jsonify
import modelo
from data.busqueda import research_info, ml_use_cases, team_info, study_metrics
import regresion_logistica  
import numpy as np
import traceback

app = Flask(__name__)

# ===============================
# VARIABLES GLOBALES PARA EL MODELO
# ===============================
accuracy_global = 0.0
report_global = ""
cm_img_global = ""

# ===============================
# LÓGICA DE INICIALIZACIÓN DEL MODELO DE REGRESIÓN LOGÍSTICA
# ===============================

def initialize_model():
    """Entrena el modelo de regresión logística solo una vez al iniciar la app."""
    print("Entrenando el modelo de Regresión Logística...")
    global accuracy_global, report_global, cm_img_global
    try:
       
        accuracy_global, report_global, cm_img_global = regresion_logistica.train_and_evaluate_model()
        print("Modelo entrenado y métricas generadas.")
        print(f"Accuracy obtenida: {accuracy_global}")
    except Exception as e:
        print(f"Error al entrenar el modelo de Regresión Logística: {e}")
        accuracy_global = 0.0
        report_global = "Error al generar el reporte."
        cm_img_global = ""

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
    try:
        datos = modelo.generar_conceptos()
        return render_template(
            'conceptos.html', 
            title="Conceptos básicos de Regresión Lineal", 
            plot_conceptos=datos["plot_conceptos"]
        )
    except Exception as e:
        print(f"Error en conceptos: {e}")
        return render_template(
            'conceptos.html', 
            title="Conceptos básicos de Regresión Lineal", 
            plot_conceptos=""
        )

@app.route('/RL', methods=['GET', 'POST'])
def RL():
    prediccion = None
    metrics = None
    plot_url = None
    data = None

    try:
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
    except Exception as e:
        print(f"Error en RL: {e}")
        # Valores por defecto en caso de error
        metrics = {"error": "Error al cargar el modelo"}
        plot_url = ""
        data = []

    return render_template(
        'rl.html',
        title="Regresión Lineal",
        prediccion=prediccion,
        metrics=metrics,
        plot_url=plot_url,
        data=data
    )

# ===============================
# RUTAS DE REGRESIÓN LOGÍSTICA
# ===============================

@app.route('/regresion-logistica/conceptos')
def regresion_conceptos():
    """Ruta para la página de conceptos de regresión logística."""
    return render_template('conceptos.html', title="Conceptos de Regresión Logística")

@app.route('/regresion-logistica/practico')
def regresion_practico():
    """Ruta para la página práctica con métricas y formulario."""
    global accuracy_global, report_global, cm_img_global
    return render_template(
        'R_LOG.html', 
        title="Regresión Logística - Práctico", 
        accuracy=accuracy_global, 
        report=report_global, 
        cm_img=cm_img_global
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para manejar las predicciones del formulario."""
    try:
        print("=== INICIO PREDICCIÓN ===")
        print(f"Datos recibidos del formulario: {dict(request.form)}")
        
        # Validar que todos los campos estén presentes
        required_fields = ['horas_trabajadas', 'nivel_seguridad', 'edad', 'tiempo_puesto']
        for field in required_fields:
            if field not in request.form:
                return jsonify(error=f"Campo requerido faltante: {field}"), 400

        nivel_seguridad_text = request.form['nivel_seguridad']
        valid_security_levels = ['Bajo', 'Medio', 'Alto']
        
        if nivel_seguridad_text not in valid_security_levels:
            return jsonify(error=f"Nivel de seguridad inválido: {nivel_seguridad_text}"), 400
        
        try:
            horas_trabajadas = float(request.form['horas_trabajadas'])
            edad = float(request.form['edad'])
            tiempo_puesto = float(request.form['tiempo_puesto'])
        except ValueError as ve:
            return jsonify(error=f"Error en conversión de datos: {ve}"), 400
        
        if horas_trabajadas < 0 or horas_trabajadas > 24:
            return jsonify(error="Las horas trabajadas deben estar entre 0 y 24"), 400
        if edad < 16 or edad > 100:
            return jsonify(error="La edad debe estar entre 16 y 100 años"), 400
        if tiempo_puesto < 0:
            return jsonify(error="El tiempo en el puesto no puede ser negativo"), 400
        
        input_data = [horas_trabajadas, edad, tiempo_puesto, nivel_seguridad_text]
        
        print(f"Datos preparados para enviar: {input_data}")
        print(f"Tipos de datos: {[type(x) for x in input_data]}")
        
        prediction, probability = regresion_logistica.predict_label(input_data)
        
        response = {
            'prediction': str(prediction),
            'probability': f"{float(probability):.2f}",
            'input_data': {
                'horas_trabajadas': horas_trabajadas,
                'nivel_seguridad': nivel_seguridad_text,
                'edad': edad,
                'tiempo_puesto': tiempo_puesto
            }
        }
        
        print(f"Predicción exitosa: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error en la predicción: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify(error=error_msg), 500

# ===============================
# RUTAS ADICIONALES Y UTILIDADES
# ===============================

@app.route('/health')
def health_check():
    global accuracy_global
    return jsonify({
        'status': 'OK',
        'model_loaded': accuracy_global > 0,
        'model_accuracy': accuracy_global
    })

@app.route('/model-info')
def model_info():
    global accuracy_global, report_global
    return jsonify({
        'accuracy': accuracy_global,
        'report': report_global,
        'model_status': 'trained' if accuracy_global > 0 else 'not_trained'
    })

# ===============================
# MANEJO DE ERRORES
# ===============================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', title='Página no encontrada'), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Error interno del servidor: {error}")
    return jsonify(error="Error interno del servidor"), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify(error="Solicitud incorrecta"), 400

# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    print("Iniciando aplicación Flask...")
    
    initialize_model()
    
    if accuracy_global > 0:
        print(" Modelo inicializado correctamente")
    else:
        print("  Advertencia: El modelo no se inicializó correctamente")
    
    print(" Iniciando servidor Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)
