from flask import Flask, render_template, request, jsonify
import modelo
from data.busqueda import research_info, ml_use_cases, team_info, study_metrics
import regresion_logistica  
import numpy as np
import traceback
import subprocess
import os
import sys

# Agregar el directorio scripts al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Importar el módulo de Random Forest
try:
    from scripts import random_forest_rotacion
    RF_AVAILABLE = True
except ImportError:
    print("Advertencia: No se pudo importar random_forest_rotacion")
    RF_AVAILABLE = False

app = Flask(__name__)

# ===============================
# VARIABLES GLOBALES PARA LOS MODELOS
# ===============================

# Regresión Logística
accuracy_global = 0.0
report_global = ""
cm_img_global = ""

# Random Forest
rf_metrics_global = None
rf_report_global = None
rf_cm_img_global = None

# ===============================
# INICIALIZACIÓN DE MODELOS
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

def initialize_rf_model():
    """Entrena el modelo de Random Forest al iniciar la app."""
    global rf_metrics_global, rf_report_global, rf_cm_img_global
    
    if not RF_AVAILABLE:
        print("Random Forest no está disponible")
        return
    
    print("Entrenando modelo Random Forest...")
    try:
        # Verificar que existe el dataset
        csv_path = "data/rotacion_empleados.csv"
        
        if not os.path.exists(csv_path):
            print(f"Dataset no encontrado en {csv_path}. Generando...")
            
            # Método 1: Intentar importar desde data.generate_dataset
            try:
                import importlib.util
                generate_dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'generate_dataset.py')
                spec = importlib.util.spec_from_file_location("generate_dataset", generate_dataset_path)
                generate_dataset_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(generate_dataset_module)
                generate_dataset_module.generar_dataset()
                print("Dataset generado exitosamente (método 1)")
            except Exception as e1:
                print(f"Método 1 falló: {e1}")
                
                # Método 2: Generar directamente aquí
                try:
                    print("Intentando generar dataset directamente...")
                    import pandas as pd
                    
                    RND = 42
                    np.random.seed(RND)
                    
                    os.makedirs("data", exist_ok=True)
                    n = 1000
                    
                    anios_empresa = np.clip(np.round(np.random.exponential(scale=3.0, size=n), 1), 0, 40)
                    nivel_satisfaccion = np.clip(np.round(np.random.normal(loc=0.6, scale=0.2, size=n), 3), 0, 1)
                    salario = np.round(np.random.normal(loc=3000, scale=800, size=n), 2)
                    n_capacitaciones = np.random.poisson(lam=2, size=n)
                    eval_desempeno = np.clip(np.round(np.random.normal(loc=3.5, scale=0.8, size=n), 2), 1, 5)
                    
                    prob_base = (
                        0.4 * (1 - nivel_satisfaccion) +
                        0.2 * (1 / (1 + anios_empresa)) +
                        0.15 * (1 - (eval_desempeno - 1) / 4) +
                        0.1 * (1 - np.clip((salario - salario.min()) / (salario.max() - salario.min()), 0, 1)) +
                        0.15 * (1 - (n_capacitaciones / (n_capacitaciones.max() + 1)))
                    )
                    prob_base = (prob_base - prob_base.min()) / (prob_base.max() - prob_base.min())
                    prob_final = 0.85 * prob_base + 0.15 * np.random.rand(n)
                    label = np.where(prob_final > np.quantile(prob_final, 0.6), "Alta", "Baja")
                    
                    df = pd.DataFrame({
                        "empleado_id": np.arange(1, n + 1),
                        "anios_empresa": anios_empresa,
                        "nivel_satisfaccion": nivel_satisfaccion,
                        "salario": salario,
                        "n_capacitaciones": n_capacitaciones,
                        "eval_desempeno": eval_desempeno,
                        "rotacion": label
                    })
                    
                    for col in ["nivel_satisfaccion", "salario", "eval_desempeno"]:
                        mask = np.random.rand(n) < 0.05
                        df.loc[mask, col] = np.nan
                    
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"Dataset generado exitosamente en {csv_path} (método 2)")
                    
                except Exception as e2:
                    print(f"Error al generar dataset (método 2): {e2}")
                    traceback.print_exc()
                    return
        
        # Entrenar y evaluar
        metrics, report, cm_path = random_forest_rotacion.train_and_evaluate_model(
            csv_path=csv_path, 
            save=True
        )
        
        rf_metrics_global = metrics
        rf_report_global = report
        
        # Convertir path absoluto a relativo para templates
        rf_cm_img_global = cm_path.replace('static/', '').replace('static\\', '').replace('\\', '/')
        if rf_cm_img_global.startswith('/'):
            rf_cm_img_global = rf_cm_img_global[1:]
        
        print(f"Random Forest entrenado exitosamente")
        print(f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
    except Exception as e:
        print(f"Error al entrenar Random Forest: {e}")
        traceback.print_exc()
        rf_metrics_global = {'accuracy': 0.0}
        rf_report_global = {}
        rf_cm_img_global = None

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
    return render_template('conceptos.html', title="Conceptos de Regresión Logística")

@app.route('/regresion-logistica/practico')
def regresion_practico():
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
    try:
        print("=== INICIO PREDICCIÓN REGRESIÓN LOGÍSTICA ===")
        print(f"Datos recibidos del formulario: {dict(request.form)}")
        
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
# RUTAS DE CLASIFICACIÓN (RANDOM FOREST)
# ===============================

@app.route('/clasificacion/conceptos')
def clasificacion_conceptos():
    """Página de conceptos básicos con mapa conceptual"""
    return render_template(
        'clasificacion_mapaconceptual.html',
        title='Conceptos Básicos de Clasificación'
    )

@app.route('/clasificacion/practico')
def clasificacion_practico():
    """Página del caso práctico de Random Forest"""
    global rf_metrics_global, rf_report_global, rf_cm_img_global
    
    # Si el modelo no está inicializado, intentar inicializarlo
    if rf_metrics_global is None:
        initialize_rf_model()
    
    return render_template(
        'clasificacion_caso_practico.html',
        title='Caso Práctico - Random Forest',
        metrics=rf_metrics_global or {},
        report=rf_report_global or {},
        cm_img=rf_cm_img_global
    )

@app.route('/clasificacion/predict', methods=['POST'])
def clasificacion_predict():
    """Endpoint para predicción con Random Forest"""
    try:
        print("=== INICIO PREDICCIÓN RANDOM FOREST ===")
        data = request.get_json()
        print(f"Datos recibidos: {data}")
        
        # Validar datos requeridos
        required = ['anios_empresa', 'nivel_satisfaccion', 'salario', 
                   'n_capacitaciones', 'eval_desempeno']
        
        for field in required:
            if field not in data:
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        # Preparar input
        input_dict = {
            'anios_empresa': float(data['anios_empresa']),
            'nivel_satisfaccion': float(data['nivel_satisfaccion']),
            'salario': float(data['salario']),
            'n_capacitaciones': int(data['n_capacitaciones']),
            'eval_desempeno': float(data['eval_desempeno'])
        }
        
        threshold = float(data.get('threshold', 0.5))
        
        # Validaciones
        if not (0 <= input_dict['nivel_satisfaccion'] <= 1):
            return jsonify({'error': 'nivel_satisfaccion debe estar entre 0 y 1'}), 400
        
        if not (1 <= input_dict['eval_desempeno'] <= 5):
            return jsonify({'error': 'eval_desempeno debe estar entre 1 y 5'}), 400
        
        if not (0 <= threshold <= 1):
            return jsonify({'error': 'threshold debe estar entre 0 y 1'}), 400
        
        # Realizar predicción
        result = random_forest_rotacion.predict_label(input_dict, threshold)
        
        response = {
            'label': result['label'],
            'probability': result['probability']
        }
        
        print(f"Predicción exitosa: {response}")
        return jsonify(response)
        
    except FileNotFoundError as e:
        error_msg = "Modelo no encontrado. Por favor, entrena el modelo primero."
        print(f"ERROR: {error_msg}")
        return jsonify({'error': error_msg}), 500
        
    except Exception as e:
        error_msg = f"Error en la predicción: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

# ===============================
# GENERAR DATASET AUTOMÁTICAMENTE
# ===============================

@app.route('/generar-dataset', methods=['GET'])
def generar_dataset():
    try:
        result = subprocess.run(['python', 'data/generate_dataset.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({"status": "success", "message": "Dataset generado correctamente."})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===============================
# RUTAS ADICIONALES Y UTILIDADES
# ===============================

@app.route('/health')
def health_check():
    global accuracy_global, rf_metrics_global
    return jsonify({
        'status': 'OK',
        'logistic_model_loaded': accuracy_global > 0,
        'logistic_accuracy': accuracy_global,
        'rf_model_loaded': rf_metrics_global is not None and rf_metrics_global.get('accuracy', 0) > 0,
        'rf_accuracy': rf_metrics_global.get('accuracy', 0) if rf_metrics_global else 0
    })

@app.route('/model-info')
def model_info():
    global accuracy_global, report_global, rf_metrics_global, rf_report_global
    return jsonify({
        'logistic': {
            'accuracy': accuracy_global,
            'report': report_global,
            'status': 'trained' if accuracy_global > 0 else 'not_trained'
        },
        'random_forest': {
            'metrics': rf_metrics_global,
            'status': 'trained' if (rf_metrics_global and rf_metrics_global.get('accuracy', 0) > 0) else 'not_trained'
        }
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
    print("="*60)
    print("Iniciando aplicación Flask...")
    print("="*60)
    
    # Inicializar modelo de Regresión Logística
    print("\n[1/2] Inicializando Regresión Logística...")
    initialize_model()
    if accuracy_global > 0:
        print("Regresión Logística inicializada correctamente")
    else:
        print("Advertencia: Regresión Logística no se inicializó correctamente")
    
    # Inicializar modelo de Random Forest
    print("\n[2/2] Inicializando Random Forest...")
    initialize_rf_model()
    if rf_metrics_global and rf_metrics_global.get('accuracy', 0) > 0:
        print("Random Forest inicializado correctamente")
    else:
        print("Advertencia: Random Forest no se inicializó correctamente")
    
    print("\n" + "="*60)
    print("Iniciando servidor Flask en http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)