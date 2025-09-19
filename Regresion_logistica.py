import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import traceback
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    """Clase para manejar el modelo de regresión logística de manera encapsulada."""
    
    def __init__(self):
        self.model = None
        self.column_transformer = None
        self.accuracy = 0.0
        self.report_html = ""
        self.cm_img_base64 = ""
        self.is_trained = False
        self.features_columns = ['Horas trabajadas', 'Edad', 'Tiempo en el puesto']
        self.categorical_features = ['Nivel de seguridad']
        self.model_path = 'models/logistic_regression_model.pkl'
        
    def _create_dataset(self):
        """Crea un dataset sintético para el entrenamiento."""
        logger.info("Creando dataset sintético...")
        
        np.random.seed(42)  # Para reproducibilidad
        n_samples = 1000
        
        # Generar datos más realistas
        horas = np.random.normal(8, 2, n_samples)  # Media 8 horas, desv 2
        horas = np.clip(horas, 1, 16)  # Entre 1 y 16 horas
        
        niveles = np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.3, 0.5, 0.2])
        
        edades = np.random.normal(35, 10, n_samples)  # Media 35 años, desv 10
        edades = np.clip(edades, 18, 65).astype(int)
        
        tiempos = np.random.exponential(3, n_samples)  # Distribución exponencial
        tiempos = np.clip(tiempos, 0.1, 20)
        
        # Variable objetivo: probabilidad de accidente basada en las características
        prob_accidente = []
        for i in range(n_samples):
            prob = 0.1  # Probabilidad base
            
            # Más horas = más probabilidad de accidente
            if horas[i] > 12:
                prob += 0.3
            elif horas[i] > 8:
                prob += 0.1
                
            # Nivel de seguridad afecta la probabilidad
            if niveles[i] == 'Bajo':
                prob += 0.4
            elif niveles[i] == 'Medio':
                prob += 0.2
                
            # Edad muy joven o muy mayor = más riesgo
            if edades[i] < 25 or edades[i] > 55:
                prob += 0.1
                
            # Poco tiempo en el puesto = más riesgo
            if tiempos[i] < 1:
                prob += 0.2
                
            prob = min(prob, 0.8)  # Máximo 80% de probabilidad
            prob_accidente.append(prob)
        
        # Generar variable objetivo binaria
        accidentes = np.random.binomial(1, prob_accidente, n_samples)
        
        # Crear DataFrame
        data = {
            'Horas trabajadas': horas,
            'Nivel de seguridad': niveles,
            'Edad': edades,
            'Tiempo en el puesto': tiempos,
            'Tiene accidente': accidentes
        }
        df = pd.DataFrame(data)
        
        logger.info(f"Dataset creado: {len(df)} registros")
        logger.info(f"Distribución de accidentes: {df['Tiene accidente'].value_counts().to_dict()}")
        
        return df
    
    def _generate_confusion_matrix_image(self, y_test, y_pred):
        """Genera la imagen de la matriz de confusión en base64."""
        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Matriz de Confusión', fontsize=14)
            plt.colorbar()
            
            # Añadir números en la matriz
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=12)
            
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['No Accidente', 'Accidente'])
            plt.yticks(tick_marks, ['No Accidente', 'Accidente'])
            plt.ylabel('Valor Real', fontsize=12)
            plt.xlabel('Predicción', fontsize=12)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return img_base64
        except Exception as e:
            logger.error(f"Error generando matriz de confusión: {e}")
            return ""
    
    def train_and_evaluate(self):
        """Entrena el modelo de regresión logística y devuelve las métricas."""
        try:
            logger.info("Iniciando entrenamiento del modelo...")
            
            # Crear dataset
            df = self._create_dataset()
            
            # Define X (variables predictoras) y y (variable objetivo)
            X = df[self.features_columns + self.categorical_features]
            y = df['Tiene accidente']

            # Preprocesamiento: Codificación de variables categóricas
            self.column_transformer = ColumnTransformer(
                [('encoder', OneHotEncoder(drop='first'), self.categorical_features)],
                remainder='passthrough'
            )
            
            logger.info("Aplicando transformaciones...")
            X_transformed = self.column_transformer.fit_transform(X)
            logger.info(f"Forma después de transformación: {X_transformed.shape}")

            # División de datos con estratificación
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Entrenamiento del modelo
            logger.info("Entrenando modelo de regresión logística...")
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Evaluación del modelo
            y_pred = self.model.predict(X_test)
            
            # 1. Exactitud (Accuracy)
            self.accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Accuracy del modelo: {self.accuracy:.4f}")
            
            # 2. Reporte de Clasificación (como HTML)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            self.report_html = report_df.to_html(classes='table table-striped', table_id='classification-report')

            # 3. Matriz de Confusión (como imagen Base64)
            self.cm_img_base64 = self._generate_confusion_matrix_image(y_test, y_pred)
            
            # Marcar como entrenado
            self.is_trained = True
            
            # Guardar el modelo
            self.save_model()
            
            logger.info("Modelo entrenado exitosamente!")
            return self.accuracy, self.report_html, self.cm_img_base64

        except Exception as e:
            logger.error(f"Error en el entrenamiento del modelo: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0, "<tr><td>Error al cargar el reporte.</td></tr>", ""
    
    def save_model(self):
        """Guarda el modelo entrenado en disco."""
        try:
            if not os.path.exists('models'):
                os.makedirs('models')
            
            model_data = {
                'model': self.model,
                'column_transformer': self.column_transformer,
                'accuracy': self.accuracy,
                'report_html': self.report_html,
                'cm_img_base64': self.cm_img_base64,
                'features_columns': self.features_columns,
                'categorical_features': self.categorical_features
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Modelo guardado en: {self.model_path}")
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    def load_model(self):
        """Carga el modelo desde disco si existe."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Cargando modelo desde: {self.model_path}")
                model_data = joblib.load(self.model_path)
                
                self.model = model_data['model']
                self.column_transformer = model_data['column_transformer']
                self.accuracy = model_data['accuracy']
                self.report_html = model_data['report_html']
                self.cm_img_base64 = model_data['cm_img_base64']
                self.features_columns = model_data['features_columns']
                self.categorical_features = model_data['categorical_features']
                self.is_trained = True
                
                logger.info("Modelo cargado exitosamente!")
                return True
            else:
                logger.info("No se encontró modelo guardado")
                return False
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def validate_input(self, features):
        """Valida los datos de entrada antes de hacer predicciones."""
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError(f"Se esperan 4 características en lista/tupla, recibidas: {len(features) if hasattr(features, '__len__') else 'formato inválido'}")
        
        horas_trabajadas, edad, tiempo_puesto, nivel_seguridad = features
        
        # Validar tipos y rangos
        try:
            horas_trabajadas = float(horas_trabajadas)
            if not (1 <= horas_trabajadas <= 16):
                raise ValueError("Las horas trabajadas deben estar entre 1 y 16")
                
            edad = float(edad)
            if not (18 <= edad <= 65):
                raise ValueError("La edad debe estar entre 18 y 65 años")
                
            tiempo_puesto = float(tiempo_puesto)
            if tiempo_puesto < 0.1:
                raise ValueError("El tiempo en el puesto debe ser mayor a 0.1 años")
                
            nivel_seguridad = str(nivel_seguridad)
            valid_levels = ['Bajo', 'Medio', 'Alto']
            if nivel_seguridad not in valid_levels:
                raise ValueError(f"Nivel de seguridad inválido: {nivel_seguridad}. Válidos: {valid_levels}")
                
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error en validación: {str(e)}")
        
        return horas_trabajadas, edad, tiempo_puesto, nivel_seguridad
    
    def predict(self, features):
        """
        Realiza la predicción usando el modelo entrenado.
        
        Args:
            features: Lista con [horas_trabajadas, edad, tiempo_puesto, nivel_seguridad]
        
        Returns:
            tuple: (prediccion, probabilidad)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        try:
            logger.info(f"=== INICIO PREDICCIÓN ===")
            logger.info(f"Features recibidas: {features}")
            
            # Validar entrada
            horas_trabajadas, edad, tiempo_puesto, nivel_seguridad = self.validate_input(features)
            
            logger.info(f"Características validadas:")
            logger.info(f"  - Horas trabajadas: {horas_trabajadas}")
            logger.info(f"  - Edad: {edad}")
            logger.info(f"  - Tiempo puesto: {tiempo_puesto}")
            logger.info(f"  - Nivel seguridad: {nivel_seguridad}")
            
            # Crear DataFrame en el orden correcto
            input_data = {
                'Horas trabajadas': horas_trabajadas,
                'Edad': edad,
                'Tiempo en el puesto': tiempo_puesto,
                'Nivel de seguridad': nivel_seguridad
            }
            
            # Crear DataFrame con las columnas en el orden del entrenamiento
            input_df = pd.DataFrame([input_data])
            input_df = input_df[self.features_columns + self.categorical_features]
            
            logger.info(f"DataFrame creado: {input_df.iloc[0].to_dict()}")
            
            # Transformar usando el column_transformer entrenado
            processed_features = self.column_transformer.transform(input_df)
            logger.info(f"Features procesadas: forma {processed_features.shape}")
            
            # Realizar predicción
            prediction_prob = self.model.predict_proba(processed_features)[0]
            probability_accident = prediction_prob[1]  # Probabilidad de accidente (clase 1)
            
            logger.info(f"Probabilidades: No accidente={prediction_prob[0]:.4f}, Accidente={prediction_prob[1]:.4f}")
            
            # Determinar predicción final
            prediction = 'Sí' if probability_accident >= 0.5 else 'No'
            
            logger.info(f"Predicción final: {prediction} (probabilidad: {probability_accident:.4f})")
            logger.info(f"=== FIN PREDICCIÓN ===")
            
            return prediction, float(probability_accident)
            
        except Exception as e:
            error_msg = f"Error en predicción: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)

# Instancia global del modelo (singleton)
_model_instance = None

def get_model_instance():
    """Obtiene la instancia singleton del modelo."""
    global _model_instance
    if _model_instance is None:
        _model_instance = LogisticRegressionModel()
    return _model_instance

def initialize_model():
    """Inicializa el modelo cargando desde disco o entrenando uno nuevo."""
    model = get_model_instance()
    
    # Intentar cargar modelo existente
    if model.load_model():
        return model.accuracy, model.report_html, model.cm_img_base64
    else:
        # Si no existe, entrenar uno nuevo
        return model.train_and_evaluate()

def predict_label(features):
    """Función de compatibilidad para predicciones."""
    model = get_model_instance()
    return model.predict(features)

def train_and_evaluate_model():
    """Función de compatibilidad para entrenamiento."""
    model = get_model_instance()
    return model.train_and_evaluate()

# Función auxiliar para testing
def test_prediction():
    """Función de prueba para verificar que todo funciona."""
    try:
        logger.info("=== TESTING PREDICTION ===")
        test_data = [8.0, 30.0, 2.5, 'Medio']
        prediction, probability = predict_label(test_data)
        logger.info(f"Test exitoso: {prediction}, {probability}")
        return True
    except Exception as e:
        logger.error(f"Test falló: {e}")
        return False

if __name__ == "__main__":
    # Test standalone
    print("Iniciando test standalone...")
    accuracy, report, img = initialize_model()
    print(f"Modelo inicializado con accuracy: {accuracy}")
    
    if accuracy > 0:
        success = test_prediction()
        print(f"Test de predicción: {'EXITOSO' if success else 'FALLÓ'}")
    else:
        print("No se pudo inicializar el modelo para testing")