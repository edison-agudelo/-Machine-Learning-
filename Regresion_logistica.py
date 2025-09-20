import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# Variables globales
scaler = None
logistic_model = None

def train_and_evaluate_model():
    global scaler, logistic_model
    
    # 1. Cargar datos
    data = pd.read_csv('dataset/data.csv')
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    # 2. Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Modelo
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_scaled, y_train)

    # 5. Predicciones
    y_pred = logistic_model.predict(X_test_scaled)

    # 6. Métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 7. Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusión')

    # Guardar imagen en carpeta static
    cm_path = os.path.join("static", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return accuracy, report, "confusion_matrix.png"


def predict_label(input_data):
    """
    input_data = [horas_trabajadas, edad, tiempo_puesto, nivel_seguridad_text]
    Aquí deberías transformar input_data según tu dataset real.
    """

    global scaler, logistic_model

    if logistic_model is None or scaler is None:
        raise ValueError("El modelo no ha sido entrenado. Llama primero a train_and_evaluate_model().")

    # Ejemplo: convertir seguridad de texto a número
    nivel_map = {"Bajo": 0, "Medio": 1, "Alto": 2}
    horas_trabajadas, edad, tiempo_puesto, nivel_seguridad_text = input_data
    nivel_seguridad = nivel_map[nivel_seguridad_text]

    # Preparar input para el modelo
    X_new = np.array([[horas_trabajadas, edad, tiempo_puesto, nivel_seguridad]])
    X_new_scaled = scaler.transform(X_new)

    # Predicción
    prediction = logistic_model.predict(X_new_scaled)[0]
    probability = logistic_model.predict_proba(X_new_scaled)[0][prediction]

    return prediction, probability
