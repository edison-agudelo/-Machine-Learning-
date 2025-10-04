import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

RND = 42
ROOT = os.path.dirname(os.path.dirname(__file__)) if __name__ != "__main__" else os.path.dirname(__file__) + "/.."
MODEL_DIR = os.path.join(ROOT, "models")
STATIC_IMG_DIR = os.path.join(ROOT, "static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_rotacion.joblib")
CM_IMG_PATH = os.path.join(STATIC_IMG_DIR, "confusion_matrix_rf.png")

FEATURE_COLS = ['anios_empresa','nivel_satisfaccion','salario','n_capacitaciones','eval_desempeno']
TARGET_COL = 'rotacion'  # 'Alta'/'Baja'

def load_data(path="data/rotacion_empleados.csv"):
    df = pd.read_csv(path)
    return df

def build_pipeline():
    # Imputar (median) + escalado para numéricos, RandomForest (no requiere escalado pero lo ponemos para consistencia con otros)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RND))
    ])
    return pipe

def train_and_evaluate_model(csv_path="data/rotacion_empleados.csv", save=True):
    df = load_data(csv_path)
    df = df.copy()
    # transformar etiquetas 'Alta'/'Baja' a 1/0
    if df[TARGET_COL].dtype == object or df[TARGET_COL].dtype.name == 'category':
        df[TARGET_COL] = df[TARGET_COL].map({'Baja':0, 'Alta':1})

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RND
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None

    acc = round(accuracy_score(y_test, y_pred), 4)
    report = classification_report(y_test, y_pred, target_names=['Baja','Alta'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC
    roc = None
    if y_proba is not None:
        try:
            roc = round(roc_auc_score(y_test, y_proba), 4)
        except Exception:
            roc = None

    # Guardar modelo
    if save:
        joblib.dump(pipe, MODEL_PATH)

    # Guardar matriz de confusión como imagen (2x2)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Baja (0)','Pred Alta (1)'],
                yticklabels=['Real Baja (0)','Real Alta (1)'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Random Forest')
    plt.tight_layout()
    plt.savefig(CM_IMG_PATH)
    plt.close()

    metrics = {
        'accuracy': acc,
        'roc_auc': roc
    }

    return metrics, report, CM_IMG_PATH

def evaluate():
    """Función expuesta para evaluación (genera métricas e imágenes)"""
    return train_and_evaluate_model(save=False)

def predict_label(input_dict, threshold=0.5):
    """
    input_dict: dict con keys en FEATURE_COLS (valores numéricos)
    retorna {'label': 'Alta'/'Baja', 'probability': float}
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelo no encontrado. Entrena primero (train_and_evaluate_model).")

    pipe = joblib.load(MODEL_PATH)
    X = pd.DataFrame([input_dict], columns=FEATURE_COLS)
    # Imputación: si faltan valores SimpleImputer manejará en pipeline
    proba = pipe.predict_proba(X)[0,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None
    if proba is None:
        # usar decision_function con calibración aproximada
        dec = pipe.decision_function(X)[0]
        proba = 1/(1+np.exp(-dec))

    label = "Alta" if proba >= threshold else "Baja"
    return {'label': label, 'probability': round(float(proba), 4)}

# Permitir ejecutar desde CLI para entrenar
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        print("Entrenando Random Forest con data/rotacion_empleados.csv ...")
        metrics, report, cm_path = train_and_evaluate_model(save=True)
        print("Metrics:", metrics)
        print("Matriz guardada en:", cm_path)
    else:
        print("Comando no reconocido. Usar: python random_forest_rotacion.py train")