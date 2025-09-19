import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cargar los datos desde un archivo CSV
data = pd.read_csv('dataset/data.csv')

# Explorar el conjunto de datos 
print(data.head())       # Ver primeras filas
print(data.info())       # Info general del dataset
print(data.describe())   # Estadísticas descriptivas
print(data.columns)      # Nombres de todas las columnas

# Separar variables independientes (X) y dependiente (y)
X = data.drop('HeartDisease', axis=1)   # Cambié a mayúscula por convención
y = data['HeartDisease']

# Dividir el dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarizar los datos (muy recomendable para regresión logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión logística
logistic_model = LogisticRegression()

# Entrenar el modelo 
logistic_model.fit(X_train_scaled, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = logistic_model.predict(X_test_scaled)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación 
print(classification_report(y_test, y_pred))

# Imprimir la exactitud del modelo 
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
