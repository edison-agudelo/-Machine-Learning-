import pandas as pd # pip install pandas
import matplotlib.pyplot as mp # pip install matplotlib
from sklearn.linear_model import LinearRegression # pip install scikit-learn

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
} # Datos de ejemplo de horas de estudio y calificaciones finales

df= pd.DataFrame(data) # Crear un DataFrame de pandas

x = df[["Study Hours"]] # Variable independiente (horas de estudio)
y = df["Final Grade"] # Variable dependiente (calificación final)

model = LinearRegression() # Crear el modelo de regresión lineal
model.fit(x, y) # Ajustar el modelo a los datos

def calculate_grade(hours):
    result = model.predict([[hours]])[0] # Predecir la calificación basada en las horas de estudio
    return result