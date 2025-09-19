import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#cargar los datos desde un archivo csv
data = pd.read_csv('dataset\data.csv')


#explorar el conjunto de datos 
print(data.head()) # Para ver las primeras filas
print(data.info())
print(data.describe())     
print(data.columns)     # Para ver los nombres de todas las columnas


#separar las variables independientes (x) y la dependiente(y)
x = data.drop ('HeartDisease', axis=1) #supon que 'target es la columna de eqtiueta/clase
y = data['HeartDisease']

#dividir el datashet en conjunto de entrenamiento y prueba (80% entrenamiento, 20prueba)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42) 

#estandarizar los datos (opcional pero recomendable para regresion logistica)

scaler = StandardScaler()
x_train_scaled = scaler.fit_trasnform(x_train)
x_test_sacled = scaler.transform(x_test)

#crear el modelo de regresion logistica
logistic_model = LogisticRegression()

#entrenar el modelo 
logistic_model.fit(x_train_scaled, y_train)

#realizar predicciones con el conjunto de prueba
y_pred = logistic_model.predict(x_test_sacled)

#crear la matriz de connfuncion
conf_matrix = confusion_matrix(y_test, y_pred)

#visualizar la matriz de confunsion 
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', camp='blues', cbar=False)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confunsion matrix')
plt.show()

#implementar el reporte de clasificacion 
print(classification_report(y_test, y_pred))

#imprimir la exactitud del modelo 
accuracy = accuracy_score(y_test, y_pred)
print(f'exactitud del modelo: {accuracy * 100: .2f}%')