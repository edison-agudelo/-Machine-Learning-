# generate_dataset.py
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 150

tiempo_espera = np.random.uniform(1, 60, n)        # minutos
calidad_servicio = np.random.uniform(1, 10, n)     # 1-10

b0 = 20
b1 = -0.5    # penaliza tiempo de espera
b2 = 5.0     # beneficio por calidad
epsilon = np.random.normal(0, 5, n)

satisfaccion = b0 + b1*tiempo_espera + b2*calidad_servicio + epsilon

df = pd.DataFrame({
    "Tiempo_espera": tiempo_espera,
    "Calidad_servicio": calidad_servicio,
    "Satisfaccion": satisfaccion
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/clientes.csv", index=False)
print("Dataset guardado en data/clientes.csv (n={})".format(n))
