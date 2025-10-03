import numpy as np
import pandas as pd
import os

print("="*60)
print("GENERADOR DE DATASET - ROTACIÓN DE EMPLEADOS")
print("="*60)

RND = 42
np.random.seed(RND)

# Crear carpeta data si no existe
os.makedirs("data", exist_ok=True)

n = 1000  # filas

print(f"\nGenerando {n} registros de empleados...")

# Features
anios_empresa = np.clip(np.round(np.random.exponential(scale=3.0, size=n), 1), 0, 40)
nivel_satisfaccion = np.clip(np.round(np.random.normal(loc=0.6, scale=0.2, size=n), 3), 0, 1)
salario = np.round(np.random.normal(loc=3000, scale=800, size=n), 2)
n_capacitaciones = np.random.poisson(lam=2, size=n)
eval_desempeno = np.clip(np.round(np.random.normal(loc=3.5, scale=0.8, size=n), 2), 1, 5)

# Generar probabilidad base y label sintético
prob_base = (
    0.4 * (1 - nivel_satisfaccion) +
    0.2 * (1 / (1 + anios_empresa)) +
    0.15 * (1 - (eval_desempeno - 1) / 4) +
    0.1 * (1 - np.clip((salario - salario.min()) / (salario.max() - salario.min()), 0, 1)) +
    0.15 * (1 - (n_capacitaciones / (n_capacitaciones.max() + 1)))
)
prob_base = (prob_base - prob_base.min()) / (prob_base.max() - prob_base.min())

# Añadir ruido y convertir a etiquetas Alta/Baja
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

print("\nInsertando valores faltantes (5% en algunas columnas)...")

# Insertar valores faltantes aleatoriamente (5%)
for col in ["nivel_satisfaccion", "salario", "eval_desempeno"]:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan
    n_missing = mask.sum()
    print(f"  - {col}: {n_missing} valores faltantes")

out_path = "data/rotacion_empleados.csv"
df.to_csv(out_path, index=False, encoding='utf-8')

print(f"\n{'='*60}")
print(f"DATASET GENERADO EXITOSAMENTE")
print(f"{'='*60}")
print(f"Ubicación: {out_path}")
print(f"Total de filas: {len(df)}")
print(f"\nDistribución de rotación:")
print(df['rotacion'].value_counts())
print(f"\n{'='*60}")
print("Ahora puedes ejecutar: python app.py")
print(f"{'='*60}\n")