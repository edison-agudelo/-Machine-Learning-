import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 1. Cargar dataset
# ===============================
def cargar_datos():
    try:
        df = pd.read_csv("data/clientes.csv")  # Ruta a tu dataset
    except FileNotFoundError:
        # Generar dataset ficticio si no existe el archivo
        np.random.seed(42)
        df = pd.DataFrame({
            "tiempo_espera": np.random.randint(1, 60, 50),   # minutos
            "calidad_servicio": np.random.randint(1, 11, 50), # 1 a 10
        })
        df["satisfaccion"] = (
            10 - 0.1*df["tiempo_espera"] + 0.5*df["calidad_servicio"] 
            + np.random.normal(0, 1, 50)
        )
        df.to_csv("data/clientes.csv", index=False)
    return df

# ===============================
# 2. Entrenar modelo
# ===============================
def entrenar_modelo(df):
    X = df[["tiempo_espera", "calidad_servicio"]]
    y = df["satisfaccion"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    predicciones = modelo.predict(X)

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y, predicciones))
    r2 = r2_score(y, predicciones)

    metrics = {"rmse": rmse, "r2": r2}

    return modelo, metrics, predicciones

# ===============================
# 3. Generar gráfico
# ===============================
def generar_grafico(df, predicciones):
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(
        df["tiempo_espera"], 
        df["satisfaccion"], 
        c=df["calidad_servicio"], cmap="viridis", s=60
    )
    plt.colorbar(scatter, label="Calidad del servicio")
    plt.plot(df["tiempo_espera"], predicciones, color="red", linewidth=2)
    plt.xlabel("Tiempo de espera (min)")
    plt.ylabel("Satisfacción")
    plt.title("Regresión lineal: satisfacción de clientes")

    # Convertir a base64 para mostrar en HTML
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{plot_url}"

# ===============================
# 4. Predicción individual
# ===============================
def predecir(modelo, tiempo, calidad):
    X_nuevo = np.array([[tiempo, calidad]])
    return modelo.predict(X_nuevo)[0]

# ===============================
# 5. Función principal
# ===============================
def ejecutar_modelo(tiempo=None, calidad=None):
    df = cargar_datos()
    modelo, metrics, predicciones = entrenar_modelo(df)
    plot_url = generar_grafico(df, predicciones)

    prediccion = None
    if tiempo is not None and calidad is not None:
        prediccion = predecir(modelo, tiempo, calidad)

    return {
        "metrics": metrics,
        "plot_url": plot_url,
        "prediccion": prediccion,
        "data": df.head().to_html(classes="table table-striped", index=False)
    }
