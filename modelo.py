import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io, base64

# Dataset de ejemplo
df = pd.DataFrame({
    "tiempo": [5, 10, 15, 20, 25],
    "calidad": [7, 8, 5, 6, 9],
    "satisfaccion": [8, 7, 6, 7, 8]
})

# Entrenar modelo
X = df[["tiempo", "calidad"]]
y = df["satisfaccion"]
model = LinearRegression()
model.fit(X, y)

# Métricas
preds = model.predict(X)
metrics = {
    "rmse": mean_squared_error(y, preds) ** 0.5,
    "r2": r2_score(y, preds)
}

def ejecutar_modelo(tiempo=None, calidad=None):
    prediccion = None
    plot_df = df.copy()

    if tiempo is not None and calidad is not None:
        prediccion = model.predict(np.array([[tiempo, calidad]]))[0]
        plot_df = pd.concat([plot_df, pd.DataFrame({"tiempo":[tiempo], "calidad":[calidad], "satisfaccion":[prediccion]})], ignore_index=True)

    plt.figure(figsize=(6,4))
    # colores: rojo para la predicción nueva, azul para los demás
    colors = ['red' if tiempo is not None and calidad is not None and i == len(plot_df)-1 else 'blue' for i in range(len(plot_df))]
    plt.scatter(plot_df["tiempo"], plot_df["satisfaccion"], c=colors)

    # línea de regresión con todos los puntos (incluyendo predicción)
    X_plot = plot_df[["tiempo"]]
    y_plot = plot_df["satisfaccion"]
    coef = np.polyfit(X_plot["tiempo"], y_plot, 1)
    y_fit = np.polyval(coef, X_plot["tiempo"])
    plt.plot(X_plot["tiempo"], y_fit, color="green", label="Línea de regresión")

    plt.xlabel("Tiempo de espera")
    plt.ylabel("Satisfacción")
    plt.title("Regresión Lineal")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    plt.close()

    data_html = plot_df.to_html(classes="table table-striped", index=False)

    return {
        "prediccion": prediccion,
        "metrics": metrics,
        "plot_url": plot_url,
        "data": data_html
    }

def generar_conceptos():
    df_plot = pd.DataFrame({"tiempo": [5, 10, 15, 20, 25], "satisfaccion": [8, 7, 6, 7, 8]})
    coef = np.polyfit(df_plot["tiempo"], df_plot["satisfaccion"], 1)
    y_fit = np.polyval(coef, df_plot["tiempo"])

    plt.figure(figsize=(6,4))
    plt.scatter(df_plot["tiempo"], df_plot["satisfaccion"], color="blue")
    plt.plot(df_plot["tiempo"], y_fit, color="red")
    plt.xlabel("Tiempo de espera")
    plt.ylabel("Satisfacción")
    plt.title("Ejemplo de regresión lineal")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    plot_conceptos = f"data:image/png;base64,{plot_data}"

    return {"plot_conceptos": plot_conceptos}
