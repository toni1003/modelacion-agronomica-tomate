import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
ARCHIVO_DATOS = "Datos_Agronomicos_Tomate.xlsx"
df = pd.read_excel(ARCHIVO_DATOS, sheet_name="Datos_Tomate")

X = df[["pH", "Salinidad (dS/m)"]]
y_rend = df["Rendimiento (t/ha)"]
y_biom = df["Biomasa (g/planta)"]

modelos = {
    "Regresión Lineal": LinearRegression(),
    "Regresión Polinómica (Grado 2)": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    ),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def evaluar_y_graficar(y_target, nombre_var, unidad, cmap_color, archivo_salida):
  resultados = []
  for nombre, modelo in modelos.items():
    scores = cross_validate(
        modelo,
        X,
        y_target,
        cv=kf,
        scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
    )
    resultados.append({
        "Modelo": nombre,
        "R2_CV": round(scores["test_r2"].mean(), 3),
        f"RMSE_CV ({unidad})": round(
            -scores["test_neg_root_mean_squared_error"].mean(), 2
        ),
        f"MAE_CV ({unidad})": round(
            -scores["test_neg_mean_absolute_error"].mean(), 2
        ),
    })

  df_res = pd.DataFrame(resultados)
  print(f"\n=== COMPARACIÓN DE MODELOS: {nombre_var.upper()} (5-FOLD CV) ===")
  print(df_res.to_string(index=False))

  # Entrenar mejor modelo (Polinómico Grado 2)
  mejor_modelo = modelos["Regresión Polinómica (Grado 2)"]
  mejor_modelo.fit(X, y_target)

  # Ecuación interpretable en escala original
  poly_raw = make_pipeline(
      PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
  )
  poly_raw.fit(X, y_target)
  vars_out = poly_raw.named_steps[
      "polynomialfeatures"
  ].get_feature_names_out()
  coefs = poly_raw.named_steps["linearregression"].coef_
  intercepto = poly_raw.named_steps["linearregression"].intercept_

  print(f"\n=== ECUACIÓN AGRONÓMICA: {nombre_var.upper()} ===")
  print(f"{nombre_var} ({unidad}) = {intercepto:.2f}", end="")
  for v, c in zip(vars_out, coefs):
    signo = "+" if c >= 0 else "-"
    print(f" {signo} {abs(c):.2f}*({v})", end="")
  print("\n")

  # Gráficos: Superficie de respuesta + Real vs Predicho
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))

  ph_grid = np.linspace(df["pH"].min(), df["pH"].max(), 50)
  sal_grid = np.linspace(
      df["Salinidad (dS/m)"].min(), df["Salinidad (dS/m)"].max(), 50
  )
  PH, SAL = np.meshgrid(ph_grid, sal_grid)
  grid_df = pd.DataFrame({"pH": PH.ravel(), "Salinidad (dS/m)": SAL.ravel()})
  Z = mejor_modelo.predict(grid_df).reshape(PH.shape)

  contour = axes[0].contourf(PH, SAL, Z, levels=15, cmap=cmap_color)
  plt.colorbar(contour, ax=axes[0], label=f"{nombre_var} Predicha ({unidad})")
  axes[0].scatter(
      df["pH"],
      df["Salinidad (dS/m)"],
      c="black",
      edgecolors="white",
      s=40,
      alpha=0.8,
      label="Observaciones (n=60)",
  )
  axes[0].set_title(f"Superficie de Respuesta: {nombre_var} vs pH y Salinidad")
  axes[0].set_xlabel("pH del suelo")
  axes[0].set_ylabel("Salinidad (dS/m)")
  axes[0].legend()

  y_pred = mejor_modelo.predict(X)
  r2_cv = df_res.loc[
      df_res["Modelo"] == "Regresión Polinómica (Grado 2)", "R2_CV"
  ].values[0]
  color_puntos = "#2b5c8f" if nombre_var == "Rendimiento" else "#2e7d32"
  axes[1].scatter(y_target, y_pred, color=color_puntos, alpha=0.8, edgecolors="k")
  axes[1].plot(
      [y_target.min(), y_target.max()],
      [y_target.min(), y_target.max()],
      "r--",
      lw=2,
      label="Ajuste ideal (1:1)",
  )
  axes[1].set_title(f"{nombre_var} Real vs. Predicha (R² CV = {r2_cv})")
  axes[1].set_xlabel(f"{nombre_var} Real ({unidad})")
  axes[1].set_ylabel(f"{nombre_var} Predicha ({unidad})")
  axes[1].legend()

  plt.tight_layout()
  plt.savefig(archivo_salida, dpi=150)
  plt.close()

  # Predicción para escenario óptimo de ejemplo
  nuevo_suelo = pd.DataFrame({"pH": [6.4], "Salinidad (dS/m)": [1.8]})
  pred = mejor_modelo.predict(nuevo_suelo)[0]
  print(
      f"Predicción {nombre_var} (pH 6.4, CE 1.8 dS/m): {pred:.2f} {unidad}"
  )


# ==============================================================================
# 2. EJECUCIÓN PARA AMBAS VARIABLES OBJETIVO POR SEPARADO
# ==============================================================================
evaluar_y_graficar(
    y_rend, "Rendimiento", "t/ha", "YlGnBu", "superficie_respuesta_tomate.png"
)
evaluar_y_graficar(
    y_biom, "Biomasa", "g/planta", "YlGn", "superficie_respuesta_biomasa.png"
)
