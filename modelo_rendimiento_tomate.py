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

# Variables predictoras edáficas (evitando fuga de datos con Biomasa)
X = df[["pH", "Salinidad (dS/m)"]]
y = df["Rendimiento (t/ha)"]

# ==============================================================================
# 2. DEFINICIÓN DE MODELOS Y VALIDACIÓN CRUZADA (5-FOLD)
# ==============================================================================
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
resultados = []

for nombre, modelo in modelos.items():
  scores = cross_validate(
      modelo,
      X,
      y,
      cv=kf,
      scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
  )
  resultados.append({
      "Modelo": nombre,
      "R2_CV": round(scores["test_r2"].mean(), 3),
      "RMSE_CV (t/ha)": round(
          -scores["test_neg_root_mean_squared_error"].mean(), 2
      ),
      "MAE_CV (t/ha)": round(
          -scores["test_neg_mean_absolute_error"].mean(), 2
      ),
  })

df_resultados = pd.DataFrame(resultados)
print("=== COMPARACIÓN DE MODELOS (VALIDACIÓN CRUZADA 5-FOLD) ===")
print(df_resultados.to_string(index=False))

# ==============================================================================
# 3. ENTRENAMIENTO DEL MEJOR MODELO Y SUPERFICIE DE RESPUESTA
# ==============================================================================
mejor_modelo = modelos["Regresión Polinómica (Grado 2)"]
mejor_modelo.fit(X, y)

# Ecuación polinómica interpretable (en escala original)
poly_interpretable = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
)
poly_interpretable.fit(X, y)
nombres_vars = poly_interpretable.named_steps[
    "polynomialfeatures"
].get_feature_names_out()
coefs = poly_interpretable.named_steps["linearregression"].coef_
intercepto = poly_interpretable.named_steps["linearregression"].intercept_

print("\n=== ECUACIÓN AGRONÓMICA AJUSTADA ===")
print(f"Rendimiento (t/ha) = {intercepto:.2f}", end="")
for var, c in zip(nombres_vars, coefs):
  signo = "+" if c >= 0 else "-"
  print(f" {signo} {abs(c):.2f}*({var})", end="")
print("\n")

# ==============================================================================
# 4. VISUALIZACIÓN PARA EL REPOSITORIO
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: Superficie de respuesta (pH vs Salinidad)
ph_grid = np.linspace(df["pH"].min(), df["pH"].max(), 50)
sal_grid = np.linspace(
    df["Salinidad (dS/m)"].min(), df["Salinidad (dS/m)"].max(), 50
)
PH, SAL = np.meshgrid(ph_grid, sal_grid)
grid_df = pd.DataFrame({"pH": PH.ravel(), "Salinidad (dS/m)": SAL.ravel()})
Z = mejor_modelo.predict(grid_df).reshape(PH.shape)

contour = axes[0].contourf(PH, SAL, Z, levels=15, cmap="YlGnBu")
plt.colorbar(contour, ax=axes[0], label="Rendimiento Predicho (t/ha)")
axes[0].scatter(
    df["pH"],
    df["Salinidad (dS/m)"],
    c="black",
    edgecolors="white",
    s=40,
    alpha=0.8,
    label="Observaciones (n=60)",
)
axes[0].set_title("Superficie de Respuesta: Rendimiento vs pH y Salinidad")
axes[0].set_xlabel("pH del suelo")
axes[0].set_ylabel("Salinidad (dS/m)")
axes[0].legend()

# Panel 2: Valores Reales vs Predichos
y_pred = mejor_modelo.predict(X)
r2_cv = df_resultados.loc[
    df_resultados["Modelo"] == "Regresión Polinómica (Grado 2)", "R2_CV"
].values[0]
axes[1].scatter(y, y_pred, color="#2b5c8f", alpha=0.8, edgecolors="k")
axes[1].plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--",
    lw=2,
    label="Ajuste ideal (1:1)",
)
axes[1].set_title(f"Rendimiento Real vs. Predicho (R² CV = {r2_cv})")
axes[1].set_xlabel("Rendimiento Real (t/ha)")
axes[1].set_ylabel("Rendimiento Predicho (t/ha)")
axes[1].legend()

plt.tight_layout()
plt.savefig("superficie_respuesta_tomate.png", dpi=150)
plt.show()

# ==============================================================================
# 5. PREDICCIÓN PARA UN NUEVO ESCENARIO DE SUELO
# ==============================================================================
nuevo_suelo = pd.DataFrame({"pH": [6.4], "Salinidad (dS/m)": [1.8]})
prediccion_nueva = mejor_modelo.predict(nuevo_suelo)[0]
print(
    f"Predicción para suelo con pH 6.4 y CE 1.8 dS/m: {prediccion_nueva:.2f}"
    " t/ha"
)
