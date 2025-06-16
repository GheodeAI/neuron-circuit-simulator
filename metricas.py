import pandas as pd
import numpy as np

# Cargar los dos archivos, asegurándose de que NA se interprete como NaN
df_py = pd.read_csv("salida_final_python2.csv", sep=';', header=None, na_values="NA")
df_r  = pd.read_csv("salida_final_r2.csv", sep=';', header=None, na_values="NA")

# Recortar número de filas
min_filas = min(len(df_py), len(df_r))
df_py = df_py.iloc[:min_filas]
df_r = df_r.iloc[:min_filas]

# Asegurar que tienen la misma forma
assert df_py.shape == df_r.shape, "Los DataFrames tienen formas distintas"

# Crear un DataFrame para las métricas
resultados = []

# Calcular métricas columna a columna
for col in df_py.columns:
    x = df_py[col]
    y = df_r[col]

    # Eliminar NaN en común
    mask = ~(x.isna() | y.isna())
    x_valid = x[mask]
    y_valid = y[mask]

    # Si no hay datos válidos, saltamos
    if len(x_valid) == 0:
        continue

    # Métricas
    mae = np.mean(np.abs(x_valid - y_valid))                    # Error absoluto medio
    mse = np.mean((x_valid - y_valid) ** 2)                     # Error cuadrático medio
    rmse = np.sqrt(mse)                                         # Raíz del ECM
    corr = np.corrcoef(x_valid, y_valid)[0, 1]                  # Correlación de Pearson
    diff_mean = abs(np.mean(x_valid) - np.mean(y_valid))       # Diferencia de medias
    diff_std = abs(np.std(x_valid) - np.std(y_valid))          # Diferencia de std

    resultados.append({
        'Columna': col,
        'MAE': mae,
        'RMSE': rmse,
        'Correlación': corr,
        'Δ Media': diff_mean,
        'Δ STD': diff_std,
        'Datos comparados': len(x_valid)
    })

# Mostrar resultados como DataFrame
res_df = pd.DataFrame(resultados)
print(res_df)

res_df.to_csv("resultados_metricas2.csv", index=False, header=False, sep=';', na_rep='NaN')