import pandas as pd
import numpy as np

# Cargar los dos archivos
df_py = pd.read_csv("salida_final_python5.csv", sep=';', header=None, na_values="NA")
df_r  = pd.read_csv("salida_final_r5.csv", sep=';', header=None, na_values="NA")

# Recortar número de filas, ambos dataframes deben ser iguales en forma y longitud
min_filas = min(len(df_py), len(df_r))
df_py = df_py.iloc[:min_filas]
df_r = df_r.iloc[:min_filas]
assert df_py.shape == df_r.shape, "Los DataFrames tienen formas distintas"

# Crear un DataFrame para las métricas
resultados = []
resultados2 = []

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

    # Cálculo de métricas
    mae = np.mean(np.abs(x_valid - y_valid))                    # Error absoluto medio
    mse = np.mean((x_valid - y_valid) ** 2)                     # Error cuadrático medio
    rmse = np.sqrt(mse)                                         # Raíz del ECM
    corr = np.corrcoef(x_valid, y_valid)[0, 1]                  # Correlación de Pearson

    mean_py = abs(np.mean(x_valid))           # Media Python
    mean_r = abs(np.mean(y_valid))           # Media R
    std_py = abs(np.std(x_valid))            # Desviación típica Python
    std_r = abs(np.std(y_valid))            # Desviación típica R


    resultados.append({
        'Columna': col,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Correlación': corr,
        'Datos comparados': len(x_valid)
    })

    resultados2.append({
        'Columna': col,
        'Media Python': mean_py,
        'Media R': mean_r,
        'Desv. típica Python': std_py,
        'Desv. típica R': std_r,
    })

res_df = pd.DataFrame(resultados)
res2_df = pd.DataFrame(resultados2)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#print(res_df)
print(res2_df)

#res_df.to_csv("resultados_metricas5.csv", index=False, header=False, sep=';', na_rep='NaN')
res2_df.to_csv("resultados2_metricas5.csv", index=False, header=False, sep=';', na_rep='NaN')



'''
diff_mean = abs(np.mean(x_valid) - np.mean(y_valid))  # Diferencia de medias    QUITAR!!!!!!!!!!!!!!!!!!!!!!!!!
diff_std = abs(np.std(x_valid) - np.std(y_valid))  # Diferencia de std     QUITAR!!!!!!!!!!!!!!!!!!!!!!!!!!

'Δ Media': diff_mean,
'Δ STD': diff_std,
'''