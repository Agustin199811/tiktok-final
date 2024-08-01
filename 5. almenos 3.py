import pandas as pd

# Cargar el archivo CSV
file_path = 'procesados/transformado_digg_playcount.csv'
df = pd.read_csv(file_path)

# Filtrar las filas que tienen al menos 3 columnas con el valor de 1
filtered_df = df[df.apply(lambda row: (row == 1).sum() >= 3, axis=1)]

# Guardar el resultado en un nuevo archivo CSV (opcional)
filtered_df.to_csv('procesados/filtered_rows.csv', index=False)

# Mostrar las primeras filas del dataframe filtrado
print(filtered_df.head())
