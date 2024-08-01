import pandas as pd

# Cargar el archivo CSV
file_path = 'procesados/extendido_digg_playcount.csv'
df = pd.read_csv(file_path)

# Convertir columnas booleanas a 1 y 0
df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

# Guardar el DataFrame modificado en un nuevo archivo CSV
output_file_path = 'procesados/transformado_digg_playcount.csv'
df.to_csv(output_file_path, index=False)

print(f'Archivo transformado guardado como {output_file_path}')
