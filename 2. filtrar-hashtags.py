import pandas as pd

# Cargar el archivo CSV
file_path = 'procesados/data_162col.csv'  # Reemplaza con la ruta correcta
df = pd.read_csv(file_path)

# Lista de hashtags a filtrar
hashtags = [
    "ad", "ecuador", "publicidad", "gratis", "publi", "anuncio", "free", "quito",
    "ofertas", "local", "tienda", "comercial", "machala", "promo", "ambato",
    "loja",
    "promoecuador",
    "comerciales", "santodomingoecuador", "guayaquil", "cuenca"
]

# Lista de columnas de hashtags
hashtag_columns = [f'hashtags_{i}_name' for i in range(48) if f'hashtags_{i}_name' in df.columns]

# Añadir una columna por cada hashtag
for hashtag in hashtags:
    df[hashtag] = df[hashtag_columns].apply(lambda row: hashtag in row.values, axis=1)

# Guardar el dataframe con las nuevas columnas en un nuevo archivo CSV
output_file_path_extended = 'procesados/extended_data.csv'  # Reemplaza con la ruta correcta
df.to_csv(output_file_path_extended, index=False)

print(f"Proceso completado. El dataframe con las nuevas columnas se ha guardado en {output_file_path_extended}")
