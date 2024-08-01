import pandas as pd

# Ruta del dataset
dataset_path = 'procesados/dummies_transformado_digg_playcount.csv'

# Cargar el dataset con todas las columnas
raw_dataset = pd.read_csv(dataset_path, dtype='unicode')
ft_df = pd.read_csv('procesados/best-ft.csv', dtype='unicode')

ft_columns = ft_df.iloc[:,0]

print(ft_columns.head())

dataset = raw_dataset[ft_columns]
dataset = dataset.join(raw_dataset['diggCount'])
# Mostrar las últimas filas del dataset seleccionado
print(dataset.tail())

with open('procesados/ft_diggcount.csv', 'wb') as file:
    dataset.to_csv(file, index=False)