import pandas as pd

# Ruta del dataset
dataset_path = 'procesados/extended_data.csv'

# Cargar el dataset con todas las columnas
raw_dataset = pd.read_csv(dataset_path, dtype='unicode')

hashtag_columns = [f'hashtags_{i}_name' for i in range(48) if f'hashtags_{i}_name' in raw_dataset.columns]
# Columnas que queremos obtener del dataset
column_namesShort = ['diggCount','createTimeISO',
                #'collectCount','commentCount','shareCount',
                     #'input',
                     'isAd','isMuted',
                     #'musicMeta_musicName',
                     #'searchHashtag_name',
                     #'searchHashtag_views',
                     #'searchQuery',
                     'videoMeta_duration','ad',
                     'ecuador','publicidad','gratis','publi',
                     'anuncio','free','quito','ofertas','local',
                     'tienda','comercial','machala','promo',
                     'ambato','loja',
                     'promoecuador',
                     'comerciales',
                     'santodomingoecuador','guayaquil','cuenca']

# Seleccionar solo las columnas de column_namesShort
dataset = raw_dataset[column_namesShort]
#dataset = dataset.join(raw_dataset[hashtag_columns])
#dataset.columns = [col.replace('_', '_') for col in dataset.columns]

# Mostrar las últimas filas del dataset seleccionado
print(dataset.tail())

with open('procesados/extendido_digg_playcount.csv', 'wb') as file:
    dataset.to_csv(file, index=False)