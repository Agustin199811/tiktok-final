import pandas as pd

# Ruta del dataset
dataset_path = 'dataset_free-tiktok-scraper_2024-07-11_03-20-24-387.csv'

# Columnas que queremos obtener del dataset
column_namesShort = ['diggCount', 'createTimeISO',
                     #'collectCount','commentCount','shareCount',
                    "hashtags/0/name","hashtags/0/title",
                    "hashtags/1/name","hashtags/1/title",
                    "hashtags/2/name","hashtags/2/title",
                    "hashtags/3/name","hashtags/3/title",
                    "hashtags/4/name","hashtags/4/title",
                    "hashtags/5/name","hashtags/5/title",
                    "hashtags/6/name","hashtags/6/title",
                    "hashtags/7/name","hashtags/7/title",
                    "hashtags/8/name","hashtags/8/title",
                    "hashtags/9/name","hashtags/9/title",
                    "hashtags/10/name","hashtags/10/title",
                    "hashtags/11/name","hashtags/11/title",
                    "hashtags/12/name","hashtags/12/title",
                    "hashtags/13/name","hashtags/13/title",
                    "hashtags/14/name","hashtags/14/title",
                    "hashtags/15/name","hashtags/15/title",
                    "hashtags/16/name","hashtags/16/title",
                    "hashtags/17/name","hashtags/17/title",
                    "hashtags/18/name","hashtags/18/title",
                    "hashtags/19/name","hashtags/19/title",
                    "hashtags/20/name","hashtags/20/title",
                    "hashtags/21/name","hashtags/21/title",
                    "hashtags/22/name","hashtags/22/title",
                    "hashtags/23/name","hashtags/23/title",
                    "hashtags/24/name","hashtags/24/title",
                    "hashtags/25/name","hashtags/25/title",
                    "hashtags/26/name","hashtags/26/title",
                    "hashtags/27/name","hashtags/27/title",
                    "hashtags/28/name","hashtags/28/title",
                    "hashtags/29/name","hashtags/29/title",
                    "hashtags/30/name","hashtags/30/title",
                    "hashtags/31/name","hashtags/31/title",
                    "hashtags/32/name","hashtags/32/title",
                    "hashtags/33/name","hashtags/33/title",
                    "hashtags/34/name","hashtags/34/title",
                    "hashtags/35/name","hashtags/35/title",
                    "hashtags/36/name","hashtags/36/title",
                    "hashtags/37/name","hashtags/37/title",
                    "hashtags/38/name","hashtags/38/title",
                    "hashtags/39/name","hashtags/39/title",
                    "hashtags/40/name","hashtags/40/title",
                    "hashtags/41/name","hashtags/41/title",
                    "hashtags/42/name","hashtags/42/title",
                    "hashtags/43/name","hashtags/43/title",
                    "hashtags/44/name","hashtags/44/title",
                    "hashtags/45/name","hashtags/45/title",
                    "hashtags/46/name","hashtags/47/name",
                    "hashtags/47/title","hashtags/48/name",
                    "hashtags/48/title",
                     'input','isAd','isMuted','musicMeta/musicName',
                     #'playCount',
                     'searchHashtag/name','searchHashtag/views','videoMeta/duration']

# Cargar el dataset completo con los nombres de todas las columnas
column_names = ['collectCount','commentCount','createTime','createTimeISO','diggCount','effectStickers/0/ID','effectStickers/0/name','effectStickers/0/stickerStats/useCount','effectStickers/1/ID','effectStickers/1/name','effectStickers/1/stickerStats/useCount','hashtags/0/cover','hashtags/0/id','hashtags/0/name','hashtags/0/title','hashtags/1/cover','hashtags/1/id','hashtags/1/name','hashtags/1/title','hashtags/2/cover','hashtags/2/id','hashtags/2/name','hashtags/2/title','hashtags/3/cover','hashtags/3/id','hashtags/3/name','hashtags/3/title','hashtags/4/cover','hashtags/4/id','hashtags/4/name','hashtags/4/title','hashtags/5/cover','hashtags/5/id','hashtags/5/name','hashtags/5/title','hashtags/6/cover','hashtags/6/id','hashtags/6/name','hashtags/6/title','hashtags/7/cover','hashtags/7/id','hashtags/7/name','hashtags/7/title','hashtags/8/cover','hashtags/8/id','hashtags/8/name','hashtags/8/title','hashtags/9/cover','hashtags/9/id','hashtags/9/name','hashtags/9/title','hashtags/10/cover','hashtags/10/id','hashtags/10/name','hashtags/10/title','hashtags/11/cover','hashtags/11/id','hashtags/11/name','hashtags/11/title','hashtags/12/cover','hashtags/12/id','hashtags/12/name','hashtags/12/title','hashtags/13/cover','hashtags/13/id','hashtags/13/name','hashtags/13/title','hashtags/14/cover','hashtags/14/id','hashtags/14/name','hashtags/14/title','hashtags/15/cover','hashtags/15/id','hashtags/15/name','hashtags/15/title','hashtags/16/cover','hashtags/16/id','hashtags/16/name','hashtags/16/title','hashtags/17/cover','hashtags/17/id','hashtags/17/name','hashtags/17/title','hashtags/18/cover','hashtags/18/id','hashtags/18/name','hashtags/18/title','hashtags/19/cover','hashtags/19/id','hashtags/19/name','hashtags/19/title','hashtags/20/cover','hashtags/20/id','hashtags/20/name','hashtags/20/title','hashtags/21/cover','hashtags/21/id','hashtags/21/name','hashtags/21/title','hashtags/22/cover','hashtags/22/id','hashtags/22/name','hashtags/22/title','hashtags/23/cover','hashtags/23/id','hashtags/23/name','hashtags/23/title','hashtags/24/cover','hashtags/24/id','hashtags/24/name','hashtags/24/title','hashtags/25/cover','hashtags/25/id','hashtags/25/name','hashtags/25/title','hashtags/26/cover','hashtags/26/id','hashtags/26/name','hashtags/26/title','hashtags/27/cover','hashtags/27/id','hashtags/27/name','hashtags/27/title','hashtags/28/cover','hashtags/28/id','hashtags/28/name','hashtags/28/title','hashtags/29/cover','hashtags/29/id','hashtags/29/name','hashtags/29/title','hashtags/30/cover','hashtags/30/id','hashtags/30/name','hashtags/30/title','hashtags/31/cover','hashtags/31/id','hashtags/31/name','hashtags/31/title','hashtags/32/cover','hashtags/32/id','hashtags/32/name','hashtags/32/title','hashtags/33/cover','hashtags/33/id','hashtags/33/name','hashtags/33/title','hashtags/34/cover','hashtags/34/id','hashtags/34/name','hashtags/34/title','hashtags/35/cover','hashtags/35/id','hashtags/35/name','hashtags/35/title','hashtags/36/cover','hashtags/36/id','hashtags/36/name','hashtags/36/title','hashtags/37/cover','hashtags/37/id','hashtags/37/name','hashtags/37/title','hashtags/38/cover','hashtags/38/id','hashtags/38/name','hashtags/38/title','hashtags/39/cover','hashtags/39/id','hashtags/39/name','hashtags/39/title','hashtags/40/cover','hashtags/40/id','hashtags/40/name','hashtags/40/title','hashtags/41/cover','hashtags/41/id','hashtags/41/name','hashtags/41/title','hashtags/42/cover','hashtags/42/id','hashtags/42/name','hashtags/42/title','hashtags/43/cover','hashtags/43/id','hashtags/43/name','hashtags/43/title','hashtags/44/cover','hashtags/44/id','hashtags/44/name','hashtags/44/title','hashtags/45/cover','hashtags/45/id','hashtags/45/name','hashtags/45/title','hashtags/46/id','hashtags/46/name','hashtags/47/cover','hashtags/47/id','hashtags/47/name','hashtags/47/title','hashtags/48/cover','hashtags/48/id','hashtags/48/name','hashtags/48/title','id','input','isAd','isMuted','isPinned','isSlideshow','mentions/0','mentions/1','mentions/2','mentions/3','mentions/4','mentions/5','mentions/6','mentions/7','mentions/8','mentions/9','mentions/10','mentions/11','mentions/12','musicMeta/coverMediumUrl','musicMeta/musicAlbum','musicMeta/musicAuthor','musicMeta/musicId','musicMeta/musicName','musicMeta/musicOriginal','musicMeta/playUrl','playCount','searchHashtag/name','searchHashtag/views','searchQuery','shareCount','text','videoMeta/coverUrl','videoMeta/definition','videoMeta/downloadAddr','videoMeta/duration','videoMeta/format','videoMeta/height','videoMeta/originalCoverUrl','videoMeta/originalDownloadAddr','videoMeta/subtitleLinks/0/downloadLink','videoMeta/subtitleLinks/0/language','videoMeta/subtitleLinks/0/tiktokLink','videoMeta/subtitleLinks/1/downloadLink','videoMeta/subtitleLinks/1/language','videoMeta/subtitleLinks/1/tiktokLink','videoMeta/subtitleLinks/2/downloadLink','videoMeta/subtitleLinks/2/language','videoMeta/subtitleLinks/2/tiktokLink','videoMeta/subtitleLinks/3/downloadLink','videoMeta/subtitleLinks/3/language','videoMeta/subtitleLinks/3/tiktokLink','videoMeta/subtitleLinks/4/downloadLink','videoMeta/subtitleLinks/4/language','videoMeta/subtitleLinks/4/tiktokLink','videoMeta/subtitleLinks/5/downloadLink','videoMeta/subtitleLinks/5/language','videoMeta/subtitleLinks/5/tiktokLink','videoMeta/subtitleLinks/6/downloadLink','videoMeta/subtitleLinks/6/language','videoMeta/subtitleLinks/6/tiktokLink','videoMeta/subtitleLinks/7/downloadLink','videoMeta/subtitleLinks/7/language','videoMeta/subtitleLinks/7/tiktokLink','videoMeta/subtitleLinks/8/downloadLink','videoMeta/subtitleLinks/8/language','videoMeta/subtitleLinks/8/tiktokLink','videoMeta/subtitleLinks/9/downloadLink','videoMeta/subtitleLinks/9/language','videoMeta/subtitleLinks/9/tiktokLink','videoMeta/subtitleLinks/10/downloadLink','videoMeta/subtitleLinks/10/language','videoMeta/subtitleLinks/10/tiktokLink','videoMeta/subtitleLinks/11/downloadLink','videoMeta/subtitleLinks/11/language','videoMeta/subtitleLinks/11/tiktokLink','videoMeta/subtitleLinks/12/downloadLink','videoMeta/subtitleLinks/12/language','videoMeta/subtitleLinks/12/tiktokLink','videoMeta/subtitleLinks/13/downloadLink','videoMeta/subtitleLinks/13/language','videoMeta/subtitleLinks/13/tiktokLink','videoMeta/subtitleLinks/14/downloadLink','videoMeta/subtitleLinks/14/language','videoMeta/subtitleLinks/14/tiktokLink','videoMeta/subtitleLinks/15/downloadLink','videoMeta/subtitleLinks/15/language','videoMeta/subtitleLinks/15/tiktokLink','videoMeta/subtitleLinks/16/downloadLink','videoMeta/subtitleLinks/16/language','videoMeta/subtitleLinks/16/tiktokLink','videoMeta/subtitleLinks/17/downloadLink','videoMeta/subtitleLinks/17/language','videoMeta/subtitleLinks/17/tiktokLink','videoMeta/subtitleLinks/18/downloadLink','videoMeta/subtitleLinks/18/language','videoMeta/subtitleLinks/18/tiktokLink','videoMeta/subtitleLinks/19/downloadLink','videoMeta/subtitleLinks/19/language','videoMeta/subtitleLinks/19/tiktokLink','videoMeta/subtitleLinks/20/downloadLink','videoMeta/subtitleLinks/20/language','videoMeta/subtitleLinks/20/tiktokLink','videoMeta/subtitleLinks/21/downloadLink','videoMeta/subtitleLinks/21/language','videoMeta/subtitleLinks/21/tiktokLink','videoMeta/subtitleLinks/22/downloadLink','videoMeta/subtitleLinks/22/language','videoMeta/subtitleLinks/22/tiktokLink','videoMeta/subtitleLinks/23/downloadLink','videoMeta/subtitleLinks/23/language','videoMeta/subtitleLinks/23/tiktokLink','videoMeta/subtitleLinks/24/downloadLink','videoMeta/subtitleLinks/24/language','videoMeta/subtitleLinks/24/tiktokLink','videoMeta/subtitleLinks/25/downloadLink','videoMeta/subtitleLinks/25/language','videoMeta/subtitleLinks/25/tiktokLink','videoMeta/subtitleLinks/26/downloadLink','videoMeta/subtitleLinks/26/language','videoMeta/subtitleLinks/26/tiktokLink','videoMeta/subtitleLinks/27/downloadLink','videoMeta/subtitleLinks/27/language','videoMeta/subtitleLinks/27/tiktokLink','videoMeta/subtitleLinks/28/downloadLink','videoMeta/subtitleLinks/28/language','videoMeta/subtitleLinks/28/tiktokLink','videoMeta/subtitleLinks/29/downloadLink','videoMeta/subtitleLinks/29/language','videoMeta/subtitleLinks/29/tiktokLink','videoMeta/subtitleLinks/30/downloadLink','videoMeta/subtitleLinks/30/language','videoMeta/subtitleLinks/30/tiktokLink','videoMeta/subtitleLinks/31/downloadLink','videoMeta/subtitleLinks/31/language','videoMeta/subtitleLinks/31/tiktokLink','videoMeta/width']

# Cargar el dataset con todas las columnas
raw_dataset = pd.read_csv(dataset_path, dtype='unicode')

# Seleccionar solo las columnas de column_namesShort
dataset = raw_dataset[column_namesShort]

dataset.columns = [col.replace('/', '_') for col in dataset.columns]

# Mostrar las últimas filas del dataset seleccionado
print(dataset.tail())

dataset = dataset.sort_values(by='diggCount', ascending=False)

with open('procesados/data_162col.csv', 'wb') as file:
    dataset.to_csv(file, index=False)