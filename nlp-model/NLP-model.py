import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional

# Cargar el archivo CSV
file_path = 'procesados/dataToText.csv'
data = pd.read_csv(file_path)

# Concatenar todos los hashtags en una sola lista
hashtags = data.filter(regex='^hashtags_').fillna('').values.flatten()
hashtags = [hashtag.lower() for hashtag in hashtags if hashtag]

# Limpiar el texto
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-záéíóúñü0-9\s]', '', text)
    else:
        text = ''
    return text

data['text_clean'] = data['text'].apply(clean_text)

# Tokenizar los textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text_clean'])
sequences = tokenizer.texts_to_sequences(data['text_clean'])
word_index = tokenizer.word_index

# Pad sequences
maxlen = 100
data_padded = pad_sequences(sequences, maxlen=maxlen, padding='post')

# Etiquetar palabras clave en los textos
def label_keywords(text, hashtags):
    words = text.split()
    labels = [1 if word in hashtags else 0 for word in words]
    return labels

data['labels'] = data['text_clean'].apply(lambda x: label_keywords(x, hashtags))

# Padding labels to the same length as sequences
labels_padded = pad_sequences(data['labels'], maxlen=maxlen, padding='post')

# Mostrar una muestra de los datos preprocesados
print(data[['text_clean', 'labels']].head())
data.to_csv('procesados/preproText.csv', index=False)

# Definir el modelo
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=maxlen))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(data_padded, labels_padded, epochs=10, batch_size=32, validation_split=0.2)

# Guardar el modelo
model.save('keyword_identification_model.keras')

# Evaluar el modelo
loss, accuracy = model.evaluate(data_padded, labels_padded)
print(f'Loss: {loss}, Accuracy: {accuracy}')
