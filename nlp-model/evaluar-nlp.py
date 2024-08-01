from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models

tokenizer = Tokenizer()
maxlen = 100
model = keras.models.load_model('nlp-model/best-nlp.keras')

# Predecir palabras clave en nuevos textos
new_texts = ["Un Restaurante japones con tematicas increibles, la experiencia vale la pena. Omnia Rest, Quito  ¿Cómo es y cuánto cuesta? Inicias con una limpia de chakras y toda tu tarde o noche se convertirá en llenarte de buenas energias. El menú con precios lo pueden encontrar en su pag web y como referencia tomandote un coctel y comiendo algo ligero pagarás entre $35 a $45 por persona, la comida es espectacular y los cocteles ni se diga (recomiendo el coctel de las caras y plato fuerte el pulpo) Las tematicas son una locura que te harán vivir una experiencia asiática literalmente"]
tokenizer.fit_on_texts(new_texts)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data_padded = pad_sequences(new_sequences, maxlen=maxlen, padding='post')
predictions = model.predict(new_data_padded)

# Mostrar resultados
for text, prediction in zip(new_texts, predictions):
    print(f'Texto: {text}')
    print(f'Predicciones: {prediction}')
    
threshold = 0.9
predicted_keywords = [word for word, prob in zip(text.split(), predictions.flatten()) if prob > threshold]
print(predicted_keywords)
