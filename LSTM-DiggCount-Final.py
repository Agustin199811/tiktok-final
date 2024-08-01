import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV
file_path = 'procesados/transformado_digg_playcount.csv'
data = pd.read_csv(file_path)

# Convertir la columna createTimeISO a tipo de dato datetime y ordenar por fecha
data['createTimeISO'] = pd.to_datetime(data['createTimeISO'])
data = data.sort_values(by='createTimeISO')

# Eliminar outliers
q1 = data['diggCount'].quantile(0.25)
q3 = data['diggCount'].quantile(0.75)
iqr = q3 - q1
data = data[(data['diggCount'] >= (q1 - 1.5 * iqr)) & (data['diggCount'] <= (q3 + 1.5 * iqr))]

# Crear nuevas características
data['day_of_week'] = data['createTimeISO'].dt.dayofweek
data['month'] = data['createTimeISO'].dt.month
data['hour'] = data['createTimeISO'].dt.hour

# Verificar si hay datos suficientes para el modelado
if data.shape[0] == 0:
    raise ValueError("No hay suficientes datos después de eliminar NaN. Ajuste la longitud de las ventanas de rezago.")

# Escalar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['createTimeISO']))

# Crear secuencias de entrada y salida para el modelo LSTM
sequence_length = 60
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length][0])  # playCount está en la primera columna
X, y = np.array(X), np.array(y)

# Reshape de X para que sea compatible con LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Dividir los datos en conjuntos de entrenamiento y prueba usando train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.4))
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=200))
model.add(Dropout(0.4))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stop])

model.save('LSTM-diggCount.keras', save_format='keras')

# Realizar predicciones
predictions = model.predict(X_test)

# Asegurarse de que las predicciones no sean negativas
predictions[predictions < 0] = 0

# Invertir la escala de los datos
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

# Verificar el rango de datos originales
print("Rango de datos originales (diggCount):", y_test.min(), y_test.max())

# Verificar el rango de las predicciones
print("Rango de predicciones:", predictions.min(), predictions.max())

# Calcular el error de predicción
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualizar las predicciones frente a los valores reales con la escala de fechas correcta
plt.figure(figsize=(14, 7))
date_range = data['createTimeISO'].iloc[-len(y_test):]
plt.plot(date_range, y_test, label='Real')
plt.plot(date_range, predictions, label='Predicciones', color='red')
plt.title('Predicciones de LSTM vs Valores Reales de diggCount')
plt.xlabel('Fecha')
plt.ylabel('diggCount')
plt.legend()
plt.show()

# Imprimir la propiedad shape de las variables importantes
print(f"data shape: {data.shape}")
print(f"scaled_data shape: {scaled_data.shape}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"predictions shape: {predictions.shape}")