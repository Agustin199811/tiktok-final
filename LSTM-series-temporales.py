import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Cargar el archivo CSV
file_path = 'procesados/transformado_digg_playcount.csv'  # Actualiza con la ruta correcta
data = pd.read_csv(file_path)

# Convertir la columna createTimeISO a tipo de dato datetime y ordenar por fecha
data['createTimeISO'] = pd.to_datetime(data['createTimeISO'])
data = data.sort_values(by='createTimeISO')

# Comprobar valores faltantes en la columna diggCount
missing_values = data['diggCount'].isnull().sum()
print(f"Valores faltantes en diggCount: {missing_values}")

# Visualizar la serie temporal de diggCount
plt.figure(figsize=(14, 7))
plt.plot(data['createTimeISO'], data['diggCount'])
plt.title('Serie temporal de diggCount')
plt.xlabel('Fecha')
plt.ylabel('diggCount')
plt.show()

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['diggCount'].values.reshape(-1, 1))

# Crear secuencias de entrada y salida para el modelo LSTM
sequence_length = 60
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length])
X, y = np.array(X), np.array(y)

# Reshape de X para que sea compatible con LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32)

model.save('serietemporal.keras', save_format='keras')

# Realizar predicciones
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular el error de predicción
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(14, 7))
plt.plot(data['createTimeISO'][train_size + sequence_length:], y_test, label='Real')
plt.plot(data['createTimeISO'][train_size + sequence_length:], predictions, label='Predicciones', color='red')
plt.title('Predicciones de LSTM vs Valores Reales de diggCount')
plt.xlabel('Fecha')
plt.ylabel('diggCount')
plt.legend()
plt.show()
