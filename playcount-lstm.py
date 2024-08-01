import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Cargar el archivo CSV
file_path = 'procesados/playC_transformado_digg_playcount.csv'  # Actualiza con la ruta correcta
data = pd.read_csv(file_path)

# Convertir la columna createTimeISO a tipo de dato datetime y ordenar por fecha
data['createTimeISO'] = pd.to_datetime(data['createTimeISO'])
data = data.sort_values(by='createTimeISO')

# Comprobar valores faltantes en la columna playCount
missing_values = data['playCount'].isnull().sum()
print(f"Valores faltantes en playCount: {missing_values}")

# Visualizar la serie temporal de playCount
plt.figure(figsize=(14, 7))
plt.plot(data['createTimeISO'], data['playCount'])
plt.title('Serie temporal de playCount')
plt.xlabel('Fecha')
plt.ylabel('playCount')
plt.show()

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data['playCount'].values.reshape(-1, 1))
scaled_data = scaler.transform(data['playCount'].values.reshape(-1, 1))

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
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]



# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=1, batch_size=32)

model.save('playcount-serietemporal.keras', save_format='keras')

# Realizar predicciones
predictions = model.predict(X_test)

print(X_train + X_train.shape)
print(y_train + y_train.shape)

# Verificar el rango de datos originales y escalados
print("Rango de datos originales:", data['playCount'].min(), data['playCount'].max())
print("Rango de datos escalados:", scaled_data.min(), scaled_data.max())


# Verificar el rango de las predicciones escaladas
print("Rango de predicciones escaladas:", predictions.min(), predictions.max())

# Desnormalizar las predicciones
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Verificar el rango de las predicciones desnormalizadas
print("Rango de predicciones desnormalizadas:", predictions.min(), predictions.max())

# Calcular el error de predicción
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(14, 7))
plt.plot(data['createTimeISO'][train_size + sequence_length:], y_test, label='Real')
plt.plot(data['createTimeISO'][train_size + sequence_length:], predictions, label='Predicciones', color='red')
plt.title('Predicciones de LSTM vs Valores Reales de playCount')
plt.xlabel('Fecha')
plt.ylabel('playCount')
plt.legend()
plt.show()
