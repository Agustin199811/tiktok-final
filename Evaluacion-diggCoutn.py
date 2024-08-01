import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


model = models.load_model('LSTM-diggCount.keras')
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
