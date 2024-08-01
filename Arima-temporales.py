import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

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

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(data) * 0.8)
train, test = data['diggCount'][:train_size], data['diggCount'][train_size:]

# Diferenciar los datos para hacer la serie estacionaria
train_diff = train.diff().dropna()

# Función para encontrar los mejores hiperparámetros ARIMA
def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.8)
    train, test = X[:train_size], X[train_size:]
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, predictions)
    return mse

# Evaluar diferentes combinaciones de p, d, q
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = [0, 1, 2]
q_values = [0, 1, 2, 4, 6, 8, 10]

best_score, best_cfg = float("inf"), None
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                mse = evaluate_arima_model(train_diff, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print(f'ARIMA{order} MSE={mse:.3f}')
            except:
                continue

print(f'Best ARIMA{best_cfg} MSE={best_score:.3f}')

# Ajustar el mejor modelo ARIMA
model = ARIMA(train, order=best_cfg)
model_fit = model.fit()

# Realizar predicciones
predictions = model_fit.forecast(steps=len(test))

# Calcular el error de predicción
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(14, 7))
plt.plot(data['createTimeISO'][train_size:], test, label='Real')
plt.plot(data['createTimeISO'][train_size:], predictions, label='Predicciones', color='red')
plt.title('Predicciones de ARIMA vs Valores Reales de diggCount')
plt.xlabel('Fecha')
plt.ylabel('diggCount')
plt.legend()
plt.show()
