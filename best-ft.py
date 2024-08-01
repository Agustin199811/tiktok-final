import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Cargar el archivo CSV
#file_path = 'procesados/extended_data.csv'
file_path = 'procesados/transformado_digg_playcount.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset y la información básica
print("Primeras filas del dataset:")
print(data.head())

print("\nInformación básica del dataset:")
print(data.info())

# Explorar las columnas y la información básica
print("\nDescripción estadística del dataset:")
print(data.describe())

print("\nNúmero de valores nulos por columna:")
print(data.isnull().sum())

# Convertir variables categóricas a variables numéricas
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

print(data.head())
data.to_csv('procesados/digg_playcount_dummies.csv', index= False)
# Analizar la correlación entre las características y `diggcount`
#print("\nMatriz de correlación con 'diggcount':")
#correlation_matrix = data.corr()
#print(correlation_matrix['diggCount'].sort_values(ascending=False))

# Utilizar métodos de selección de características
X = data.drop('diggCount', axis=1)
y = data['diggCount']

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)

scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
scores = scores.sort_values(by='Score', ascending=False)
scores.to_csv('procesados/best-ft.csv', index=False)
print("\nPuntuación de las características:")
print(scores.sort_values(by='Score', ascending=False))
