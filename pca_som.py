import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# --- Primera visualización: PCA con el conjunto de datos de vino ---
# Cargar el conjunto de datos de vino
data_wine = load_wine()
X_wine = data_wine.data
y_wine = data_wine.target

# Aplicar PCA
pca = PCA(n_components=2)
X_reduced_wine = pca.fit_transform(X_wine)

# Calcular la varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.sum(varianza_explicada)

# Mostrar el porcentaje de varianza explicada
print(f"Porcentaje de varianza explicada por los 2 componentes: {varianza_acumulada * 100:.2f}%")

# Crear un DataFrame para los resultados
df_wine = pd.DataFrame(data=X_reduced_wine, columns=['Componente 1', 'Componente 2'])
df_wine['Etiqueta'] = y_wine

# Visualizar los resultados del PCA
plt.figure(figsize=(10, 7))
for target in np.unique(y_wine):
    plt.scatter(df_wine[df_wine['Etiqueta'] == target]['Componente 1'], 
                df_wine[df_wine['Etiqueta'] == target]['Componente 2'], 
                label=data_wine.target_names[target])

plt.title('Reducción de Dimensionalidad con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.grid()
plt.show(block=True)  # Bloquea hasta cerrar el gráfico

# --- Segunda visualización: SOM con el conjunto de dígitos ---
# Cargar el conjunto de datos de dígitos (MNIST reducido)
digits_data = load_digits()
scaled_digits_data = StandardScaler().fit_transform(digits_data.data)

# Crear y entrenar el SOM
som = MiniSom(10, 10, scaled_digits_data.shape[1], sigma=0.3, learning_rate=0.5)
som.random_weights_init(scaled_digits_data)
som.train_random(scaled_digits_data, 1000)

# Visualización del mapa SOM
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Distancias de los nodos del SOM
plt.colorbar()

# Superponer los dígitos en el SOM
for i, (x, y) in enumerate(som.win_map(scaled_digits_data).items()):
    plt.text(x[0] + 0.5, x[1] + 0.5, str(digits_data.target[i]), color='red', fontdict={'size': 12, 'weight': 'bold'})

plt.title('Mapa Auto-Organizado (SOM) para el conjunto de dígitos MNIST reducido')
plt.show(block=True)
