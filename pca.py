import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

# Cargar el conjunto de datos de vino
data = load_wine()
X = data.data
y = data.target

# Aplicar PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Calcular la varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.sum(varianza_explicada)

# Mostrar el porcentaje de varianza explicada
print(f"Porcentaje de varianza explicada por los 2 componentes: {varianza_acumulada * 100:.2f}%")

# Crear un DataFrame para los resultados
df = pd.DataFrame(data=X_reduced, columns=['Componente 1', 'Componente 2'])
df['Etiqueta'] = y

# Visualizar los resultados
plt.figure(figsize=(10, 7))
for target in np.unique(y):
    plt.scatter(df[df['Etiqueta'] == target]['Componente 1'], 
                df[df['Etiqueta'] == target]['Componente 2'], 
                label=data.target_names[target])

plt.title('Reducción de Dimensionalidad con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.grid()
plt.show()
