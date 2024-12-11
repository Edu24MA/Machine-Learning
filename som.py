import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from minisom import MiniSom

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
    plt.text(x[0]+0.5, x[1]+0.5, str(digits_data.target[i]), color='red', fontdict={'size': 12, 'weight': 'bold'})

plt.title('Mapa Auto-Organizado (SOM) para el conjunto de dígitos MNIST reducido')
plt.show()
