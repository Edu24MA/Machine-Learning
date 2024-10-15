from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np

# Cargamos la base de datos de Iris (balanceada)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Cargamos la base de datos de calidad de vino (desbalanceada)
wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
X_wine = wine.data
y_wine = wine.target

# Función para verificar si los sets de entrenamiento y prueba son disjuntos
def check_disjoint(train_idx, test_idx):
    intersection = np.intersect1d(train_idx, test_idx)
    if len(intersection) == 0:
        print("Los sets de entrenamiento y prueba son disjuntos.")
    else:
        print(f"Los sets de entrenamiento y prueba NO son disjuntos. Intersección: {intersection}")

# Validación Hold-Out (con r definido por el usuario)
def hold_out_validation(X, y, test_size, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Verificar si son disjuntos
    train_idx = np.array(range(len(X_train)))
    test_idx = np.array(range(len(X_train), len(X)))
    check_disjoint(train_idx, test_idx)
    
    return X_train, X_test, y_train, y_test

# Validación K-Fold (con k definido por el usuario)
def k_fold_validation(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Verificar si los sets de entrenamiento y prueba son disjuntos
        print(f"Fold {fold + 1}:")
        check_disjoint(train_idx, test_idx)
        print(f"  Set de Entrenamiento: {len(train_idx)} ejemplos, Set de Prueba: {len(test_idx)} muestras\n")

# Validación Leave-One-Out
def leave_one_out_validation(X, y):
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Verificar si los sets de entrenamiento y prueba son disjuntos
        check_disjoint(train_idx, test_idx)
        print(f"LOO Fold:")
        print(f"  Set de Entrenamiento: {len(train_idx)} muestras, Set de Prueba: 1 muestra\n")

# Inputs del usuario para la validación Hold-Out y K-Fold
r = float(input("Ingrese el valor de r (tamaño de prueba en Hold-Out, ej: 0.3): "))
k = int(input("Ingrese el valor de K (número de pliegues en K-Fold): "))

# Proceso de validación Hold-Out en el dataset de Iris
X_train_iris, X_test_iris, y_train_iris, y_test_iris = hold_out_validation(X_iris, y_iris, test_size=r)
print("Validación Hold-Out en Iris Dataset:")
print(f"Set de Entrenamiento: {len(X_train_iris)} ejemplos, Set de Prueba: {len(X_test_iris)} muestras\n")

# Proceso de validación Hold-Out en el dataset de calidad de vino
X_train_wine, X_test_wine, y_train_wine, y_test_wine = hold_out_validation(X_wine, y_wine, test_size=r)
print("Validación Hold-Out en Wine Quality Dataset:")
print(f"Set de Entrenamiento: {len(X_train_wine)} ejemplos, Set de Prueba: {len(X_test_wine)} muestras\n")

# Validación K-Fold para el dataset de Iris
print(f"K-Fold Cross-Validation en Iris Dataset (K={k}):")
k_fold_validation(X_iris, y_iris, k=k)

# Validación K-Fold para el dataset de calidad de vino
print(f"K-Fold Cross-Validation en Wine Quality Dataset (K={k}):")
k_fold_validation(X_wine.values, y_wine.values, k=k)

# Validación Leave-One-Out para el dataset de Iris
print("Leave-One-Out Cross-Validation en Iris Dataset:")
leave_one_out_validation(X_iris, y_iris)

# Validación Leave-One-Out para el dataset de calidad de vino (opcional, descomentar para ejecutar)
# print("Leave-One-Out Cross-Validation en Wine Quality Dataset:")
# leave_one_out_validation(X_wine.values, y_wine.values)
