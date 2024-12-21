import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar conjuntos de datos
datasets = {
    'iris': load_iris(),
    'wine': load_wine(),
    'breast_cancer': load_breast_cancer()
}

# Inicializar el clasificador
nb_classifier = GaussianNB()

# Función de validación
def validate_model(X, y, classifier, validation_type):
    results = {}
    
    if validation_type == 'Stratified Hold-Out':
        # Hold-Out Estratificado (70/30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results['Accuracy'] = accuracy_score(y_test, y_pred)
        results['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
    
    elif validation_type == 'Stratified 10-Fold Cross-Validation':
        # Validación cruzada estratificada (10-Fold)
        skf = StratifiedKFold(n_splits=10)
        accuracies = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
        results['Accuracy'] = np.mean(accuracies)
        
        # Matriz de confusión promedio
        conf_matrices = []
        for train_idx, test_idx in skf.split(X, y):
            classifier.fit(X[train_idx], y[train_idx])
            y_pred = classifier.predict(X[test_idx])
            conf_matrices.append(confusion_matrix(y[test_idx], y_pred))
        
        # Sumar todas las matrices de confusión
        results['Confusion Matrix'] = sum(conf_matrices)
    
    elif validation_type == 'Leave-One-Out':
        # Validación Leave-One-Out
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X, y):
            classifier.fit(X[train_idx], y[train_idx])
            y_pred.append(classifier.predict(X[test_idx])[0])
            y_true.append(y[test_idx][0])
        
        results['Accuracy'] = accuracy_score(y_true, y_pred)
        results['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
    
    return results

# Ejecutar validaciones
for dataset_name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    print(f"\nDataset: {dataset_name}")
    
    for validation_type in ['Stratified Hold-Out', 'Stratified 10-Fold Cross-Validation', 'Leave-One-Out']:
        print(f"\nValidation Type: {validation_type}")
        results = validate_model(X, y, nb_classifier, validation_type)
        print(f"Accuracy: {results['Accuracy']}")
        print(f"Confusion Matrix:\n{results['Confusion Matrix']}")
