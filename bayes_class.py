import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

datasets = {
    'iris': load_iris(),
    'wine': load_wine(),
    'breast_cancer': load_breast_cancer()
}

nb_classifier = GaussianNB()

def validate_model(X, y, classifier, validation_type):
    results = {}
    
    if validation_type == 'Hold-Out Estratificado':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results['Accuracy'] = accuracy_score(y_test, y_pred)
        results['Confusion Matrix'] = confusion_matrix(y_test, y_pred)

    elif validation_type == '10-Fold Cross-Validation Estratificado':
        skf = StratifiedKFold(n_splits=10)
        accuracies = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
        results['Accuracy'] = np.mean(accuracies)
        
        conf_matrices = []
        for train_idx, test_idx in skf.split(X, y):
            classifier.fit(X[train_idx], y[train_idx])
            y_pred = classifier.predict(X[test_idx])
            conf_matrices.append(confusion_matrix(y[test_idx], y_pred))
        results['Confusion Matrix'] = sum(conf_matrices)

    elif validation_type == 'Leave-One-Out':
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X, y):
            classifier.fit(X[train_idx], y[train_idx])
            y_pred.append(classifier.predict(X[test_idx])[0])
            y_true.append(y[test_idx][0])
        results['Accuracy'] = accuracy_score(y_true, y_pred)
        results['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
        
    return results

for dataset_name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    print(f"\nDataset: {dataset_name}")
    
    for validation_type in ['Hold-Out Estratificado', '10-Fold Cross-Validation Estratificado', 'Leave-One-Out']:
        print(f"\nValidation Type: {validation_type}")
        results = validate_model(X, y, nb_classifier, validation_type)
        print(f"Precision: {results['Accuracy']}")
        print(f"Matriz de Confusion:\n{results['Confusion Matrix']}")
