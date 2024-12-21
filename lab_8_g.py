import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Cargar datasets
datasets = [load_iris(), load_wine(), load_breast_cancer()]
results = []

for dataset in datasets:
    X, y = dataset.data, dataset.target
    dataset_name = dataset.DESCR.split("\n")[0]

    # Modelo Naïve Bayes
    model = GaussianNB()

    # Hold-Out (70/30 estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_holdout = accuracy_score(y_test, y_pred)
    conf_matrix_holdout = confusion_matrix(y_test, y_pred)

    # 10-Fold Cross-Validation estratificado
    skf = StratifiedKFold(n_splits=10)
    acc_kfold = cross_val_score(model, X, y, cv=skf).mean()

    # Leave-One-Out
    loo = LeaveOneOut()
    acc_loo = cross_val_score(model, X, y, cv=loo).mean()

    # Almacenar resultados
    results.append({
        "Dataset": dataset_name,
        "Accuracy Hold-Out": acc_holdout,
        "Confusion Matrix Hold-Out": conf_matrix_holdout,
        "Accuracy 10-Fold": acc_kfold,
        "Accuracy LOO": acc_loo
    })

# Mostrar resultados
for result in results:
    print(f"Dataset: {result['Dataset']}")
    print(f"Accuracy Hold-Out: {result['Accuracy Hold-Out']}")
    print(f"Confusion Matrix Hold-Out:\n{result['Confusion Matrix Hold-Out']}")
    print(f"Accuracy 10-Fold: {result['Accuracy 10-Fold']}")
    print(f"Accuracy LOO: {result['Accuracy LOO']}")
    print("-" * 50)