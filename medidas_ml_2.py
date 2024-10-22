import pandas as pd

def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    return correct / total

def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, TN, FP, FN

def medidas_d(TP, TN, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    true_negative_rate = TN / (TN + FP) if (TN + FP) != 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) != 0 else 0
    false_negative_rate = FN / (FN + TP) if (FN + TP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        "Precision": precision,
        "Recall (TPR)": recall,
        "True Negative Rate (TNR)": true_negative_rate,
        "False Positive Rate (FPR)": false_positive_rate,
        "False Negative Rate (FNR)": false_negative_rate,
        "F1-Score": f1_score
    }

# Function to process a dataset and calculate metrics
def dataset_p(y_true, y_pred, dataset_name):
    print(f"\n--- Dataset {dataset_name} ---")
    
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    
    metrics = medidas_d(TP, TN, FP, FN)
    
    acc = accuracy(y_true, y_pred)
    err = error_rate(y_true, y_pred)
    
    print(f"Accuracy: {acc}")
    print(f"Error Rate: {err}")
    print(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

heart_disease_data = pd.read_csv('heart_disease.csv')

X_heart_disease = heart_disease_data.drop(columns=['num'])
y_heart_disease = heart_disease_data['num']

y_pred_heart_disease = [y_heart_disease.mode()[0]] * len(y_heart_disease)

haberman_data = pd.read_csv('haberman_survival.csv')

X_haberman = haberman_data.drop(columns=['survival_status'])
y_haberman = haberman_data['survival_status']

y_pred_haberman = [y_haberman.mode()[0]] * len(y_haberman)

dataset_p(y_heart_disease, y_pred_heart_disease, "Heart Disease")

dataset_p(y_haberman, y_pred_haberman, "Haberman's Survival")
