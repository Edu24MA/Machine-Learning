    # Function to calculate Accuracy
def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    # Iterate through all actual and predicted values
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    # Accuracy = (Correct Predictions / Total Predictions)
    return correct / total

# Function to calculate Error Rate
def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)

# Function to calculate the confusion matrix for binary classification
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

# Function to calculate performance metrics from confusion matrix
def calculate_metrics(TP, TN, FP, FN):
    # Precision (Positive Predictive Value)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    
    # Recall (True Positive Rate)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    # True Negative Rate (Specificity)
    true_negative_rate = TN / (TN + FP) if (TN + FP) != 0 else 0
    
    # False Positive Rate
    false_positive_rate = FP / (FP + TN) if (FP + TN) != 0 else 0
    
    # False Negative Rate
    false_negative_rate = FN / (FN + TP) if (FN + TP) != 0 else 0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        "Precision": precision,
        "Recall (TPR)": recall,
        "True Negative Rate (TNR)": true_negative_rate,
        "False Positive Rate (FPR)": false_positive_rate,
        "False Negative Rate (FNR)": false_negative_rate,
        "F1-Score": f1_score
    }

# Example usage:
y_true = [1, 0, 1, 1, 0, 1, 0]  # True labels
y_pred = [1, 0, 0, 1, 0, 1, 1]  # Predicted labels

# Calculating performance metrics
acc = accuracy(y_true, y_pred)
err = error_rate(y_true, y_pred)

# Step 1: Calculate confusion matrix
TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

# Step 2: Calculate performance metrics
metrics = calculate_metrics(TP, TN, FP, FN)

print("Accuracy:", acc)
print("Error Rate:", err)

# Displaying results
print(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
for metric, value in metrics.items():
    print(f"{metric}: {value}")


