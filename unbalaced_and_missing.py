import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

file_path = "Machine Learning\\Machine Learning Helping Files\\crx.data"
columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
data = pd.read_csv(file_path, names=columns, na_values='?')  

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='mean')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

data = pd.get_dummies(data, drop_first=True)

X = data.drop('A16_-', axis=1)
y = data['A16_-']

plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Distrubucion de clases antes de SMOTE')
plt.xlabel('Peticiones de Credito (0 = Aprobadas, 1 = No Aprobadas)')
plt.ylabel('Count')
plt.show()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.to_csv('credit_approval_transformed.csv', index=False)
print("El dataset transformado se guardo como 'credit_approval_transformed.csv'")

plt.figure(figsize=(8, 6))
sns.countplot(x=y_resampled)
plt.title('Distribucion de Clases despues de SMOTE')
plt.xlabel('Peticiones de Credito (0 = Aprobadas, 1 = No Aprobadas)')
plt.ylabel('Count')
plt.show()

labels = ['Aprobadas', 'No Aprobadas']
sizes_before = y.value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(sizes_before, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Peticiones de Credito antes de SMOTE')
plt.axis('equal')
plt.show()

sizes_after = y_resampled.value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(sizes_after, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Peticiones de Credito despues de SMOTE')
plt.axis('equal')
plt.show()
