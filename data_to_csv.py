import pandas as pd

# Load the data
data = pd.read_csv("Machine Learning\\Machine Learning Helping Files\\heart+disease\\processed.cleveland.data", header=None)

# Add column names
data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Convert 'num' column to binary (0: no disease, 1: has disease)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Save as CSV
data.to_csv("heart_disease.csv", index=False)

# Load the data
data = pd.read_csv("Machine Learning\\Machine Learning Helping Files\\haberman.data", header=None)

# Add column names
data.columns = ['age', 'year', 'nodes', 'survival_status']

# Convert 'survival_status' column to binary (1: survived 5+ years, 0: did not)
data['survival_status'] = data['survival_status'].apply(lambda x: 1 if x == 1 else 0)

# Save as CSV
data.to_csv("haberman_survival.csv", index=False)