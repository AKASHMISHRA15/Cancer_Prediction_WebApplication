import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

file_path = "c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/breast-cancer-dataset.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df.replace('?', np.nan, inplace=True)
df['Diagnosis Result'] = df['Diagnosis Result'].astype(str).str.strip().str.lower()
df['Diagnosis Result'] = df['Diagnosis Result'].map({'malignant': 1, 'benign': 0})
df = df[df['Diagnosis Result'].notna()]

df.drop(columns=[col for col in ['S/N', 'Year'] if col in df.columns], inplace=True)

special_cols = ['Tumor Size (cm)', 'Inv-Nodes', 'Metastasis', 'History']

for col in special_cols:
    df[col] = df[col].replace('#', '99')

df['Tumor Size (cm)'] = pd.to_numeric(df['Tumor Size (cm)'], errors='coerce')

for col in ['Inv-Nodes', 'Metastasis', 'History']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Menopause'] = df['Menopause'].replace({'#': 99}).astype(int)

breast_map = {'#': 99, 'Left': 0, 'Right': 1}
df['Breast'] = df['Breast'].map(breast_map).fillna(99).astype(int)

breast_quadrant_map = {
    '#': 99,
    'Lower inner': 0,
    'Lower outer': 1,
    'Upper inner': 2,
    'Upper outer': 3
}
df['Breast Quadrant'] = df['Breast Quadrant'].map(breast_quadrant_map).fillna(99).astype(int)

df.dropna(inplace=True)

X = df.drop('Diagnosis Result', axis=1)
y = df['Diagnosis Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n Top Features by Importance:")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Breast Cancer Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/breast_feature_importance.png")
print(" Feature importance plot saved as 'breast_feature_importance.png'")


joblib.dump(scaler, 'c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/breast-scaler.pkl')
joblib.dump(model, 'c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/breast-cancer-model.pkl')
print("Model and scaler saved.")
