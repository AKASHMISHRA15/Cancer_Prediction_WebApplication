import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = "c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/lung_cancer_dataset.csv"
df = pd.read_csv(file_path)

df['GENDER'] = df['GENDER'].map({'F': 1, 'M': 2})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 1, 'YES': 2})

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")
print("Classification Report:\n", report)
