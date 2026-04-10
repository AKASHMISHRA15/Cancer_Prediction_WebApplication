import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib 

file_path = "c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/lung_cancer_dataset.csv"

df = pd.read_csv(file_path)


df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 2, 'NO': 1})

df['GENDER'] = df['GENDER'].map({'M': 2, 'F': 1})

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier( n_estimators= 500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Lung Cancer (1)", "Lung Cancer (2)"])

print(" Model Accuracy:", round(accuracy * 100, 2), "%")
print("\n Classification Report:\n")
print(report)
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
plt.title("Feature Importance in Lung Cancer Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/lung_feature_importance.png")
print("Feature importance plot saved as 'lung_feature_importance.png'")


model_path = "c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/lung_cancer_model.pkl"
joblib.dump(model, model_path)
print(f"\n💾 Model saved successfully at: {model_path}")