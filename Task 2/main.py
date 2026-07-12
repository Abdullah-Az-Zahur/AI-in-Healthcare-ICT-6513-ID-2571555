import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from typing import Any, cast

raw_data = cast(Bunch, load_breast_cancer())
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['target'] = raw_data.target

missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"[LOG] Warning: Found {missing_values} missing values in the dataset.")
else:
    print("[LOG] Data Integrity Check Passed: No missing or null values detected.")

correlations = df.corr()['target'].abs().sort_values(ascending=False)
top_5_features = correlations.index[1:6].tolist()
print(f"[LOG] Selected Top 5 Features: {top_5_features}")

X = df[top_5_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

gb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='f1')
gb_grid.fit(X_train_scaled, y_train)
best_gb_model = gb_grid.best_estimator_

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
svm_grid = GridSearchCV(SVC(kernel='rbf', random_state=42), svm_param_grid, cv=5, scoring='f1')
svm_grid.fit(X_train_scaled, y_train)
best_svm_model = svm_grid.best_estimator_

calibrated_svm_model = CalibratedClassifierCV(best_svm_model, method='sigmoid', cv=5)
calibrated_svm_model.fit(X_train_scaled, y_train)

n_estimators_list = [10, 30, 50, 100, 150, 200]
train_accs = []
val_accs = []

X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

for n in n_estimators_list:
    clf = GradientBoostingClassifier(n_estimators=n, random_state=42)
    clf.fit(X_tr, y_tr)
    train_accs.append(accuracy_score(y_tr, clf.predict(X_tr)))
    val_accs.append(accuracy_score(y_val, clf.predict(X_val)))

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_list, train_accs, marker='o', label='Training Accuracy', color='blue')
plt.plot(n_estimators_list, val_accs, marker='s', label='Validation Accuracy', color='orange')
plt.title('Hyperparameter Impact: n_estimators in Gradient Boosting')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('hyperparameter_impact.png')
plt.show()


models = {
    'Decision Tree': dt_model,
    'Gradient Boosting (Opt)': best_gb_model,
    'SVM RBF (Opt)': calibrated_svm_model
}

metrics = {'Accuracy': [], 'F1-Score': [], 'ROC-AUC': []}


def get_roc_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


for name, model in models.items():
    preds = model.predict(X_test_scaled)
    scores = get_roc_scores(model, X_test_scaled)
    
    metrics['Accuracy'].append(accuracy_score(y_test, preds))
    metrics['F1-Score'].append(f1_score(y_test, preds))
    metrics['ROC-AUC'].append(roc_auc_score(y_test, scores))

x_indices = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x_indices - width, metrics['Accuracy'], width, label='Accuracy', color='#2c3e50')
plt.bar(x_indices, metrics['F1-Score'], width, label='F1-Score', color='#27ae60')
plt.bar(x_indices + width, metrics['ROC-AUC'], width, label='ROC-AUC Score', color='#e74c3c')

plt.xticks(x_indices, list(models.keys()))
plt.ylim(0, 1.1)
plt.title('Model Performance Comparison Matrix')
plt.ylabel('Scores')
plt.legend(loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison_matrix.png')
plt.show()


best_model_name = 'Gradient Boosting (Opt)' if metrics['F1-Score'][1] >= metrics['F1-Score'][2] else 'SVM RBF (Opt)'
best_model = best_gb_model if best_model_name == 'Gradient Boosting (Opt)' else calibrated_svm_model

test_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, test_preds)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(raw_data.target_names), yticklabels=list(raw_data.target_names), ax=ax)
plt.title(f'Confusion Matrix: {best_model_name}')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.savefig('confusion_matrix_best_model.png')
plt.show()

print("\n[LOG] Task 2 implementation complete. All 3 requested plots saved successfully!")