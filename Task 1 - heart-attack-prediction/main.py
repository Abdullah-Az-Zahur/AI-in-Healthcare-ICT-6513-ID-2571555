import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import cast

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import Bunch


def plot_confusion_matrix(ax, matrix, title):
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred No", "Pred Yes"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual No", "Actual Yes"])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            text_color = "white" if matrix[row_index, col_index] > matrix.max() / 2 else "black"
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center", color=text_color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


data_bunch = cast(Bunch, load_breast_cancer())
data = pd.DataFrame(data=data_bunch.data, columns=data_bunch.feature_names)
data["target"] = data_bunch.target
data["target_name"] = pd.Series(data_bunch.target).map({0: "malignant", 1: "benign"})

print("Dataset shape:", data.shape)
print("Missing values per column:\n", data.isnull().sum().sort_values(ascending=False).head())
print("\nFirst rows:\n", data.head())

features = list(data_bunch.feature_names)
x = data[features]
y = pd.Series(data_bunch.target, name="target")

correlations = data[features].corrwith(y).abs().sort_values(ascending=False)
top_5_correlated_features = correlations.head(5)
print("\nTop 5 features correlated with target:\n", top_5_correlated_features)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=42)),
        ]
    ),
}

results = {}
roc_data = {}

for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[model_name] = (fpr, tpr, auc(fpr, tpr))

    results[model_name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

results_frame = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
best_model_name = results_frame.index[0]
best_model_confusion_matrix = results[best_model_name]["confusion_matrix"]

print("\nModel comparison:\n", results_frame[["accuracy", "precision", "recall"]])
print("\nBest model:", best_model_name)

comparison_fig, comparison_ax = plt.subplots(figsize=(8, 5))
results_frame[["accuracy", "precision", "recall"]].plot(kind="bar", ax=comparison_ax)
comparison_ax.set_title("Model Comparison")
comparison_ax.set_ylim(0, 1.05)
comparison_ax.set_ylabel("Score")
comparison_ax.set_xticklabels(results_frame.index, rotation=0)
comparison_ax.legend(loc="lower right")
comparison_fig.tight_layout()
comparison_fig.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")

confusion_fig, confusion_ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(confusion_ax, best_model_confusion_matrix, f"Confusion Matrix: {best_model_name}")
confusion_fig.tight_layout()
confusion_fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")

roc_fig, roc_ax = plt.subplots(figsize=(8, 5))
for model_name, (fpr, tpr, roc_auc) in roc_data.items():
    roc_ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
roc_ax.set_title("ROC Curves")
roc_ax.set_xlabel("False Positive Rate")
roc_ax.set_ylabel("True Positive Rate")
roc_ax.legend()
roc_fig.tight_layout()
roc_fig.savefig(output_dir / "roc_curves.png", dpi=200, bbox_inches="tight")

correlation_fig, correlation_ax = plt.subplots(figsize=(8, 5))
top_5_correlated_features.sort_values().plot(kind="barh", ax=correlation_ax, color="teal")
correlation_ax.set_title("Top 5 Features Correlated with Target")
correlation_ax.set_xlabel("Absolute Correlation")
correlation_fig.tight_layout()
correlation_fig.savefig(output_dir / "top_correlated_features.png", dpi=200, bbox_inches="tight")

dashboard_fig, dashboard_axes = plt.subplots(2, 2, figsize=(14, 10))
results_frame[["accuracy", "precision", "recall"]].plot(kind="bar", ax=dashboard_axes[0, 0])
dashboard_axes[0, 0].set_title("Model Comparison")
dashboard_axes[0, 0].set_ylim(0, 1.05)
dashboard_axes[0, 0].set_ylabel("Score")
dashboard_axes[0, 0].set_xticklabels(results_frame.index, rotation=0)
dashboard_axes[0, 0].legend(loc="lower right")

plot_confusion_matrix(dashboard_axes[0, 1], best_model_confusion_matrix, f"Confusion Matrix: {best_model_name}")

for model_name, (fpr, tpr, roc_auc) in roc_data.items():
    dashboard_axes[1, 0].plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
dashboard_axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
dashboard_axes[1, 0].set_title("ROC Curves")
dashboard_axes[1, 0].set_xlabel("False Positive Rate")
dashboard_axes[1, 0].set_ylabel("True Positive Rate")
dashboard_axes[1, 0].legend()

top_5_correlated_features.sort_values().plot(kind="barh", ax=dashboard_axes[1, 1], color="teal")
dashboard_axes[1, 1].set_title("Top 5 Features Correlated with Target")
dashboard_axes[1, 1].set_xlabel("Absolute Correlation")

dashboard_fig.tight_layout()
dashboard_fig.savefig(output_dir / "dashboard.png", dpi=200, bbox_inches="tight")

plt.show()
