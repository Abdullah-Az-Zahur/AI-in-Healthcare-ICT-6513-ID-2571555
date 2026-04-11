import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

data = pd.read_csv("heart.csv")

print(data.head())

x = data.drop("target", axis=1)
y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)



# ===== Metrics =====
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

importance = model.coef_[0]
features = x.columns

y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ===== Dashboard Layout =====
plt.figure(figsize=(12, 10))

# 1. Confusion Matrix
plt.subplot(2, 2, 1)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])

for i in range(len(cm)):
    for j in range(len(cm)):
        color = "white" if cm[i][j] > cm.max() / 2 else "black"
        plt.text(j, i, cm[i][j], ha="center", color=color)

plt.xlabel("Predicted")
plt.ylabel("Actual")

# 2. Accuracy Bar
plt.subplot(2, 2, 2)
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title(f"Accuracy: {accuracy:.2f}")

# 3. Feature Importance
plt.subplot(2, 2, 3)
plt.barh(features, importance)
plt.title("Feature Importance")

# 🔷 4. ROC Curve
plt.subplot(2, 2, 4)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()
