import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 60, 65, 70, 75, 85, 90],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["StudyHours", "Attendance"]]
y = df["Pass"]

model = LogisticRegression()
model.fit(X, y)

probability = model.predict_proba(X)[:, 1]

fpr, tpr, threshold = roc_curve(y, probability)
auc = roc_auc_score(y, probability)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

print("AUC:", auc)
