import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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

for threshold in [0.3, 0.5, 0.7]:
    prediction = (probability >= threshold).astype(int)

    print("\nThreshold:", threshold)
    print("Precision:", precision_score(y, prediction))
    print("Recall:", recall_score(y, prediction))
