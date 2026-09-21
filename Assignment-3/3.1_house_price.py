import pandas as pd
from sklearn.linear_model import LogisticRegression

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

student = [[5, 75]]
prediction = model.predict(student)

print("Pass" if prediction[0] == 1 else "Fail")
