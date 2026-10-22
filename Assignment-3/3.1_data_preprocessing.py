
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create Dataset
data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Department": ["IT", "HR", "IT", "Sales", "HR"],
    "Years_of_Experience": [0, 1, 1, 2, 3]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

num = ["Age", "Salary", "Years_of_Experience"]
cat = ["Department"]

# Numerical Preprocessing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Preprocessing
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num),
    ("cat", cat_pipe, cat)
])

X = preprocessor.fit_transform(df)

print("\nProcessed Data:")
print(X)
print("Shape:", X.shape)
