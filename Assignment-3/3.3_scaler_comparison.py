
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create Dataset
data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Years_of_Experience": [0, 1, 1, 2, 3]
}

df = pd.DataFrame(data)

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df)

# Apply Scalers
standard = StandardScaler().fit_transform(X)
minmax = MinMaxScaler().fit_transform(X)

print("Original Data:")
print(X)

print("\nStandardScaler Output:")
print(standard)

print("\nMinMaxScaler Output:")
print(minmax)

print("\nStandardScaler Range:")
print(standard.min(), "to", standard.max())

print("\nMinMaxScaler Range:")
print(minmax.min(), "to", minmax.max())
