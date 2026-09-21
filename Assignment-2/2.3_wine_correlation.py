import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

correlation = df.corr()

sns.heatmap(correlation, annot=True)

plt.title("Wine Feature Correlation")
plt.show()

correlation = correlation.where(
    ~pd.DataFrame(
        [[i == j for j in range(len(correlation.columns))]
         for i in range(len(correlation.columns))],
        index=correlation.index,
        columns=correlation.columns
    )
)

print("Strongest Positive Correlation:")
print(correlation.stack().idxmax())
print("Correlation Value:", correlation.stack().max())
