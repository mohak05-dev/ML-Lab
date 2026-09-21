import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

df.boxplot(figsize=(12, 6))

plt.xticks(rotation=90)
plt.title("Boxplots of Wine Features")
plt.show()
