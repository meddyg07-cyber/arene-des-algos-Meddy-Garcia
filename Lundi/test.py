from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame

# On regarde les 10 premières mesures pour que la carte reste lisible
cols = data.feature_names[:10]
matrice = df[cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matrice, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Corrélations entre mesures de tumeurs")
plt.tight_layout()
plt.show()
 