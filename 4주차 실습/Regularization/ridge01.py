import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = sns.load_dataset('mpg').dropna().copy()
# print(df.head())
# print(df.shape) # (392,9)

num_cols = df.select_dtypes(include=['int','float']).columns.tolist()
target = 'mpg'
num_cols = [ c for c in num_cols if c != target ]

X = df[num_cols].copy()
y = df[target].copy()

print(f"Numeric features : {num_cols}")

corr = X.corr(numeric_only=True).values
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
plt.xticks(range(len(num_cols)), labels=num_cols, rotation=45)
plt.yticks(range(len(num_cols)), labels=num_cols, rotation=0)
plt.tight_layout()
plt.show()

def compute_vif(df_num:pd.DataFrame) -> pd.DataFrame:
    Z = sm.add_constant(df_num.values, has_constant='skip')
    vifs = []

    for i in range(1, Z.shape[1]):
        vifs.append(variance_inflation_factor(Z, i))
    return pd.DataFrame( { "feature" : df_num.columns, "VIF" : vifs } ).sort_values('VIF', ascending=False)