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

vif_before = compute_vif(X.astype(float))
print(vif_before) 
# displacement  19.641683
# weight  10.731681
# cylinders  10.633049
# horsepower   9.398043
# acceleration   2.625581
# model_year   1.244829


def drop_high_corr_features(df_num:pd.DataFrame, thr:float=0.9) -> list:
    cols = df_num.columns.tolist()

    while True:
        C = df_num[cols].corr(numeric_only=True).abs()
        np.fill_diagonal(C.values, 0) # 대각선은 자기자신과의 상관관계를 뜻하므로 0으로 바꾸기
        max_corr = C.values.max()
        if max_corr < thr or len(cols) <= 1: # 임계값보다 높은 상관계수가 없거나 남은 컬럼이 없다면 종료
            break

        i, j = np.where(C.values == max_corr) # (2,3) (3,4) => i=[2,3], j=[3,4] 이렇게 들어감
        i0, j0 = int(i[0]), int(j[0])
        col_i, col_j = cols[i0], cols[j0] # 가장 큰 상관계수를 가지는 피처들을 추출
        mean_i = C.loc[col_i, :].mean()
        mean_j = C.loc[col_j, :].mean()

        drop_col = col_i if mean_i >= mean_j else col_j # 다른 피처들과의 상관계수의 평균이 더 큰 피처를 드랍하기 위함
        cols.remove(drop_col)

    return cols

reduced_cols = drop_high_corr_features(X, thr=0.9)
print(f'Kept features : {reduced_cols}')
vif_after = compute_vif(X[reduced_cols].astype(float))
print(vif_after) 