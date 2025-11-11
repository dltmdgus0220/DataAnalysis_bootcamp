import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

df = sns.load_dataset('mpg').dropna().copy()
X = df.drop(columns=['mpg','name'])
y = df['mpg']
# print(df.isnull().sum())

num_cols = X.select_dtypes(include=['float','int']).columns.to_list()
cat_cols = X.select_dtypes(include=['category','object']).columns.to_list()

print(f'num_cols : {num_cols}')
print(f'cat_cols : {cat_cols}')

preprocess = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('prep', preprocess),
    ('reg', LinearRegression())
])

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(x_tr, y_tr)
pred = model.predict(x_te)

r2 = r2_score(y_te, pred)
mae = mean_absolute_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
rmse = np.sqrt(mse)
print(f'R2 : {r2}')
print(f'MAE : {mae}')
print(f'MSE : {mse}')
print(f'RMSE : {rmse}')

resid = y_te - pred

# fig, axes = plt.subplots(1,3, figsize=(14,4))
plt.figure(figsize=(6,4))
plt.scatter(pred, resid, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual vs Predicted')
plt.tight_layout()
plt.show()