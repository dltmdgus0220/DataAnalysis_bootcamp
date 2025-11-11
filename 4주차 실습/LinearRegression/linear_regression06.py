import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


#=============
# 1. 데이터로드 
#=============
df = sns.load_dataset('mpg').dropna().copy()

X = df.drop(columns=['mpg','name'])
y = df['mpg']

num_cols = X.select_dtypes(include=['int', 'float']).columns.to_list()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()

#===============================================
# 2. 선형성 검증을 위한 시각화 (horsepower, weight)
#===============================================
def fit_and_plot_linear(x, y):
    X = x.reshape(-1,1) # 다행1열로 reshape, x:1차원 X:2차원
    model = LinearRegression().fit(X, y) # 회귀식 생성, linearregression은 무조건 2d 형태여야함.
    y_pred = model.predict(X) # 오차를 계산하기 위해 X로 다시 예측
    resid = y - y_pred

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, alpha=0.7)
    order = np.argsort(x) # 정렬인덱스 저장
    plt.plot(x[order], y_pred[order]) # 선형회귀식에 의한 1차직선
    plt.title("y vs x (+Linear Fit)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()
    return y_pred, resid

def plot_residual(y_pred, resid):
    # 잔차 시각화 : 패턴이 있으면 안됨. 패턴이 있으면 모델이 데이터 구조를 제대로 설명하지 못했다는 신호.
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, resid, alpha=0.7) # 예측값에 따른 잔차를 산점도로 확인
    plt.axhline(0, linestyle='--') # 수평선 긋기, 양수음수를 구분하기 위해 y축 생성, 즉 잔차가 0인 기준선 긋기
    plt.title(f'Residual VS Fitted')
    plt.xlabel('Predicted (Fitted)')
    plt.ylabel('Residuals (y-y_pred)')
    plt.tight_layout()
    plt.show()


y_pred0, resid0 = fit_and_plot_linear(X['horsepower'].to_numpy(), y)
plot_residual(y_pred0, resid0)
y_pred1, resid1 = fit_and_plot_linear(X['weight'].to_numpy(), y)
plot_residual(y_pred1, resid1)

#=============
# 3. 베이스라인
#=============
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop'
)

base_model = Pipeline(steps=[
    ('prep', preprocess),
    ('model', LinearRegression())
])

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

base_model.fit(x_tr, y_tr)
pred0 = base_model.predict(x_te)

print("[베이스모델]")
print(f'R2 : {r2_score(y_te, pred0):.4f}')
print(f'MAE : {mean_absolute_error(y_te, pred0):.4f}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_te, pred0)):.4f}')