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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

df = sns.load_dataset('mpg').dropna().copy()
X = df.drop(columns=['mpg','name'])
y = df['mpg']

num_cols = X.select_dtypes(include=['float','int']).columns.to_list()
cat_cols = X.select_dtypes(include=['category','object']).columns.to_list()

print(f'num_cols : {num_cols}')
print(f'cat_cols : {cat_cols}')

# 다중공선성 검사
X_num = df[num_cols].astype(float).copy()
X_num = sm.add_constant(X_num)

vif_values = []

for i in range(1, X_num.shape[1]): # 상수항은 vif 계산할 필요가 없으므로 1번 인덱스부터
    vif_values.append(variance_inflation_factor(X_num.values, i))
vif = pd.Series(vif_values, index=X_num.columns[1:]).sort_values(ascending=False)
print(vif)
# vif 내부 동작 원리
# i번째 열을 타겟 변수로 선택
# 나머지 변수들만 가지고 선형회귀 수행
# 그 회귀의 결정계수 r2 계산
# 1/(1-r2) 로 vif 계산
# 결론 : 나머지 변수들로 해당 타겟 변수를 얼마나 잘 설명할 수 있나를 계산.
# 다중공선성이 크면 어떤 변수가 진짜 영향을 미치는지 구분이 어려워 모델 해석력이 떨어짐(사람의 관점), 모델예측성능에는 큰 문제 없을 수도 있음.
# 해결방법 : 상관높은변수제거, 변수결합, 정규화회귀(Ridge,Lasso) 사용, 주성분분석(PCA)를 통해 상관성 제거, 변수스케일조정으로 상관성완화


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
plt.figure(figsize=(6,4))
plt.scatter(pred, resid, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual vs Predicted')
plt.tight_layout()
plt.show()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x_tr, y_tr, cv=cv, scoring='r2') # 각 5번의 검증 결과 저장
print(f'CV R2 : {cv_scores}')
print(f'CV Mean : {cv_scores.mean()}')
print(f'CV Std : {cv_scores.std()}')

poly_features = ['displacement','horsepower','weight','acceleration']
num_rest = [ c for c in num_cols if c not in poly_features] # 수치형 컬럼 중 poly features를 제외한 컬럼들만 추가

poly_ct = ColumnTransformer(
    transformers=[
        ( 'poly_num', Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2,include_bias=False)),
            ('scaler', StandardScaler())
        ]), poly_features ),
        ('num_rest', StandardScaler(), num_rest),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('prep', poly_ct),
    ('model', LinearRegression())
])

model.fit(x_tr, y_tr)
pred2 = model.predict(x_te)

r2_poly = r2_score(y_te, pred2)
mae_poly = mean_absolute_error(y_te, pred2)
mse_poly = mean_squared_error(y_te, pred2)
rmse_poly = np.sqrt(mse_poly)
print(f'Poly R2 : {r2_poly}')
print(f'Poly MAE : {mae_poly}')
print(f'Poly MSE : {mse_poly}')
print(f'Poly RMSE : {rmse_poly}')

resid2 = y_te - pred2
plt.figure(figsize=(6,4))
plt.scatter(pred2, resid2, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual vs Predicted')
plt.tight_layout()
plt.show()

cv_scores = cross_val_score(model, x_tr, y_tr, cv=cv, scoring='r2') # 각 5번의 검증 결과 저장
print(f'CV R2 : {cv_scores}')
print(f'CV Mean : {cv_scores.mean()}')
print(f'CV Std : {cv_scores.std()}')

