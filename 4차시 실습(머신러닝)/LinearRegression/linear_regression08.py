# ============================================================
# mpg 회귀: (hp, weight만 다항) + GridSearchCV로 차수/규제 튜닝
# ============================================================
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1) 데이터 로드/선택
df = sns.load_dataset("mpg")

poly_cols = ["horsepower", "weight"]   # 다항식 적용 대상
num_all   = ["horsepower", "weight", "acceleration", "displacement", "cylinders", "model_year"]
cat_cols  = ["origin"]                  # 범주형
use_cols  = ["mpg"] + num_all + cat_cols

df = df[use_cols].dropna()

X = df[num_all + cat_cols]
y = df["mpg"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) 공통 전처리 구성
# 2-1) 베이스라인(모든 수치 선형) 전처리
num_linear_transformer = Pipeline([
    ("scaler", StandardScaler())
])
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", num_linear_transformer, num_all),
        ("cat", cat_transformer, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

pipe_linear = Pipeline([
    ("prep", preprocess_linear),
    ("reg", LinearRegression())
])

# 2-2) (hp, weight만 다항) 전처리
poly_transformer = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # degree는 그리드에서 튜닝
    ("scaler", StandardScaler())
])

num_rest = [c for c in num_all if c not in poly_cols]
num_rest_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocess_poly = ColumnTransformer(
    transformers=[
        ("polyNum",  poly_transformer, poly_cols),         # hp, weight만 다항+스케일
        ("numRest",  num_rest_transformer, num_rest),      # 나머지 수치형은 선형+스케일
        ("cat",      cat_transformer, cat_cols),           # 범주 원-핫
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# 3) 베이스라인 학습/성능
pipe_linear.fit(X_tr, y_tr)

y_tr_pred_lin  = pipe_linear.predict(X_tr)
r2_lin_tr   = r2_score(y_tr, y_tr_pred_lin)
mae_lin_tr  = mean_absolute_error(y_tr, y_tr_pred_lin)
rmse_lin_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred_lin))


y_te_pred_lin = pipe_linear.predict(X_te)

r2_lin   = r2_score(y_te, y_te_pred_lin)
mae_lin  = mean_absolute_error(y_te, y_te_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_te, y_te_pred_lin))

print("=== [Baseline: All Linear] ===")
print(f"R²   : test :{r2_lin:.4f}, train:{r2_lin_tr:.4f}")
print(f"MAE  : test :{mae_lin:.4f}, train:{mae_lin_tr:.4f}")
print(f"RMSE : test :{rmse_lin:.4f}, train:{rmse_lin_tr:.4f}")

# 4) (hp, weight만 다항) + 모델/차수/규제 그리드 탐색
pipe_poly = Pipeline([
    ("prep", preprocess_poly),
    ("reg", Ridge())   # estimator는 그리드에서 교체 가능
])

param_grid = [
    # (A) LinearRegression + degree 튜닝
    {
        "prep__polyNum__poly__degree": [1, 2, ],          # 1이면 사실상 다항 미적용과 같음
        "reg": [LinearRegression()]
    },
    # (B) Ridge + degree/alpha 튜닝
    {
        "prep__polyNum__poly__degree": [1, 2, ],
        "reg": [Ridge(max_iter=10000)],
        "reg__alpha":[0.1, 3.0,  5.0, 10.0]
    },
    # (C) Lasso + degree/alpha 튜닝
    {
        "prep__polyNum__poly__degree": [1, 2, ],
        "reg": [Lasso(max_iter=10000)],
        "reg__alpha": [0.001, 0.01, 0.1, 1.0]
    },
]

# RMSE 최소화로 선택 (scoring은 "neg_root_mean_squared_error" → 값이 클수록 좋음)
gs = GridSearchCV(
    estimator=pipe_poly,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    refit=True,    
)
gs.fit(X_tr, y_tr)

best_model = gs.best_estimator_
best_params = gs.best_params_
print("\n=== [GridSearch Best Params] ===")
print(best_params)

y_tr_pred_best = best_model.predict(X_tr)
r2_best_tr   = r2_score(y_tr, y_tr_pred_best)
mae_best_tr  = mean_absolute_error(y_tr, y_tr_pred_best)
rmse_best_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred_best))


# 5) 최적 모델 성능 (Test)
y_te_pred_best = best_model.predict(X_te)
r2_best   = r2_score(y_te, y_te_pred_best)
mae_best  = mean_absolute_error(y_te, y_te_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_te, y_te_pred_best))

print("\n=== [Best (Poly on hp,wt only)] Test ===")
print(f"R²   : test:{r2_best:.4f}, train:{r2_best_tr:.4f}")
print(f"MAE  : test:{mae_best:.4f}, train:{mae_best_tr:.4f}")
print(f"RMSE : test:{rmse_best:.4f}, train:{rmse_best_tr:.4f}")

# 6) 성능 비교표
comp = pd.DataFrame({
    "Model": ["Baseline: All Linear", "Best: Poly on [hp, wt] only"],
    "R2":   [r2_lin, r2_best],
    "MAE":  [mae_lin, mae_best],
    "RMSE": [rmse_lin, rmse_best]
})
print("\n=== 성능 비교 (Test) ===")
print(comp)

lasso = best_model.named_steps["reg"]

coef = lasso.coef_
feature_names = best_model.named_steps['prep'].get_feature_names_out()
print("coef : \n",coef)
print("feature names : \n", feature_names)
print("intercept : ", lasso.intercept_)

coef_map = dict(zip(feature_names, coef))

coef_sorted = sorted(coef_map.items(), key=lambda x: abs(x[1]), reverse=True)

for name, w in coef_sorted:
    print(f"name:{name}, w:{w:.4f}")


