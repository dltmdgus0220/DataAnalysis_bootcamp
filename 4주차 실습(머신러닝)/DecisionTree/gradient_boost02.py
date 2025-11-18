
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
import pandas as pd

# 1. 데이터 로드
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 2. 데이터 분할
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

base = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1, # 조기종료를 하려면 성능이 개선되었는지를 확인해야하는데 확인할때 사용할 검증셋 비율 설정
    n_iter_no_change=10, # 성능개선이 없었을 때 10번 더 해보고 여전히 성능개선없으면 조기종료
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

base.fit(X_tr, y_tr)
proba_base = base.predict_proba(X_te)[:,1]
pred_base = (proba_base >= 0.5).astype(int) # 임계값 튜닝

print("[Baseline]")
print(f"Accuracy : {accuracy_score(y_te, pred_base)}")
print(f"F1 : {f1_score(y_te, pred_base)}")
print(f"ROC-AUC : {roc_auc_score(y_te, proba_base)}")
print()

num_selector = selector(dtype_include=["int64","float64"])
cat_selector = selector(dtype_include=["object","category"])

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_selector),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_selector)
], remainder='drop')

pipe = Pipeline([
    # ("preprocess", preprocess),
    ('model', HistGradientBoostingClassifier(
        early_stopping=True,
        validation_fraction=0.1, # 0.1이 기본값
        n_iter_no_change=10, # 10이 기본값
        random_state=42
    ))
])

# 그리드 정의
param_gird = [{
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
    #"model__min_samples_leaf": [10, 20, 30]
}]

# 교차검증 설정 (층화+셔플)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 다중 스코어 기록, 최종 refit은 ROC-AUC
scoring = {
    "roc_auc":'roc_auc',
    'f1':'f1',
    'accuracy':'accuracy'
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_gird,
    scoring=scoring,
    refit='roc_auc',
    cv=cv,
    n_jobs=-1,
    return_train_score=False
)

# 학습
search.fit(X_tr, y_tr)

# cv 결과 표 정리
def tidy_cv_results(gscv):
    df = pd.DataFrame(gscv.cv_results_)
    keep = [
        'param_model__learning_rate', 'param_model__max_leaf_nodes', 'param_model__l2_regularization',
        'mean_test_roc_auc', 'std_test_roc_auc',
        'mean_test_f1', 'std_test_f1',
        'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_roc_auc' # GridSearchCV의 cv_results_ 테이블에서 각 하이퍼파라미터 조합의 mean_test_roc_auc가 몇 위인지를 나타내는
    ]
    return df[keep].sort_values('rank_test_roc_auc').reset_index(drop=True)
    # 데이터프레임의 인덱스를 0부터 다시 매기고 기존 인덱스는 버림.
    # reset_index만 하면 인덱스가 새롭게 바뀌지만 기존 인덱스는 새로운 컬럼으로 가지고 있음.

cv_table = tidy_cv_results(search)
print('[Top 10 CV rows by ROC-AUC]')
print(cv_table.head(10).to_string(index=False))
print()

print("[Best Params by ROC-AUC]")
print(search.best_params_)
print()

# 테스트셋 평가
best_model = search.best_estimator_
y_proba = best_model.predict_proba(X_te)[:,1]
y_pred = (y_proba > 0.5).astype(int)

print("[Best Model on Test]")
print(f"Accuracy : {accuracy_score(y_te, y_pred)}")
print(f"F1 : {f1_score(y_te, y_pred)}")
print(f"ROC-AUC : {roc_auc_score(y_te, y_proba)}")
print('\nClassification Report\n', classification_report(y_te, y_pred, digits=3))
print()

# 순열 중요도(MDA, 테스트셋, ROC-AUC 기준)
perm = permutation_importance(
    best_model, X_te, y_te,
    scoring="roc_auc",
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

imp_mean = perm.importances_mean
imp_std = perm.importances_std
rank = np.argsort(-imp_mean)

print('[Top 10 Permutation Importance]')
for idx in rank[:10]:
    print(f"{X.columns[idx]:.25s} :\t mean={imp_mean[idx]:.4f}, std={imp_std[idx]:.4f}")