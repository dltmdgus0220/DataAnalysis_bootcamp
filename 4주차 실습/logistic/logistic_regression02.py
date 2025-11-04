import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# 1. 데이터 로드
data = load_breast_cancer(as_frame=True)
x = data.frame.drop(columns=['target'])
y = data.frame['target']
# print(x,y)
# print(data.feature_names)
# print(data.target_names) # malignant : 악성, benign : 정상 혹은 양성종양

# 클래스별 불균형 확인
sns.countplot(data=data.frame, x='target')
plt.show()

# 결측치 확인
# print(data.frame.isnull().sum())
# print(data.frame.info())

# 2. 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# 3. pipeline 세팅
pipe = Pipeline([
    ("scaler", StandardScaler()), # 스케일러 설정
    ("clf", LogisticRegression(
        class_weight="balanced", # 데이터 개수에 반비례하게 가중치 적용
        max_iter=1000, 
        solver="lbfgs", # 다수 특성에서 안정적, liblinear,saga,newton-cg / sag 등 있음
        n_jobs=None, # 병렬 처리에 쓸 CPU 코어 수. -1이면 모두 사용
        random_state=42
    )) # 분류기 설정
])

# 4. 교차 검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 하이퍼파라미터 그리드 (규제 강도 C)
param_grid = { "clf__C": [0.01, 0.1, 0.3, 1, 3, 10]} # c가 작을수록 규제가 강함.

# GridSearchCV: 훈련셋 안에서만 CV 수행 → 최고 조합 선택 후 refit
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc", 
    cv=cv,
    n_jobs=-1, 
    refit=True, 
    return_train_score=True
)

grid.fit(x_train, y_train)
print("Best params:", grid.best_params_)
print(f"Best CV ROC AUC: {grid.best_score_:.4f}")

cvres = pd.DataFrame(grid.cv_results_).loc[:,["params","mean_test_score","mean_train_score"]].sort_values('mean_test_score', ascending=False)
print(cvres)