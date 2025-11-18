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
cv_auc = cross_val_score(pipe, x_train, y_train, cv=cv, scoring='roc_auc')
cv_acc = cross_val_score(pipe, x_train, y_train, cv=cv, scoring='accuracy')
print(f"auc : {cv_auc}")
print(f"acc : {cv_acc}")
print(f"[cv] ROC_AUC : {cv_auc.mean():.3f}")
print(f"[cv] ACCARACY : {cv_acc.mean():.3f}")

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
y_pred_proba = pipe.predict_proba(x_test)
# print(y_pred_proba)

acc = accuracy_score(y_test, y_pred) # 정확도
prec = precision_score(y_test, y_pred) # 정밀도
rec = recall_score(y_test, y_pred) # 재현율
auc = roc_auc_score(y_test, y_pred_proba[:,[1]]) # roc_auc, auc는 positive class를 얼마나 잘 구분하는지를 측정하는 지표
# roc커브는 x축-특이도, y축-재현율
# 양성을 잘 맞추려다 보면 음성을 틀릴 위험이 커진다 라는 트레이드오프(Trade-off)를 시각화한 곡선
print(f"acc : {acc:.4f}")
print(f"prec : {prec:.4f}")
print(f"rec : {rec:.4f}")
print(f"auc : {auc:.4f}")

print(classification_report(y_test, y_pred, digits=3))