from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance # 변수 중요도를 계산하는 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
bc = load_breast_cancer(as_frame=True)
x, y = bc.data, bc.target
feature_names = x.columns

# 2. 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 3. 모델 정의
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth = None,
    min_samples_leaf = 2, # 하나의 리프노드(최종노드)에 최소 2개의 샘플은 있어야함.
    max_features = 'sqrt',
    bootstrap = True,
    oob_score = True,
    n_jobs = -1,
    random_state = 42
)

# 4. 모델 학습
rf.fit(x_train, y_train)

# 5. 모델 성능 출력
pred = rf.predict(x_test)
pred_proba = rf.predict_proba(x_test)[:,1]
print(f"OOB score : {rf.oob_score_:.4f}")
print(f"Test Accuracy : {accuracy_score(y_test,pred):.4f}")
print(f"Test ROCAUC : {roc_auc_score(y_test,pred_proba):.4f}")
print(f"Classification Report : \n {classification_report(y_test, pred, target_names=bc.target_names)}")
