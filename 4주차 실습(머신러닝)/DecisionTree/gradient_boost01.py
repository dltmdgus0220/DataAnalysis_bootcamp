from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

# 데이터 로드
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 홀드아웃 분할(검증 세트 포함: early_stopping용)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 정의
clf = HistGradientBoostingClassifier(
    learning_rate=0.08,  # 작게 → 과적합 완화, 더 많은 트리 필요 
    max_leaf_nodes=31,  # 트리 복잡도(비슷한 역할: max_depth)
    l2_regularization=0.0,
    early_stopping=True,
    random_state=42
)

# 학습
clf.fit(X_tr, y_tr)

# 평가
y_pred = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)[:, 1]
print("Accuracy:", accuracy_score(y_te, y_pred))
print("ROC-AUC :", roc_auc_score(y_te, y_proba))
print()
print(classification_report(y_te, y_pred, digits=3)) 

# 피처 중요도(순열 중요도: 검증/테스트 기반 권장)
perm = permutation_importance(
    clf, X_te, y_te,
    scoring="roc_auc",
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

importances_mean = perm.importances_mean
importances_std  = perm.importances_std
# 중요도 상위 10개 출력
rank = np.argsort(-importances_mean)  # argsort는 오름차순으로 인덱스를 반환, 때문에 -를 붙여서 내림차순정렬
for idx in rank[:10]:
    print(f"{X.columns[idx]:25s}  mean={importances_mean[idx]:.4f}  std={importances_std[idx]:.4f}")
