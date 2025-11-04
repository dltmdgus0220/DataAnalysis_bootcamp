import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
print(x,y)
print(data.feature_names)
print(data.target_names) # malignant : 악성, benign : 정상 혹은 양성종양
print()

# 클래스별 불균형 확인
sns.countplot(data=data.frame, x='target')
plt.show()

# 결측치 확인
print(data.frame.isnull().sum())
# print(data.frame.info())