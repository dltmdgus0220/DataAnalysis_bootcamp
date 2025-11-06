from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance # 변수 중요도를 계산하는 라이브러리
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
penguins = sns.load_dataset('penguins')

# 2. 데이터 전처리
# 수치형/범주형 변수 분리
num_cols = penguins.select_dtypes(include='number') # 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'
cat_cols = penguins.select_dtypes(exclude='number') # 'species', 'island', 'sex'

# 결측치 처리(수치형:중앙값, 범주형:최빈값)
# print(num_cols.isnull().sum())
# print(cat_cols.isnull().sum())
for n in num_cols.columns:
    penguins[n] = penguins[n].fillna(penguins[n].median())
for c in cat_cols.columns:
    penguins[c] = penguins[c].fillna(penguins[c].mode()[0])
# print(penguins.isnull().sum())

# 범주형 원핫인코딩
x = pd.get_dummies(penguins, columns=cat_cols, drop_first=True).drop('species', axis=1)