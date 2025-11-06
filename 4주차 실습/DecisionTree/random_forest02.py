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
penguins = pd.get_dummies(penguins, columns=cat_cols.columns, drop_first=True)


# 3. 데이터 시각화
# 클래스 불균형 파악
sns.countplot(data=penguins, x='species')
plt.title("종별 분포")
plt.show()
# Adelie와 Chinstrap 차이가 너무 남. 
# Chinstrap 데이터를 증강하거나 가중치를 줘서 해결해야함. (일단 패스)

# 수치형 변수 분포 파악
for n in num_cols.columns:
    sns.histplot(data=penguins, x=n, hue='species', kde=True, fill=True)
    plt.title(f"{n}의 히스토그램")
    plt.show()

# 수치형 변수 이상치 및 분포 확인
for n in num_cols.columns:
    sns.boxplot(data=penguins, x='species', y=n, hue='island') # hue='island'
    # sns.violinplot(data=penguins, x='species', y=n, hue='sex', split=False) 
    plt.title(f"{n}의 박스플롯") # 바이올린플롯
    plt.show()
# 전체적으로 male의 수치형 변수 값들이 더 큼
# Adelie만 모든 island에 살고, island에 따른 수치형 변수의 값들은 비슷함.

# 수치형 변수 간 상관관계
plt.figure(figsize=(7,5))
corr = penguins.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('수치형 변수 상관관계')
plt.tight_layout()
plt.show()