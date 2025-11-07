
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
 
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


# 데이터 설명
# mpg : float, 연비 (miles per gallon, 1갤런당 주행거리)
# cylinders : int, 엔진 실린더 수 (예: 4, 6, 8기통) — 엔진 크기/출력 관련 (범주형으로 바꿔서 처리)
# displacement : float, 배기량 (입방인치, cubic inches) — 엔진 크기
# horsepower : float, 마력 (horsepower) — 엔진 출력
# weight : float, 차량 무게 (파운드, pounds)
# acceleration : float, 0→60 mph 가속 시간 (초)
# model_year : int, 차량 모델 연식 (예: 70은 1970년)
# origin : category, 생산 지역: 'usa', 'europe', 'japan'
# name : string, 자동차 이름 (모델명)


# origin을 gradient_boost 모델로 예측하기

################
# 1. 데이터 로드 #
################
mpg = sns.load_dataset('mpg') # cylinders를 범주형으로 바꾸기
# print(mpg.info())
# print(mpg.describe())

##########################
# 2. 데이터 전처리(시각화전) #
##########################
# 결측치 확인
# print(mpg.isnull().sum())
# horsepower 결측치 6개 존재
mpg['horsepower'] = mpg['horsepower'].fillna(mpg['horsepower'].median())
# print(mpg.isnull().sum())

# cylinders 컬럼 범주형으로 바꾸기
mpg['cylinders'] = mpg['cylinders'].astype('category')
# print(mpg.info())

# 타켓/특징 분리                 
X = mpg.drop(columns=["origin"])
y = mpg["origin"]

num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

##################
# 3. 데이터 시각화 #
##################
# 수치형 변수들의 히스토그램
mpg[num_cols].hist(bins=20, figsize=(10, 8))
plt.suptitle("[원시] 수치형 변수 히스토그램", y=0.98)
plt.tight_layout()
plt.show()

# 박스 플롯
for col in num_cols:
    sns.boxplot(data=mpg, x="origin", y=col, hue="origin")
    plt.title("[원시] 클래스별 수치형 박스플롯")
    plt.tight_layout()
    plt.show()

# 상관행렬(수치형만)
plt.figure(figsize=(7, 5))
corr = mpg[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("[원시] 수치형 상관행렬")
plt.tight_layout()
plt.show()
# displacement, horsepower, weight 끼리 아주 강한 양의 상관관계를 가짐
