import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)

import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 0. 데이터 컬럼 정보
# survived : 생존여부, int, 0=사망/1=생존
# pclass : 객실등급, int, 1=1등급/2=2등급/3=3등급
# sex : 성별, female/male
# age : 나이, float
# sibsp : 함께 탑승한 형제자매/배우자 수, int
# parch : 함께 탑승한 부모/자녀 수, int
# fare : 요금, float
# embarked : 탑승한 항구, C=Cherbourg/Q=Queenstown/S=Southampton
# class : 객실등급(문자열), First/Second/Third
# who : 성인남성/성인여성/어린이 구분, man/woman/child
# adult_male : 성인남성여부, bool
# deck : 객실이 위치한 갑판 정보, A~G
# embark_town : 탑승항구전체이름, Cherbourg/Queenstown/Southampton
# alive : 생존여부(문자열), yes/no
# alone : 혼자 탑승했는지 여부, bool


# 1. 데이터로드
titanic = pd.DataFrame(sns.load_dataset("titanic"))
df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']].copy() # 깊은 복사

# 2. 원시데이터 시각화
# 클래스별 불균형 확인
sns.countplot(data=df, x='survived')
plt.title('클래스 분포(0=사망, 1=생존)')
plt.show()

# 결측치 처리
df_num = df.select_dtypes(include=['number']).copy()
print(df_num.isnull().sum())

for c in df_num.columns:
    df_num[c] = df_num[c].fillna(df_num[c].median())

# 상관관계 히트맵
plt.figure(figsize=(7,5))
corr = df_num.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('수치형 피쳐 상관관계')
plt.show()
# 해석1. 객실등급이 높을수록 요금도 높아지므로 둘 중 하나만 유지하는게 좋아보임.(다중공선성)
# -> 더 풍부한 정보를 학습시키기 위해 fare 남기기
# 실험적으로 둘 다 포함해서 학습시켰을 때 성능이 더 좋음. 
# 아무래도 다중공선성은 해석의 관점에서 혼동이 있을 수 있기 때문에 빼는 것이라 성능의 관점에서는 강한 상관관계를 가지는 것이 아니기 때문에 포함하는게 좋을 수 있음.

# 박스플롯으로 이상치 탐색
feat = 'age'
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='survived', y=feat, hue='pclass')
plt.title(f'{feat}의 박스플롯')
plt.show()

# 3. 학습
cat_cols = ['sex','embarked','alone']
num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']  

df[num_cols] = df[num_cols].apply(lambda s : s.fillna(s.median())) # s는 각 컬럼을 series로 받음.
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0]) # .mode는 최빈값을 반환, 최빈값은 여러 개일 수 있으니 첫번째 최빈값을 받기 위해 [0], 즉 최빈값으로 결측치 채움

# 범주형 원핫인코딩
x = pd.get_dummies(df[cat_cols+num_cols], columns=cat_cols, drop_first=True)
y = df['survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 수치형 스케일링
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols)
    ],
    remainder='passthrough' # 지금처럼 numerical 데이터에 대해서만 전처리할 때 passthrough 옵션을 주면 그 외 칼럼은 그냥 두기, drop은 삭제
)

# 파이프라인
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model",LogisticRegression(max_iter=1000))
])

param_grid = [
    {
        'model__solver':['lbfgs'],
        'model__penalty':['l2'],
        'model__C': [0.01, 0.1, 1, 3, 10],
        'model__class_weight' : [None, 'balanced']
    },
    
    # {
    #     'model__solver':['saga'],
    #     'model__penalty':['l1'],
    #     'modle__C': [0.01, 0.1, 1, 3, 10],
    #     'model__class_weight' : ['balanced']
    # }
]

# 교차 검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# 학습
grid.fit(x_train, y_train)
best_model = grid.best_estimator_
print("Best Params :", grid.best_params_)
print("CV Best ROC_AUC :", grid.best_score_)

# 4. 성능 평가
y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:,1]

print("accuracy :", accuracy_score(y_test,y_pred))
print("roc_auc :", roc_auc_score(y_test,y_pred_proba))
print("\nClassification Report : \n", classification_report(y_test, y_pred, digits=3))

# roc curve 시각화
RocCurveDisplay.from_estimator(best_model, x_test, y_test)
plt.title("ROC Curve")
plt.show()