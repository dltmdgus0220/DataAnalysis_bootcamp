
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

# 데이터 설명
# mpg : float, 연비 (miles per gallon, 1갤런당 주행거리) → 예측/타깃 변수로 자주 사용
# cylinders : int, 엔진 실린더 수 (예: 4, 6, 8기통) — 엔진 크기/출력 관련 (범주형으로 바꿔서 처리)
# displacement : float, 배기량 (입방인치, cubic inches) — 엔진 크기
# horsepower : float, 마력 (horsepower) — 엔진 출력
# weight : float, 차량 무게 (파운드, pounds)
# acceleration : float, 0→60 mph 가속 시간 (초)
# model_year : int, 차량 모델 연식 (예: 70은 1970년)
# origin : category, 생산 지역: 'usa', 'europe', 'japan'
# name : string, 자동차 이름 (모델명)


# mpg를 gradient_boost 모델로 예측하기

