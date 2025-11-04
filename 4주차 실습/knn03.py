import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

from matplotlib import rc
import platform

if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")

plt.rcParams['axes.unicode_minus']=False


# 1. 데이터 로드
wine = load_wine()
x = wine.data
y = wine.target

print("=== 와인 데이터셋 정보 ===")
print(f"데이터 개수 : {len(x)}")
print(f"특성 : {wine.feature_names}")
print(f"클래스 : {wine.target_names}")
print()
df = pd.DataFrame(x, columns=wine.feature_names)
df["class_name"] = [wine.target_names[i] for i in y]
print(df.head())
print("클래스 별 개수")
print(df["class_name"].value_counts())
print("통계 요약")
print(df.describe())

# 2. 데이터 시각화
# 히스토그램
# for col in wine.feature_names: # 각 특성별 분포 파악
#     plt.figure(figsize=(6,3))
#     sns.kdeplot(data=df, x=col, hue="class_name", fill=True)
#     plt.title(f"{col} 분포 (클래스별)")
#     plt.tight_layout()
#     plt.show()

# # 박스플롯
# for col in wine.feature_names: # 각 특성별 이상치 파악
#     plt.figure(figsize=(12,6))
#     sns.boxplot(data=df, x="class_name", y=col)
#     plt.title(f"{col} 박스플롯")
#     plt.show()

# 3. 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print()
print(f"train : {len(x_train)}개, val : {len(x_val)}개, test : {len(x_test)}개")

# 4. 데이터 스케일링
# 히스토그램으로 분포 확인해본 결과 정규분포를 따르지 않는 특성들 존재 (만약 정규분포 따른다면 standardscaler)
# 박스플롯으로 확인해본 결과 이상치 존재 (이상치가 많이 없다면 minmaxscaler)
# 따라서 robust 선택
scaler = RobustScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)