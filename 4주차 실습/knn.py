import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# 0. 한글 설정
from matplotlib import font_manager, rc
import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 특성 (꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비), ndarray
y = iris.target  # 타겟 (0: Setosa, 1: Versicolor, 2: Virginica), ndarray
print(X[:5], y[:5])
# X, y = load_iris(return_X_y=True, as_frame=True), as_frame을 true로 하면 각각 dataframe, series로 받음.

print("=== 붓꽃 데이터셋 정보 ===")
print(f"데이터 개수: {len(X)}")
print(f"특성(features): {iris.feature_names}")
print(f"클래스(species): {iris.target_names}")
print()

df_sepal = pd.DataFrame(X[:, [0,1]], columns=iris.feature_names[:2]) # sepal length,width만 보기
df_sepal['species'] = [iris.target_names[i] for i in y]
# print(df_sepal)

df_petal = pd.DataFrame(X[:, [2,3]], columns=iris.feature_names[2:]) # petal length,width만 보기
df_petal['species'] = [iris.target_names[i] for i in y]
# print(df_petal)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

sns.scatterplot(data=df_sepal, x=iris.feature_names[0], y=iris.feature_names[1], hue='species', ax=ax1)
ax1.set_xlabel('꽃받침 길이(cm)')
ax1.set_ylabel('꽃받침 너비(cm)')
ax1.set_title('붓꽃 데이터 분포')
ax1.grid()

sns.scatterplot(data=df_petal, x=iris.feature_names[2], y=iris.feature_names[3], hue='species', ax=ax2)
ax2.set_xlabel('꽃잎 길이(cm)')
ax2.set_ylabel('꽃잎 너비(cm)')
ax2.set_title('붓꽃 데이터 분포')
ax2.grid()
plt.tight_layout()
plt.show()