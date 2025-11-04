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


wine = load_wine()
x = wine.data
y = wine.target

df = pd.DataFrame(x, columns=wine.feature_names)
df['target'] = y
df['target_name'] = df['target'].map(dict(enumerate(wine.target_names)))

# 데이터 불균형 확인 시각화
# df['target'].value_counts().sort_index().plot(kind='bar')
# plt.xticks([0,1,2], labels=wine.target_names)
# plt.show()
sns.countplot(data=df, x='target_name', )
plt.show()

# 단변량 분포
feat = "alcohol"
# sns.histplot(data=df, x=feat, hue='target_name', bins=20)
sns.kdeplot(data=df, x=feat, hue='target_name', fill=True)
plt.show()

# 분포 및 이상치 확인
sns.boxplot(data=df, x='target_name', y=feat)
plt.show()

# 두 특징 간 관계 및 분포
xfeat, yfeat = "alcohol", "color_intensity"
sns.scatterplot(data=df, x=xfeat, y=yfeat, hue='target_name')
plt.show()

# 모든 특성 간 상관관계
corr = df.drop(columns=["target","target_name"]).corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f") # Reds, Blues, YlGnBu
plt.show()