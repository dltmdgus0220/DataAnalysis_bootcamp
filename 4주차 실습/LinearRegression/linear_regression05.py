import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


np.random.seed(0) # 전역시드고정
x = np.linspace(1,20,50) # 1이상 20이하 균등하게 50개 생성
noise = np.random.chisquare(df=2, size=50) # 카이제곱분포:비대칭적인(오른쪽으로 꼬리가 긴) 확률 분포, 자유도(df)가 커질수록 정규분포처럼 대칭에 가까워짐.
y = 3 * x + noise

X = sm.add_constant(x)
model = sm.OLS(y, X).fit() # OLS:잔차를 최소화하도록 학습하는 회귀분석모델
resid = model.resid

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(resid, kde=True, ax=axes[0])
axes[0].set_title('잔차 분포')
sm.qqplot(resid, line='45', ax=axes[1]) # histplot과 마찬가지로 잔차가 정규분포를 따르는지 확인하기 위한 시각화로, 직선의 형태로 확인하는 그래프.
axes[1].set_title('Q-Q plot')
plt.tight_layout()
plt.show()

shapiro_test = shapiro(resid)
print(f"Shapiro-Wilk 검정 : {shapiro_test.statistic:.3f}, {shapiro_test.pvalue:.3f}") # statistic는 정규분포에 얼마나 가까운지 수치화한것. 0.9보다는 커야 정규분포라고 얘기할 수 있음.
# 귀무가설 : 정규분포다 / 대립가설 : 정규분포가 아니다
# p-value < 0.05 이므로 귀무가설기각/대립가설채택
# 즉, 정규분포를 따르지않는다.