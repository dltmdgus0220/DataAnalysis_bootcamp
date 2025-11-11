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
