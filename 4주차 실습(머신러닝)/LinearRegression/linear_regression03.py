import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

rng = np.random.default_rng(42)
x = np.arange(1, 21)
# y = 3 * x + rng.normal(0, 3, size=20)
y = 3 * x + np.cumsum(rng.normal(0,3, size=20))
# cumsum:누적합, 종속적인 잔차를 만들기 위해 임의로 사용함
# x1
# x1+x2
# x1+x2+x3

X = sm.add_constant(x)
# OLS:잔차를 최소화하는 회귀직선을 찾는 모델
model = sm.OLS(y, X) # linearregression 함수와 달리 절편을 만들어주지 않기 때문에 add_constant로 생성해줘야함.
result = model.fit() # 모델 학습(회귀계수 추정)
residual = result.resid
dw = durbin_watson(residual)
print(f"Durbin-Watson 통계량 : {dw}")
# 0-2 미만 : 양의 자기상관 존재->잔차들이 비슷한 방향으로 움직임
# 2 : 자기상관 없음->독립적, 이상적인 상태
# 3-4 : 음의 자기상관 존재->잔차가 번갈아가며 위아래로 흔들림
# 잔차가 독립적이지 않다라는 건 아직 모델이 해석하지 못한 데이터의 패턴이 남아있다는 뜻.
# 즉, vif는 전처리시 사용, dw는 학습 후 검증을 위해 사용