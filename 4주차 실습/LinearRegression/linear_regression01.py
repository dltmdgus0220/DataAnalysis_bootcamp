import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

rng = np.random.default_rng(42)
n = 120

def fit_and_plot_residual(x, y, title, save_prefix=None): # save_prefix는 저장이 필요할 때 사용할 경로
    X = x.reshape(-1,1) # 다행1열로 reshape, x:1차원 X:2차원
    model = LinearRegression().fit(X, y) # 회귀식 생성, linearregression은 무조건 2d 형태여야함.
    y_pred = model.predict(X) # 오차를 계산하기 위해 X로 다시 예측
    residual = y - y_pred

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, alpha=0.7)
    order = np.argsort(x) # 정렬인덱스 저장
    plt.plot(x[order], y_pred[order]) # 선형회귀식에 의한 1차직선
    plt.title(f"y vs x (+Linear Fit) - {title}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_scatter_fit.png')
    plt.show()

    # 잔차 시각화
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residual, alpha=0.7) # 예측값에 따른 잔차를 산점도로 확인
    plt.axhline(0, linestyle='--') # 수평선 긋기, 양수음수를 구분하기 위해 y축 생성, 즉 잔차가 0인 기준선 긋기
    plt.title(f'Residual VS Fitted - {title}')
    plt.xlabel('Predicted (Fitted)')
    plt.ylabel('Residuals (y-y_pred)')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_residual.png')
    plt.show()

    return y_pred, residual, model



x = rng.uniform(-2, 2, size=n) # 하한:-2, 상한:2 이 범위 내 n개의 균등분포를 따르는 난수 생성
yA = 2 + 3 * x + rng.normal(0, 0.8, size=n)
fit_and_plot_residual(x, yA, '선형성 만족 그래프')