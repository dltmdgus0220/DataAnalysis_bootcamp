import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False

# linear_regression() 사용
# horsepower, weight 선형성 검증, 다항식 추가 성능 개선


#=============
# 1. 데이터로드 
#=============
df = sns.load_dataset('mpg').dropna().copy()
num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()

X = df[num_cols].copy()
y = df['mpg'].copy()

