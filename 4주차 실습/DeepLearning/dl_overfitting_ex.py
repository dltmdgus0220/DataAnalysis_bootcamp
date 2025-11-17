import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 로드
df = sns.load_dataset('mpg').dropna()

X = df[['horsepower', 'weight']]
y = df['mpg'].values

# train/val/test 분할 (6:2:2)
x_train, x_te, y_train, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# 스케일링
scaler = StandardScaler()

x_tr_scaled = scaler.fit_transform(x_tr)
x_va_scaled = scaler.transform(x_va)
x_te_scaled = scaler.transform(x_te)
x_train_scaled = scaler.transform(x_train) # 베이스라인에 사용할 데이터

# 텐서 변환
x_tr_tensor = torch.tensor(x_tr_scaled, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

x_va_tensor = torch.tensor(x_va_scaled, dtype=torch.float32)
y_va_tensor = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

x_te_tensor = torch.tensor(x_te_scaled, dtype=torch.float32)
y_te_tensor = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

