import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = sns.load_dataset('mpg').dropna().copy()

X = df[['horsepower', 'weight']]
y = df['mpg']

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr) # 학습하고 변환까지, 학습한 결과도 저장
x_te_scaled = scaler.transform(x_te)

lin_reg = LinearRegression()
lin_reg.fit(x_tr_scaled, y_tr)

y_pred_lr = lin_reg.predict(x_te_scaled)
rmse_lr = np.sqrt(mean_squared_error(y_te, y_pred_lr))

print(f'[Linear Regression] Test RMSE: {rmse_lr:.4f}')

print(x_tr_scaled.shape, y_tr.shape) # (313, 2) (313,)
print(x_te_scaled.shape, y_te.shape) # (79, 2) (79,)

x_tr_tensor = torch.tensor(x_tr_scaled, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1) # (313,1)

x_te_tensor = torch.tensor(x_te_scaled, dtype=torch.float32)
y_te_tensor = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1) # (79,1)

train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
test_dataset = TensorDataset(x_te_tensor, y_te_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

