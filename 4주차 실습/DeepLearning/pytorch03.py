import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pandas as pd

X = torch.linspace(0, 1, steps=100).unsqueeze(1) # 0부터 1까지 균등하게 100개 생성, unsqueeze(n):dim=n 위치에 새로운 차원 추가
print(X.shape)

y = 3*X + 1 + 0.1*torch.randn_like(X)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


