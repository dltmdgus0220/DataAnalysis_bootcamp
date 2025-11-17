import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pandas as pd
import torch.nn as nn

X = torch.linspace(0, 1, steps=100).unsqueeze(1) # 0부터 1까지 균등하게 100개 생성, unsqueeze(n):dim=n 위치에 새로운 차원 추가
print(X.shape)

y = 3*X + 1 + 0.1*torch.randn_like(X)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), # 1차원이 들어오면 16개로 늘려줌
            nn.ReLU(), # 선형 아웃풋에 비선형성을 추가해줌
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLPRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epoch = 200
for epoch in range(num_epoch):
    model.train() # 훈련모드, 과적합을 방지하기 위해 느슨하게 학습하는 느낌.
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # running_loss += loss.item() * xb.size(0) # loss.item():현재 배치(batch)의 평균 loss, xb.size(0):현재 배치사이즈
        # 이때는 len(train_dataset)으로 나눠줘야함

    if (epoch + 1) % 20 == 0: # 20에폭마다 로스 출력
        print(f"Epoch [{epoch+1}/{num_epoch}]  Loss: {running_loss/len(train_loader):.4f}")