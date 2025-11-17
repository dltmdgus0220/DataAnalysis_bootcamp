import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

x_train, x_te, y_train, y_te = train_test_split(X, y, train_size=0.2, stratify=y, random_state=42)
x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, train_size=0.25, stratify=y_train, random_state=42)

print(x_tr.shape, x_va.shape, x_te.shape)

scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_va_scaled = scaler.transform(x_va)
x_te_scaled = scaler.transform(x_te)

x_tr_tensor = torch.tensor(x_tr_scaled, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

x_va_tensor = torch.tensor(x_va_scaled, dtype=torch.float32)
y_va_tensor = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

x_te_tensor = torch.tensor(x_te_scaled, dtype=torch.float32)
y_te_tensor = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)


train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(x_va_tensor, y_va_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


in_dim = x_tr_tensor.shape[1] # (샘플수, 피쳐수)
model = MLPClassifier(in_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
patience = 15 # 조기 종료를 위해 성능이 15번 개선이 없으면 종료

best_val_loss = np.inf
best_state = None
no_improve_cnt = 0 # 성능 개선 안된 횟수 저장

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

    # 조기 종료
    if epoch_val_loss < best_val_loss:
        # print(f'Update best model at epoch {epoch+1}')
        # print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        best_val_loss = epoch_val_loss
        best_state = model.state_dict() # 모델 파라미터(즉, 가중치)를 저장
        no_improve_cnt = 0
    else:
        no_improve_cnt += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print(f'(no_imporve_cnt: {no_improve_cnt})')

    if no_improve_cnt >= patience:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

if best_state is not None:
    model.load_state_dict(best_state) # 가장 성능이 좋았던 파라미터로 모델 저장


model.eval()
with torch.no_grad():
    logits_test = model(x_te_tensor)
    probs_test = torch.sigmoid(logits_test) # 0-1 사이 확률값으로 변환
    y_pred = (probs_test >= 0.5).int().squeeze(1).numpy() # squeeze(dim):dim의 size가 1이면 없애버림, 1이 아니면 아무변화없음.

test_acc = accuracy_score(y_te, y_pred)
print(f'\n[Test] Accuracy (with Early stopping): {test_acc:.4f}')


plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss')
plt.show()