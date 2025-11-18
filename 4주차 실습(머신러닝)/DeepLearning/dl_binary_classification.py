import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. 데이터 불러오기 (breast_cancer) + train/test 분할 + 스케일링
# =====================================================================
data = load_breast_cancer()
X = data.data          # (n_samples, n_features)
y = data.target        # (n_samples,)  0 = 양성, 1 = 악성

print("특성 개수:", X.shape[1])
print("클래스 라벨:", data.target_names)  # ['malignant' 'benign'] 등

# train/test 분할 (stratify=y로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (입력 특성만)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =====================================================================
# 2. 기준 모델: Logistic Regression (선형 분류모델)
# =====================================================================
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_scaled, y_train)

y_pred_log = log_clf.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)
print(f"[Logistic Regression] Test Accuracy: {acc_log:.4f}")

# =====================================================================
# 3. PyTorch용 Tensor / Dataset / DataLoader 준비
# =====================================================================
# numpy -> torch tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Dataset, DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================================================================
# 4. 딥러닝 MLP 분류 모델 정의 (이진분류, 출력 1개 + BCEWithLogitsLoss)
# =====================================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)   # 출력 1 → sigmoid로 0~1 확률로 해석
        )

    def forward(self, x):
        return self.net(x)  # 로짓(logit) 출력 (sigmoid 전)

in_dim = X_train_t.shape[1]
model = MLPClassifier(in_dim)

criterion = nn.BCEWithLogitsLoss()                # 이진분류용 손실, 다중은 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =====================================================================
# 5. 학습 루프
# =====================================================================
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model(xb)               # (batch, 1)
        loss = criterion(logits, yb)     # BCEWithLogitsLoss: sigmoid 내부 포함

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_dataset)

    if (epoch + 1) % 10 == 0:
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

# =====================================================================
# 6. 테스트셋에서 MLP 성능 평가 (정확도)
# =====================================================================
model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)              # (N_test, 1)이때 구해진 값은 logit값으로 각 클래스에 대한 점수일뿐 아직 확률값이 아니다.
    # print(logits_test)
    probs_test = torch.sigmoid(logits_test)    # 0~1 확률
    # print(probs_test)
    
    y_pred_mlp = (probs_test >= 0.5).int().squeeze(1).numpy()  # 0/1 예측

acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\n[MLP (PyTorch)] Test Accuracy: {acc_mlp:.4f}")

