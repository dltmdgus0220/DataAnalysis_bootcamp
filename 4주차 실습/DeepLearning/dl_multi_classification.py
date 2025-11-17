import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.datasets import load_iris   # ★ 다중분류용 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. 데이터 불러오기 (iris) + train/test 분할 + 스케일링
# =====================================================================
data = load_iris()
X = data.data          # (n_samples, n_features)
y = data.target        # (n_samples,)  0,1,2 세 클래스

print("특성 개수:", X.shape[1])
print("클래스 라벨:", data.target_names)  # ['setosa' 'versicolor' 'virginica']

# train/test 분할 (stratify=y로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (입력 특성만)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =====================================================================
# 2. 기준 모델: Logistic Regression (다중분류 소프트맥스)
# =====================================================================
# multi_class="multinomial" + 적당한 solver
log_clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
log_clf.fit(X_train_scaled, y_train)

y_pred_log = log_clf.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)
print(f"[Logistic Regression] Test Accuracy: {acc_log:.4f}")

# =====================================================================
# 3. PyTorch용 Tensor / Dataset / DataLoader 준비
# =====================================================================
# numpy -> torch tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)

# CrossEntropyLoss는 라벨을 long 타입 (정수) + shape (N,) 로 기대
y_train_t = torch.tensor(y_train, dtype=torch.long)  # (N,)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)  # (N,)

# Dataset, DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================================================================
# 4. 딥러닝 MLP 분류 모델 정의 (다중분류, 출력 = 클래스 개수)
# =====================================================================
class MLPClassifierMulti(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)   # 출력 차원 = 클래스 개수 (logits)
        )

    def forward(self, x):
        return self.net(x)  # softmax 전의 logits 반환

in_dim = X_train_t.shape[1]
num_classes = len(np.unique(y_train))

model = MLPClassifierMulti(in_dim, num_classes)

# 다중분류용 손실: CrossEntropyLoss (내부에 softmax 포함)
criterion = nn.CrossEntropyLoss()
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

        logits = model(xb)          # (batch, num_classes)
        # ★ CrossEntropyLoss: logits, 정수 라벨(yb) 그대로 전달
        loss = criterion(logits, yb)

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
    logits_test = model(X_test_t)  
  
    y_pred_mlp = logits_test.argmax(dim=1).numpy()  # (N_test,)
  

acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\n[MLP (PyTorch, Multi-class)] Test Accuracy: {acc_mlp:.4f}")
