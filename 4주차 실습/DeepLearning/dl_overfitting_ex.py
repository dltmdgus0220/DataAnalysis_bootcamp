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

# 데이터로더 생성
train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(x_va_tensor, y_va_tensor)
# test셋은 그냥 텐서만 넣어도 됨.
# 그러나 테스트셋이 너무 커서 나눠서 예측해야하거나 gpu 메모리 효율성을 높이고 싶거나
# 코드 일관성을 위해 test셋에 대해서도 TensorDataset으로 바꿔주는 경우도 있음.

torch.manual_seed(42) # DataLoader가 셔플할 때나 모델 weigth 초기화 등등에서 사용됨.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# 베이스라인(선형회귀)
lin_reg = LinearRegression()
lin_reg.fit(x_train_scaled, y_train)

y_pred_lr = lin_reg.predict(x_te_scaled)
mse_lr = mean_squared_error(y_te, y_pred_lr) # 대부분 true, pred 순 / 딥러닝 로스계산시에만 pred, true 순
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_te, y_pred_lr)

print('[Linear Regression]')
print(f'Test MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, R^2: {r2_lr:.4f}')
print()


# MLPRegressor
class MLPRegressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 16), # horsepower, weight 2차원 입력
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # mpg 예측
        )

    def forward(self, x):
        return self.net(x)
    
model = MLPRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# R2 스코어, 1 - (∑(y_i​−yhat_​i​)^2 / ∑(y_i​−ybar​)^2), yhat은 예측값, ybar는 실제값전체평균
def r2_score_torch(y_true, y_pred):
    # 리스트로 들어오면 텐서로 변환
    if isinstance(y_true, list):
        y_true = torch.cat(y_true, dim=0)
    if isinstance(y_pred, list):
        y_pred = torch.cat(y_pred, dim=0)

    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot

# 반복학습
num_epochs = 200
patience = 15 # 조기 종료 임계값

best_val_loss = np.inf
best_state = None
no_improve_cnt = 0 # 성능 개선 안된 횟수 저장

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # train
    model.train() # 훈련모드, dropout 켜고 batchnorm을 train 모드
    running_train_loss = 0.0

    # r2 score 계산을 위해 모든 예측과 타겟을 저장하는 리스트
    all_train_preds = []
    all_train_targets = []

    for xb, yb in train_loader:
        optimizer.zero_grad() # 기울기 초기화

        pred = model(xb) # 예측
        loss = criterion(pred, yb) # loss 계산

        loss.backward() # loss 기울기 계산
        optimizer.step() # 기울기로 가중치 업데이트

        running_train_loss += loss.item() # loss 합산

        all_train_preds.append(pred.detach())
        # detach:requires_grad=False로 만들어서 gradient가 추적을 못하게 해주고
        # 이로 인해 쓸데없는 메모리를 사용하거나 역전파 그래프가 커져 느려지는 상황을 예방할 수 있음.
        all_train_targets.append(yb)

    train_mse = running_train_loss/len(train_loader)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score_torch(all_train_targets, all_train_preds)
    train_losses.append(train_mse) # 시각화할 때 사용

    # validate
    model.eval()
    running_val_loss = 0.0

    # r2 score 계산을 위해 모든 예측과 타겟을 저장하는 리스트
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            running_val_loss += loss.item()

            all_val_preds.append(pred)
            all_val_targets.append(yb)

        val_mse = running_val_loss / len(val_loader)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score_torch(all_val_targets, all_val_preds)
        val_losses.append(val_mse) # 시각화할 때 사용
    
    # 베스트 모델 저장
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        best_state = model.state_dict() # 모델 파라미터(즉, 가중치)를 저장
        no_improve_cnt = 0
    else:
        no_improve_cnt += 1

    # 중간 결과 출력
    if (epoch + 1) % 10 == 0: 
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f'Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}')
        print(f'Val MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R^2: {val_r2:.4f}')
        print(f'(no_imporve_cnt: {no_improve_cnt})')

    # 조기 종료
    if no_improve_cnt >= patience:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break


if best_state is not None:
    model.load_state_dict(best_state) # 가장 성능이 좋았던 파라미터로 모델 저장
    print('[Best State]')
    print(best_state)

