import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100,1) # 정규분포
y = 3*X + 1 + 0.1*torch.randn_like(X) # randn_like:자동으로 X와 같은 shape의 정규분포를 따르는 난수생성(노이즈로써 활용)

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

for xb, yb in loader:
    print(f'batch X: {xb.shape}, batch y: {yb.shape}')