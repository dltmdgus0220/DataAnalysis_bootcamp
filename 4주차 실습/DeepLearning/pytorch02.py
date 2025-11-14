import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pandas as pd

X = torch.randn(100,1) # 정규분포
y = 3*X + 1 + 0.1*torch.randn_like(X) # randn_like:자동으로 X와 같은 shape의 정규분포를 따르는 난수생성(노이즈로써 활용)

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

for xb, yb in loader:
    print(f'batch X: {xb.shape}, batch y: {yb.shape}')

class CSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df[['close','volume']].values, dtype=torch.int64)
        self.y = torch.tensor(df['close'].values, dtype=torch.int64)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
dataset = CSVDataset('E:/데이터분석가 부트캠프/실습/DataAnalysis_bootcamp/4주차 실습/TimeSeries/stock_daily.csv')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (xb, yb) in enumerate(loader, 1):
    print(f'{i}번째 batch X: {xb.shape}, batch y: {yb.shape}')