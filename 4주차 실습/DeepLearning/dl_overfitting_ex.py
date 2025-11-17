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

