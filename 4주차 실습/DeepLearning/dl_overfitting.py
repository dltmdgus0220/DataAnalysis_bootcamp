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
x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, train_size=0.4, stratify=y_train, random_state=42)

print(x_tr.shape, x_va.shape, x_te.shape)
