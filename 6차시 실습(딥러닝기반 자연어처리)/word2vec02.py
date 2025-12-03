from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1) 예제용 문장 + 감성 라벨 (1=긍정, 0=부정)
sentences = [
    ["이", "영화", "정말", "최고", "였다"],      # 1
    ["배우", "연기", "가", "너무", "좋다"],      # 1
    ["스토리", "가", "지루하고", "별로", "였다"], # 0
    ["내용", "이", "지루하다"],                 # 0
    ["음악", "이", "감동적이고", "최고", "였다"], # 1
    ["연출", "이", "허술하고", "지루하다"],      # 0
]
labels = [1,1,0,0,1,0]

