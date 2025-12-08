import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm


raw_texts = [
    "영화가 정말 재미있고 감동적이었어요",
    "스토리가 지루하고 시간 낭비였어요",
    "배우 연기가 훌륭하고 음악도 좋았어요",
    "내용이 별로고 전개가 너무 느렸어요",
    "정말 최고의 영화였어요 또 보고 싶어요",
    "연출이 엉성하고 집중이 안 됐어요",
]
raw_labels = [1, 0, 1, 0, 1, 0]

