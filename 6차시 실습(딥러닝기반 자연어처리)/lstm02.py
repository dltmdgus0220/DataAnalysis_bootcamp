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


def simple_tokenize(text: str):   
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return tokens


tokenized = [simple_tokenize(t) for t in raw_texts]

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

word2idx = {
    PAD_TOKEN:0,
    UNK_TOKEN:1
}

counter = Counter()
for t in tokenized:
    counter.update(t)

for w, _ in counter.most_common():
    word2idx[w] = len(word2idx)
# print(word2idx)

idx2word = {i : w for w, i in word2idx.items()}
# print(idx2word)

vocab_size = len(word2idx)

def encode_tokens(tokens, word2idx, max_len):
    indices = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]

    diff = max((max_len - len(indices)), 0)
    indices += [word2idx[PAD_TOKEN]] * diff
    return indices[:max_len]
