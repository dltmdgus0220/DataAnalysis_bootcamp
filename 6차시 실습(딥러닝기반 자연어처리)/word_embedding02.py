import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# 1) 간단한 말뭉치
sentences = [
    '이 영화 정말 최고였어요.',
    '배우 연기가 최고입니다.',
    '내용이 지루하고 별로였어요.',
    '스토리가 지루하지만 배우는 좋았어요.'
]

# 2) vocab 만들기
tokens = [s.split() for s in sentences]
# print(tokens)

counter = Counter()
for t in tokens:
    counter.update(t)
# print(counter)

vocab = {'<PAD>': 0, '<UNK>': 1}
for word, _ in counter.most_common():
    vocab[word] = len(vocab)
print(vocab)
