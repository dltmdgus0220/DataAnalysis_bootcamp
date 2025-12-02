import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) 간단한 말뭉치
sentences = [
    '이 영화 정말 최고였어요.',
    '배우 연기가 최고입니다.',
    '내용이 지루하고 별로였어요.',
    '스토리가 지루하지만 배우는 좋았어요.'
]

