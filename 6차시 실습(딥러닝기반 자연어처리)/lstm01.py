import torch
import torch.nn as nn

vocab_size = 4 # 단어 4개
embed_dim = 4 # 단어를 4차원 벡터로
hidden_size = 3 # LSTM 은닉벡터 크기 = 3

vocab = {
    "나는": 0,
    "밥을": 1,
    "라면을": 2,
    "먹었다": 3
}
