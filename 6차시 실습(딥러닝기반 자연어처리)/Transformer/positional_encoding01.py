import torch
import math

def get_positional_encoding(max_len, d_model):
    """
    반환: (max_len, d_model) 텐서
    pos= 0~max_len-1 까지의 위치 벡터
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) ) # (d_model/2,) : 포지션 값에 곱할 스케일링 값들을 미리 계산해 둔 것.
    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원
    return pe  # (max_len, d_model)
