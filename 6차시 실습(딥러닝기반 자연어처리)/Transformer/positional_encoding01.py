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
    # sin,cos 주기함수이므로 일정한 패턴이 반복되고 변화가 부드러움.
    # 따라서, 두 위치 간의 거리(pos difference)를 attention이 쉽게 계산할 수 있음.
    # 또한, 주파수를 여러 개 사용하기 때문에 고주파로 가까운 관계를 표현하고 저주파로 먼 관계를 표현하는 것이 가능해
    # 문장 내 장거리 의존성과 단거리 의존성을 동시에 표현할 수 있음.
    # sin, cos 범위가 무한대이기 때문에 학습 중 보지 못한 긴 문장도 일반화 가능.
    # sinusodial 방식은 수학적 공식에 대입해 계산하면 되기 때문에 학습이 필요없다는 장점도 있음.