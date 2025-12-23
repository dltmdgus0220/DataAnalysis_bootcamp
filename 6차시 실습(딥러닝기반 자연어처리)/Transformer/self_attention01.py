import torch
import torch.nn.functional as F

torch.manual_seed(42)

def simple_self_attention(x):
    #  x: (seq_len, d_model)    
    d_model = x.size(-1)
    # 가중치 행렬(실무에서는nn.Linear로 학습됨)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(d_model, d_model)
    W_V = torch.randn(d_model, d_model)

    Q = x @ W_Q # (seq_len, d_model)
    K = x @ W_K # (seq_len, d_model)
    V = x @ W_V # (seq_len, d_model)

    # score = Q @ K^T
    scores = Q @ K.transpose(0, 1) # (seq_len, seq_len), 내적=>유사도계산
    scores = scores / (d_model ** 0.5) # 스케일링
    # softmax로 가중치
    attn_weights = F.softmax(scores, dim=-1)
    # 각 행 = 한 단어의 attention 분포

    # 가중합
    out = attn_weights @ V # (seq_len, d_model)
    return out, attn_weights

# 예제: seq_len=4, d_model=8
seq_len, d_model = 4, 8
x = torch.randn(seq_len, d_model)

out, attn_w = simple_self_attention(x)
print("out shape:", out.shape) # (4, 8)
print("attn_w shape:", attn_w.shape) # (4, 4)
print("attention weights:\n", attn_w)