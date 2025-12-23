import torch
import torch.nn as nn

torch.manual_seed(42)

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True # (batch, seq_len, d_model)
)

x = torch.randn(batch_size, seq_len, d_model)

# Self-Attention이므로 q=k=v=x
attn_out, attn_weights = mha(x, x, x)
print("attn_out shape:", attn_out.shape) # (2, 5, 16)
print("attn_weights shape:", attn_weights.shape) # (2, 4, 5, 5)