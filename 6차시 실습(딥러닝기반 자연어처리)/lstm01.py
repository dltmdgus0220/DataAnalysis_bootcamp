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
sent1 = [vocab["나는"], vocab["밥을"],  vocab["먹었다"]] # [0, 1, 3]
sent2 = [vocab["나는"], vocab["라면을"], vocab["먹었다"]] # [0, 2, 3]
# 배치 만들기: (batch_size=2, seq_len=3)
batch = torch.tensor([sent1, sent2]) # shape: (2, 3)
print("batch:", batch)
print("batch.shape:", batch.shape)

# 임베딩 레이어
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
# 단어 인덱스 → 임베딩 벡터
emb = embedding(batch) # shape: (2, 3, 4)
print("\n[임베딩 통과 후]")
print("emb.shape:", emb.shape) # (batch, seq_len, embed_dim)

# LSTM 레이어
lstm = nn.LSTM(
    input_size=embed_dim, # 각 타임스텝 입력 벡터 크기
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True # (batch, seq_len, feature) 형태 사용
)

# LSTM 통과
output, (h_n, c_n) = lstm(emb)
print("\n[LSTM 결과]")
print("output.shape:", output.shape) # (batch, seq_len, hidden_size)
print("h_n.shape:", h_n.shape) # (num_layers, batch, hidden_size)
print("c_n.shape:", c_n.shape) # (num_layers, batch, hidden_size)
print("\noutput:", output)
print("\nh_n:", h_n)
print("\nc_n:", c_n)