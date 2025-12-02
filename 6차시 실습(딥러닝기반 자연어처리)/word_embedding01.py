import torch
import torch.nn as nn

vocab = {'나는':0, '밥을':1, '먹었다':2, '학교에':3, '갔다':4}
vocab_size = len(vocab)
emb_dim = 3

embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
print(f'W 가중치 표 : {embed.weight}')

sentence = ['나는', '밥을', '먹었다']
idxs = [vocab[w] for w in sentence]
idx_tensor = torch.tensor(idxs)
print(idx_tensor.shape)
emb_vect = embed(idx_tensor)
print(emb_vect)
