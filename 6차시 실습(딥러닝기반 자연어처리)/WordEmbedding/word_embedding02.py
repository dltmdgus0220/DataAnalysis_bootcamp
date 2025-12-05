import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# 1. 간단한 말뭉치
sentences = [
    '이 영화 정말 최고였어요.',
    '배우 연기가 최고입니다.',
    '내용이 지루하고 별로였어요.',
    '스토리가 지루하지만 배우는 좋았어요.'
]

# 2. vocab 만들기
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


# 3. 임베딩
vocab_size = len(vocab)
embed_dim = 8

embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)


def sentence_to_indices(sentence_tokens, vocab):
    return [vocab[w] for w in sentence_tokens]

indexed_sentence = [sentence_to_indices(t, vocab) for t in tokens]
print('indexed_sentence :', indexed_sentence)


def get_sentence_embedding(idx_list):
    idx_tensor = torch.tensor(idx_list)
    word_embed = embed(idx_tensor)
    doc_embed = word_embed.mean(dim=0)
    return doc_embed

sentence_embedding = torch.stack([get_sentence_embedding(idx_list) for idx_list in indexed_sentence])
# print(sentence_embedding)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

print('\n문장유사도\n')
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_sim(sentence_embedding[i], sentence_embedding[j])
        print(f'({i}) {sentences[i]} vs ({j}) {sentences[j]} -> 유사도 : {sim:.3f}')