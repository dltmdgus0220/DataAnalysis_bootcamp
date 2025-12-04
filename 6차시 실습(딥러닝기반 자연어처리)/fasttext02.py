import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from gensim.models import FastText

df = pd.read_csv('5차시 실습(통계기반 자연어처리)/movie_reviews.csv', encoding='utf-8').dropna()
df = df.reset_index(drop=True) # 결측치 제거 후 인덱스 재배열
# print(df.shape) # (199992, 3)
# print(df.head())
# print(df.columns) # id, document, label
# print(df.info())
# print(df['label'].value_counts()) # 1(긍정):99996, 0(부정):99996

# 층화추출
_, df_sample = train_test_split(df, test_size=10000, random_state=42, stratify=df['label'])
tokenized_sentences = [doc.split() for doc in df_sample['document']] # 공백 기준 토큰화
labels = df_sample['label'].values # numpy array로

# FastText 모델 학습
ft_model = FastText(
    tokenized_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1, # Skip-gram
    workers=1,
    epochs=100
)

word_index = ft_model.wv.key_to_index # 인덱스 변환용 디렉토리
index_word = ft_model.wv.index_to_key 
vocab_size = len(index_word)
embed_dim = ft_model.vector_size

pretrained_weights = ft_model.wv.vectors # 사전학습된 가중치

PAD_IDX = vocab_size # 패딩 인덱스를 가장 마지막으로
vocab_size_with_pad = vocab_size + 1

extended_weights = np.zeros((vocab_size_with_pad, embed_dim), dtype=np.float32) # 패딩토큰까지 고려하여 사이즈 하나 크게
extended_weights[:vocab_size, :] = pretrained_weights
extended_weights[PAD_IDX, : ] = 0.0 # 0으로 초기화했기 때문에 안해도 됨.


class PaddedTextDataset(Dataset):
    def __init__(self, tokenized_sentences, labels, word_index):
        self.sentences = tokenized_sentences
        self.labels = labels
        self.word_index = word_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.sentences[idx] # 몇번째 문장인지
        y = self.labels[idx]

        indices = [self.word_index[token] for token in tokens]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return idx_tensor, y_tensor


class PaddedSentClassifier(nn.Module):
    def __init__(self, embedding:nn.Embedding, pad_idx:int):
        super().__init__()
        self.embedding = embedding
        self.pad_idx = pad_idx
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_batch):
        # idx_batch = [[3, 5, 0, 0]]

        # emb = tensor([
        # [ [0.1, 0.2, 0.3],   # token 3
        #     [0.4, 0.5, 0.6],   # token 5
        #     [0.7, 0.8, 0.9],   # PAD
        #     [1.0, 1.1, 1.2] ]  # PAD
        # ]) (1, 4, 3) (batch_size, seq_len, emb_dim)

        # mask = tensor([[1., 1., 0., 0.]]) (1,4) 
        # extended_mask = tensor([ [ [1.],[1.],[0.],[0.] ] ]) -> unsqueeze(2)를 통해 (1,4,1)
        # 이후 emb*extend_mask 연산 시 broadcast 규칙에 따라
        # [
        # [ [1., 1., 1.],
        #     [1., 1., 1.],
        #     [0., 0., 0.],
        #     [0., 0., 0.] ]
        # ] 이런 식으로 확장되서 연산 -> 패딩된 원소들 0으로 만들어버림

        emb = self.embedding(idx_batch) 
        mask = (idx_batch != self.pad_idx).float() # broadcast 연산

        length = mask.sum(dim=1, keepdim=True) # 패딩이 아닌 원래 길이 저장
        # [[1,1,1,1], [1,1,0,0]] (2,4) -> [[4], [2]] (2,1)

        extended_mask = mask.unsqueeze(2) # (batch_size, seq_len, 1) 이후 연산을 위해 차원 추가
        masked_emb = emb * extended_mask # (batch, seq_len, embedded_dim) * (batch, seq_len, 1)

        sum_emb = masked_emb.sum(dim=1) # (batch, seq_len, embedded_dim) -> (batch, embedded_dim)

        length = length.clamp(min=1.0) # 0으로 나누는 상황 방지, 1보다 작은 값은 모두 1로
        sent_vec = sum_emb / length # (batch, emb_dim) / (batch, 1) = (batch, emb_dim) , broadcast 연산

        logits = self.fc(sent_vec).squeeze(1) # (batch, 1) -> (batch,)

        return logits
    
