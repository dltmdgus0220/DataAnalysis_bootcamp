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

