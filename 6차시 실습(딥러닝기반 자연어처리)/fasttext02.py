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

