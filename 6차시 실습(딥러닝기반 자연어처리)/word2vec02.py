from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 예제용 문장 + 감성 라벨 (1=긍정, 0=부정)
sentences = [
    ["이", "영화", "정말", "최고", "였다"],      # 1
    ["배우", "연기", "가", "너무", "좋다"],      # 1
    ["스토리", "가", "지루하고", "별로", "였다"], # 0
    ["내용", "이", "지루하다"],                 # 0
    ["음악", "이", "감동적이고", "최고", "였다"], # 1
    ["연출", "이", "허술하고", "지루하다"],      # 0
]
labels = [1,1,0,0,1,0]


model = Word2Vec(
    sentences=sentences,
    vector_size=50, # 임베딩 차원
    window=3, # 문맥 범위
    min_count=1, # 최소 등장 횟수
    sg=1, # 1=Skip-gram, 0=CBOW
    workers=1, # 실습 환경에서는 1로 해도 충분
    epochs=200 # 에폭수를 늘려서 더 수렴시키기
)

word_index = model.wv.key_to_index
index_word = model.wv.index_to_key
vocab_size = len(index_word)
embed_dim = model.vector_size

pretrained_weights = model.wv.vectors

class SimpleTestDataset(Dataset):
    def __init__(self, sentences, labels, word_index):
        self.sentences = sentences # 문장 리스트
        self.labels = labels # 라벨 리스트
        self.word_index = word_index # 

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        tokens = self.sentences[index]
        y = self.labels[index]

        # 문장을 단어 인덱스 벡터로 변환
        idx = [self.word_index[w] for w in tokens]

        return torch.tensor(idx, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
    
