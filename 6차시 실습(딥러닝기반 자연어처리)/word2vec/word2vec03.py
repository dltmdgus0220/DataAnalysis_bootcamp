from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

#  예제용 문장 + 감성 라벨 (1=긍정, 0=부정)
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
    epochs=200
)

word_index = model.wv.key_to_index
index_word = model.wv.index_to_key
vocab_size = len(index_word)
embed_dim = model.vector_size

pretrained_weights = model.wv.vectors

PAD_IDX = vocab_size
vocab_size_with_pad = vocab_size + 1

extended_weights = np.zeros((vocab_size_with_pad, embed_dim), dtype=np.float32)
extended_weights[:vocab_size, :] = pretrained_weights
extended_weights[PAD_IDX, : ] = 0.0 # 0으로 초기화했기 때문에 안해도 되지만 그래도 하기

class PaddedTextDataset(Dataset):
    def __init__(self, tokenized_sentences, labels, word_index):
        super().__init__()
        self.sentence = tokenized_sentences
        self.labels = labels
        self.word_index = word_index

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        tokens = self.sentence[index]
        y = self.labels[index]

        # 문장을 단어 인덱스 벡터로 변환
        idx = [self.word_index[w] for w in tokens]

        return torch.tensor(idx, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
    
def collate_fn_with_pad(batch): # 배치를 만드는 규칙
    seq, labels = zip(*batch)

    padded_seq = pad_sequence(seq, batch_first=True, padding_value=PAD_IDX)

    labels = torch.stack(labels)
    # labels = [torch.tensor(1), torch.tensor(0), torch.tensor(1), torch.tensor(0)]
    # -> tensor([1,0,1,0])
    return padded_seq, labels

class PaddedSentClassifier(nn.Module):
    def __init__(self, embedding:nn.Embedding, pad_idx:int):
        super().__init__()
        self.embedding = embedding
        self.pad_idx = pad_idx
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_batch):

        emb = self.embedding(idx_batch) 
        mask = (idx_batch != self.pad_idx).float() # broadcast 연산

        length = mask.sum(dim=1, keepdim=True)
        # [[1,1,1,1], [1,1,0,0]] (2,4) -> [[4], [2]] (2,1)

        extended_mask = mask.unsqueeze(2)
        masked_emb = emb * extended_mask # (batch, vocab_size, embedded_dim) * (batch, vocab_size, 1)

        sum_emb = masked_emb.sum(dim=1)

        length = length.clamp(min=1.0)
        sent_vec = sum_emb / length

        logits = self.fc(sent_vec).squeeze(1)

        return logits
    

def train_model(model, loader, num_epochs=50, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    for epoch in range(1, num_epochs+1):
        total_loss = 0.0

        for x, y in loader: # 배치 학습
            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(logits, y)
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}] | Loss: {avg_loss:.4f}")


padded_dataset = PaddedTextDataset(sentences, labels, word_index)
padded_loader = DataLoader(
    padded_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn_with_pad
)

padded_embedding = nn.Embedding(vocab_size_with_pad, embed_dim)

with torch.no_grad():
    padded_embedding.weight.copy_(torch.from_numpy(extended_weights))

padded_embedding.weight.requires_grad = True

padded_model = PaddedSentClassifier(padded_embedding, pad_idx=PAD_IDX)
train_model(padded_model, padded_loader, 50, 0.01)