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
    

class SimpleSentClassifier(nn.Module):
    def __init__(self, embedding:nn.Embedding): # embedding:사전학습된 모델
        super().__init__()
        self.embedding = embedding
        self.fc = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, idx_tensor):
        if idx_tensor.dim() == 1:
            idx_tensor = idx_tensor.unsqueeze(0) # (1, embedding_dim) 즉 (1,1) shape으로 만들어줌, [2]->[[2]]

        emb = self.embedding(idx_tensor) # idx_tensor:(batch_size=문장수, 문장길이=단어수) -> emb:(batch_size, 문장길이, embedding_dim)
        sent_vec = emb.mean(dim=1) # 단어별 인덱스 번호별로 평균구함
        # [[[1,2,3,4],[2,2,2,2],[1,3,1,3]]] (1,3,4) -> [[[4/3,7/3,2,3]]] (1,1,4)
        
        logit = self.fc(sent_vec).squeeze(1) # (batch_size, embbeding_dim)

        return logit


def build_embedding_from_w2v(pretrained_weight, freeze:bool)->nn.Embedding:
    vocab_size, embed_dim = pretrained_weight.shape

    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    with torch.no_grad():
        embedding.weight.copy_(torch.from_numpy(pretrained_weight))
        # 자연어처리에서는 vocab_size가 크기 때문에 메모리를 고려하여 from_numpy를 사용
        # embedding.weight = torch.tensor(pretrained_weight, dtype=float32) 이런 식으로 할당하는 것과 동작은 같음.
        # 그러나 torch.tensor로 씌우는 과정에서 새 메모리 복사가 일어나기 때문에 메모리나 속도 측면에서 from_numpy를 사용.

    embedding.weight.requires_grad = not freeze

    return embedding


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


dataset = SimpleTestDataset(sentences, labels, word_index)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

    
print("\n==================")
print("1. 임베딩 고정 freeze 버전")
print("==================")

frozen_embedding = build_embedding_from_w2v(pretrained_weights, freeze=True)
model_frozen = SimpleSentClassifier(frozen_embedding)
print("임베딩 requires_grad :", model_frozen.embedding.weight.requires_grad)
print("FC Layer requires_grad :", model_frozen.fc.weight.requires_grad)

train_model(model_frozen, loader, num_epochs=50, lr=0.01)


