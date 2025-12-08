import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm


raw_texts = [
    "영화가 정말 재미있고 감동적이었어요",
    "스토리가 지루하고 시간 낭비였어요",
    "배우 연기가 훌륭하고 음악도 좋았어요",
    "내용이 별로고 전개가 너무 느렸어요",
    "정말 최고의 영화였어요 또 보고 싶어요",
    "연출이 엉성하고 집중이 안 됐어요",
]
raw_labels = [1, 0, 1, 0, 1, 0]


def simple_tokenize(text: str):   
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return tokens


tokenized = [simple_tokenize(t) for t in raw_texts]

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

word2idx = {
    PAD_TOKEN:0,
    UNK_TOKEN:1
}

counter = Counter()
for t in tokenized:
    counter.update(t)

for w, _ in counter.most_common():
    word2idx[w] = len(word2idx)
# print(word2idx)

idx2word = {i : w for w, i in word2idx.items()}
# print(idx2word)

vocab_size = len(word2idx)

def encode_tokens(tokens, word2idx, max_len):
    indices = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]

    diff = max((max_len - len(indices)), 0)
    indices += [word2idx[PAD_TOKEN]] * diff
    return indices[:max_len]

MAX_LEN = 10
encoded = [encode_tokens(t, word2idx, MAX_LEN) for t in tokenized]
print(encoded)

x = torch.tensor(encoded, dtype=torch.long)
y = torch.tensor(raw_labels, dtype=torch.float32)

dataset = TensorDataset(x, y)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

class LTSMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layer = 1, pad_idx = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_idx):
        emb = self.embedding(input_idx) # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(emb) # h_n : (n_layers, batch, hidden_size)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden).squeeze(1)
        return logits
    
embed_dim = 50
hidden_size = 64
num_layers = 1
pad_idx = word2idx[PAD_TOKEN]

model = LTSMSentimentClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    num_layer=num_layers,
    pad_idx=pad_idx
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, criterion, optimizer, epoch=1):
    model.train()

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / total
    acc = correct / total
    print(f'Train loss : {avg_loss:.4f} | Train acc : {acc:.4f} ')
