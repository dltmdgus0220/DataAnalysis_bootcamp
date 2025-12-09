import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

chars = list('abcdefghijklmnopqrstuvwxyz')
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars
stoi = {ch:i for i, ch in enumerate(itos)} # vocab 역할
# print(itos)
# print(stoi)

PAD_IDX = stoi[PAD_TOKEN]
SOS_IDX = stoi[SOS_TOKEN]
EOS_IDX = stoi[EOS_TOKEN]

vocab_size = len(stoi)

def random_string(min_len=3, max_len=7): #random.randint():문자열 길이 결정, random.choice():복원추출
    length = random.randint(min_len, max_len)
    return ''.join([random.choice(chars) for _ in range(length)])

def encode_sequence(text:str): # abcd(입력) -> dcba(예측)
    # 'abcd' -> '3 4 5 6' -> <sos><eos> 붙여서 '1 3 4 5 6 2' 이런 형태로
    encode_text = [stoi[SOS_TOKEN]] + [stoi[t] for t in text] + [stoi[EOS_TOKEN]]
    return torch.tensor(encode_text, dtype=torch.long)

def decode_sequence(indices): # '1 3 4 5 6 2' -> <sos><eos> 떼고 다시 'abcd'로
    result = []
    for idx in indices:
        ch = itos[idx]
        if ch in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            continue
        result.append(ch)
    return ''.join(result)

# text = 'abcd'
# encode_text = encode_sequence(text)
# print(encode_text)
# print(decode_sequence(encode_text))
# print(random_string())


class ReverseDataset(Dataset):
    def __init__(self, num_sample=2000, min_len=3, max_len=7):
        self.data = []
        for _ in range(num_sample):
            s = random_string(min_len, max_len)
            input = encode_sequence(s)
            target = encode_sequence(s[::-1])
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

def collate_fn(batch):
    inp_seq, tgt_seq = zip(*batch)

    # # 텐서로 변환
    # seq = [torch.tensor(s, dtype=torch.long) for s in inp_seq]
    # tgt = [torch.tensor(t, dtype=torch.long) for t in tgt_seq]

    # # 가장 긴 시퀀스 길이로 패딩
    # padded_seq = pad_sequence(seq, batch_first=True, padding_value=PAD_IDX)
    # padded_tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)

    # return padded_seq, padded_tgt

    inp_lens = [len(s) for s in inp_seq]
    tgt_lens = [len(t) for t in tgt_seq]

    # 당연히 길이 같지만 가독성을 위해
    max_inp = max(inp_lens)
    max_tgt = max(tgt_lens)

    padded_inp = []
    padded_tgt = []

    for inp, tgt in zip(inp_seq, tgt_seq):
        pad_len_inp = max_inp - len(inp)
        padded_inp.append(torch.cat([inp, torch.full((pad_len_inp,), PAD_IDX, dtype=torch.long)])) # PAD_IDX를 (pad_len_inp,) shape만큼 채우기

        pad_len_tgt = max_tgt - len(tgt)
        padded_tgt.append(torch.cat([tgt, torch.full((pad_len_tgt,), PAD_IDX, dtype=torch.long)]))

    batch_inp = torch.stack(padded_inp, dim=0)
    batch_tgt = torch.stack(padded_tgt, dim=0)

    return batch_inp, batch_tgt


train_dataset = ReverseDataset(num_sample=2000)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_step, hidden):
        input_step = input_step.unsqueeze(1) # (32,) -> (32,1)
        embedded = self.embedding(input_step) # (batch, seq_len) -> (batch, seq_len, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # output = (batch, seq_len, hidden) = (32, 1, hidden_size)
        # hidden = (1, batch, hidden) = (1, 32, hidden_size)

        output = output.squeeze(1) # (batch, 1, hidden_size) -> (batch, hidden_size)
        logits = self.fc_out(output) # (batch, vocab_size)

        return logits, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_rate=0.7):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        outputs = torch.zeros(batch_size, tgt_len, vocab_size)

        _, hidden = self.encoder(src)
        input_step = tgt[:, 0] # <SOS>

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_step, hidden)
            outputs[:, t , :] = logits # logits:(batch, vocab_size)

            teacher_force = (random.random() < teacher_forcing_rate)

            top1 = logits.argmax(dim=1)

            if teacher_force:
                input_step = tgt[:, t]
            else:
                input_step = top1

        return outputs
    
embedded_dim = 64
hidden_dim = 256
num_epochs = 30

encoder = Encoder(vocab_size, embedded_dim, hidden_dim)
decoder = Decoder(vocab_size, embedded_dim, hidden_dim)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    total_token = 0

    for src_batch, tgt_batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
        optimizer.zero_grad()

        outputs = model(src_batch, tgt_batch, teacher_forcing_rate=0.7) # (batch_size, tgt_len, vocab_size)
        outputs_reshape = outputs[:, 1:, :].reshape(-1, vocab_size) # (N, C) = (batch*tgt_len, vocab_size)
        tgt_reshape = tgt_batch[:, 1:].reshape(-1) # (N,) = (tgt_len*vocab_size)

        loss = criterion(outputs_reshape, tgt_reshape)

        loss.backward()
        optimizer.step()

        valid_tokens = (tgt_reshape != PAD_IDX).sum().item()
        total_loss += (loss.item() * valid_tokens)
        total_token += valid_tokens

    avg_loss = total_loss / total_token
    print(f'Epoch : {epoch} - loss : {avg_loss:.4f}')

def predict(model, s, max_len=20):
    model.eval()
    with torch.no_grad():
        src = encode_sequence(s).unsqueeze(0)

        _, hidden = model.encoder(src)

        input_step = torch.tensor([SOS_IDX])
        outputs = []

        for _ in range(max_len):
            logits, hidden = model.decoder(input_step, hidden)
            top1 = logits.argmax(dim=1)
            if top1.item() == EOS_IDX:
                break
            outputs.append(top1.item())
            input_step = top1
            
        pred_str = decode_sequence(outputs)
        return pred_str

test_sample = ['abcde', 'xyz', 'hello', 'korea']
for t in test_sample:
    print(f"원본 : {t}")
    print(f"정답 : {t[::-1]}")
    print(f"예측 : {predict(model, t)}")
    print()

