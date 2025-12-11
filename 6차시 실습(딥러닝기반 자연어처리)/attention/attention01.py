import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



# 상수 선언
NUM_LETTERS = 8
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
VOCAB_TOKENS = ['<pad>', '<sos>', '<eos>'] + [chr(ord('a')+i) for i in range(NUM_LETTERS)]
# print(VOCAB_TOKENS)

VOCAB_SIZE = len(VOCAB_TOKENS)

MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 7
NUM_TRAIN_SAMPLES = 2000
NUM_VALID_SAMPLES = 200

# 하이퍼파라미터
EMBED_DIM = 32
HIDDEN_SIZE = 64
ATTN_DIM = 64

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3


# 데이터셋 클래스
class CopyDataset(Dataset):
    def __init__(
            self,
            num_samples:int,
            min_len:int,
            max_len:int,
            vocab_start:int,
            vocab_end:int,
            sos_index:int,
            eos_index:int    ):
        super().__init__()

        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.sos_index = sos_index
        self.eos_index = eos_index

        self.data = [self._make_sample() for _ in range(self.num_samples)]

    def _make_sample(self) -> Tuple[torch.tensor, torch.tensor]:
        length = random.randint(self.min_len, self.max_len)
        src = [random.randint(self.vocab_start, self.vocab_end) for _ in range(length)]
        tgt = [self.sos_index] + src + [self.eos_index]
        return torch.tensor(src,dtype=torch.long), torch.tensor(tgt,dtype=torch.long)
        
    def __len__(self) -> int:
        return self.num_samples 
    
    def __getitem__(self, index:int) -> Tuple[torch.tensor, torch.tensor]:
        return self.data[index]
    
# 배치 생성 규칙 함수
def collate_fn(batch:List[Tuple[torch.tensor, torch.tensor]]):
    src, tgt = zip(*batch)
    # src=[3,7,9,12] / tgt=[1,3,7,9,12,2]
    # tgt_input = [1,3,7,9,12]
    # tgt_output = [3,7,9,12,2]

    padded_src = pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    mask = (padded_src != PAD_IDX).long()

    for t in tgt:
        tgt_input  = [t[:-1]]
        tgt_output = [t[1:]]
    padded_tgt_input = pad_sequence(tgt_input, batch_first=True, padding_value=PAD_IDX)
    padded_tgt_output = pad_sequence(tgt_output, batch_first=True, padding_value=PAD_IDX)
    

    return padded_src, mask, padded_tgt_input, padded_tgt_output # 각각은 전부 텐서

# 인코더 클래스
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, src, src_len=None):
        emb = self.embedding(src)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden.squeeze(0)

# 어텐션 클래스
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size_enc, hidden_size_dec, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_size_enc, attn_dim, bias=False)
        self.W_s = nn.Linear(hidden_size_dec, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_hidden, decoder_hidden, mask=None):
        Wh = self.W_h(encoder_hidden)
        Ws = self.W_s(decoder_hidden).unsqueeze(1)
        score = self.v_a(torch.tanh(Wh+Ws)).squeeze(-1)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_hidden) # (B, 1, H)

        return context, attn_weights
    
# 디코더 클래스
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.rnn = nn.GRU(
            embed_dim + hidden_size,
            hidden_size,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.attention = AdditiveAttention(
            hidden_size_enc=hidden_size,
            hidden_size_dec=hidden_size,
            attn_dim=hidden_size
        )

    def forward(self, tgt_input, encoder_outputs, encoder_mask, hidden):
        B, T_in = tgt_input.size()
        emb = self.embedding(tgt_input) # (B, T_in, E)

        outputs = []
        attn_list = []

        decoder_hidden = hidden
        input_step = emb[:, 0, :] # 첫 입력 <sos> , (B, E)
        
        for t in range(1, T_in):
            context, attn_weights = self.attention(
                encoder_outputs,
                decoder_hidden,
                encoder_mask
            ) # context:(B, S)

            attn_list.append(attn_weights.unsqueeze(1)) # (B, 1, S)

            rnn_input = torch.cat([input_step, context], dim=-1).unsqueeze(1) # (B,E)+(B,H) => (B,E+H) => (B,1,E+H)
            output, new_hidden = self.rnn(
                rnn_input, # (B,1,E+H)
                decoder_hidden.unsqueeze(0) # (1,B,H)
            ) # output:(B,1,H), new_hidden:(1,B,H)
            decoder_hidden = new_hidden.squeeze(0)   # (B, H)

            logits = self.fc_out(output.squeeze(1))  # (B, vocab_size)
            outputs.append(logits.unsqueeze(1))

            input_step = emb[:, t, :] # (B, E), teacher forcing

        outputs = torch.cat(outputs, dim=1) # (B, T-1, vocab_size)
        attn_weights_all = torch.cat(attn_list, 1) # (B, T-1, S)
        return outputs, attn_weights_all
    
# 공백 기준 토큰화 함수
def indices_to_tokens(indices: List[int]) -> List[str]:
    return [VOCAB_TOKENS[i] for i in indices]

# 출력 함수
def print_example(src, trg_input, trg_output, pred_indices):
    """
    src: (S,)
    trg_input: (T,)
    trg_output: (T,)  # [x1, ..., xL, <eos>]
    pred_indices: (T,) # 예측 토큰 인덱스
    """
    src_tokens = indices_to_tokens(src)
    trg_tokens = indices_to_tokens(trg_output)
    pred_tokens = indices_to_tokens(pred_indices)
    print("-------------------------------------------------")
    print("SRC        :", " ".join(src_tokens))
    print("TRG (gold) :", " ".join(trg_tokens))
    print("PRED       :", " ".join(pred_tokens))
    print("-------------------------------------------------")


"""
1. 훈련데이터셋생성(CopyDataset)
2. 검증데이터셋생성
3. 훈련/검증 데이터로더 생성
4. encoder 생성
5. decoder 생성
6. seq2seq 모델 생성
7. 손실함수 생성
8. 옵티마이저생성
9. train_one_epoch() 생성
    - 손실을 구할 때 logits(N,C) , tgt_output(N)을 넘김
    - 평균 손실은 토큰 단위로
    - 1 epoch 동안 (훈련(2000) / 검증(200)) 둘다 출력
"""

# seq2seq 클래스
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, tgt_input):
        # src:(B, src_len), mask:(B, src_len), tgt_input(B, tgt_len)
        # tgt_len = src_len+1

        encoder_outputs, encoder_hidden = self.encoder(src)
        logits, attn_weights_all = self.decoder(
            tgt_input,
            encoder_outputs,
            src_mask,
            encoder_hidden
        )

        return logits, attn_weights_all


def train_one_epoch(model, loader, criterion, optimizer, epoch=1):
    model.train()

    total_loss = 0.0
    total_token = 0

    for src_batch, src_mask, tgt_input, tgt_output in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        # src_batch:(B, src_len), src_mask:(B, src_len), tgt_input(B, tgt_len), tgt_output(B, tgt_len)
        # tgt_len = src_len+1

        optimizer.zero_grad()
        logits, attn = model(src_batch, src_mask, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        batch_tokens = (tgt_output != PAD_IDX).sum().item()

        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / total_token

# 데이터셋 생성
train_dataset = CopyDataset(
    num_samples=NUM_TRAIN_SAMPLES,
    min_len=MIN_SEQ_LEN,
    max_len=MAX_SEQ_LEN,
    vocab_start=3,
    vocab_end=3 + NUM_LETTERS - 1,
    sos_index=SOS_IDX,
    eos_index=EOS_IDX
)
valid_dataset = CopyDataset(
    num_samples=NUM_VALID_SAMPLES,
    min_len=MIN_SEQ_LEN,
    max_len=MAX_SEQ_LEN,
    vocab_start=3,
    vocab_end=3 + NUM_LETTERS - 1,
    sos_index=SOS_IDX,
    eos_index=EOS_IDX
)

# 데이터로더 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# 인코더/디코더 생성
encoder = Encoder(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX
)
decoder = DecoderWithAttention(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX
)

# seq2seq 모델 생성
model = Seq2Seq(encoder, decoder)

# 손실함수/옵티마이저 생성
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)