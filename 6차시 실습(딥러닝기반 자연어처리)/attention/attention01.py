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
