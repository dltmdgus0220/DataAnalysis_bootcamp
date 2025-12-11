import math
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# 상수 선언
NUM_LETTERS = 8
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
VOCAB_TOKENS = ['<pad>','<sos>','<eos>'] + [chr(ord('a')+i) for i in range(NUM_LETTERS)]
VOCAB_SIZE = len(VOCAB_TOKENS)

MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 8
NUM_TRAIN_SAMPLES = 20000
NUM_VALID_SAMPLES = 2000

D_MODEL = 16
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 32

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
DROPOUT = 0.2


class CopyDataset(Dataset):
    """
    랜덤 문자 시퀀스를 만들고,
    타깃은 [<sos>] + src + [<eos>] 형태로 생성하는 Dataset.
    """
    def __init__(
        self,
        num_samples: int,
        min_len: int,
        max_len: int,
        vocab_start: int,
        vocab_end: int,
        sos_idx: int,
        eos_idx: int,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.data = [self._make_sample() for _ in range(num_samples)]

    def _make_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = random.randint(self.min_len, self.max_len)

        # 문자 영역: [vocab_start, vocab_end] (정수 토큰)
        src_tokens = [random.randint(self.vocab_start, self.vocab_end) for _ in range(length)]
        src = torch.tensor(src_tokens, dtype=torch.long)

        # trg: [<sos>, x1, ..., xL, <eos>]
        trg = torch.tensor([self.sos_idx] + src_tokens + [self.eos_idx], dtype=torch.long)
        return src, trg
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
    
