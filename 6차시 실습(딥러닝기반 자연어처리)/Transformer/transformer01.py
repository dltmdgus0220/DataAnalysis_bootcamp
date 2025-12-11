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

