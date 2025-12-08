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

