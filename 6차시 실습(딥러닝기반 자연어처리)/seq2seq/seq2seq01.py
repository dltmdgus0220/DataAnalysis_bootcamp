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



    return padded_seq, padded_target
