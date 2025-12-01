import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sentences = [
    "배송이 빠르고 포장이 깔끔해요",
    "배송이 너무 느리고 제품이 마음에 안 들어요",
    "가격이 저렴해서 만족스러워요",
    "포장이 엉망이고 배송도 늦었어요",
]
labels = [1, 0, 1, 0]

def tokenize(text:str)->list:
    text = re.sub(r'[^가-힣0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    return tokens

tokenized_sentences = [tokenize(s) for s in sentences]
# print(tokenized_sentences)


counter = Counter()
for tokens in tokenized_sentences:
    counter.update(tokens)
# print(counter)


# 특수 토큰 정의
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, _ in counter.most_common():
    vocab[word] = len(vocab)
# print(vocab)

def encode(tokens, vocab, unk_token=UNK_TOKEN):
    unk_idx = vocab[unk_token]
    return [vocab.get(t, unk_idx) for t in tokens]

encoded_sentences = [encode(tokens, vocab) for tokens in tokenized_sentences]
# print(encoded_sentences)



def pad_sequences(encoded_list, max_len, pad_value=0):
    padded = []
    masks = []
    for seq in encoded_list:
        if len(seq) > max_len:
            # 너무 길면 자르기
            seq = seq[:max_len]
        # 패딩 길이 계산
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        mask = [1] * len(seq) + [0] * pad_len

        padded.append(padded_seq)
        masks.append(mask)
    return torch.tensor(padded), torch.tensor(masks)


max_len = 6
padded_inputs, attention_masks = pad_sequences(encoded_sentences, max_len, pad_value=vocab[PAD_TOKEN])
print('Padded inputs :\n')
print(padded_inputs)
print('Attention masks :\n')
print(attention_masks)
print('Tensor shape :',padded_inputs.shape)

