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


# =========================================================================
class ReviewDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids':self.inputs[idx],
            'attention_mask':self.masks[idx],
            'labels':self.labels[idx]
        }
    
dataset = ReviewDataset(padded_inputs, attention_masks, labels)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

batch = next(iter(loader))
print("input_ids batch shape:", batch["input_ids"].shape)
print("attention_mask batch shape:", batch["attention_mask"].shape)
print("labels batch:", batch["labels"])


# =========================================================
class ReviewRawDataset(Dataset):
    # 아직 패딩하지 않은 정수 시퀀스를 보관하는 Dataset
    def __init__(self, encoded_sequences, labels):
        self.encoded_sequences = [torch.tensor(seq, dtype=torch.long) for seq in encoded_sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.encoded_sequences[idx], self.labels[idx]
    

def collate_fn(batch, pad_value=0):    
    # batch: list of (seq_tensor, label)    
    seqs, labels = zip(*batch)  # unzip
    # 길이 다른 시퀀스를 pad_sequence로 패딩
    padded_seqs = pad_sequence(
        seqs,
        batch_first=True, # (batch, max_len) 형태
        padding_value=pad_value
    )

    # 마스크 생성: 패딩이 아닌 부분(≠pad_value)을 1로
    attention_mask = (padded_seqs != pad_value).long()
    labels = torch.stack(labels) # tuple을 tensor로 변경
    return {
        "input_ids": padded_seqs,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

raw_dataset = ReviewRawDataset(encoded_sentences, labels)
loader2 = DataLoader(
    raw_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_value=vocab[PAD_TOKEN])
)
batch2 = next(iter(loader2))
print("동적 패딩 input_ids shape:", batch2["input_ids"].shape)
print("동적 패딩 attention_mask:\n", batch2["attention_mask"])