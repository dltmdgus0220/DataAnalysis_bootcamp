from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from module.encoding_seq import get_encoding_value


df, vocab = get_encoding_value('5차시 실습(통계기반 자연어처리)/movie_reviews.csv', 'csv', 'document', 'label', None, 10000)
# print(df.head())
print((df['encoded'].apply(lambda x: len(x))).median()) # 6


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
encoded_list = df['encoded'].tolist()
padded_inputs, attention_masks = pad_sequences(encoded_list, max_len)
labels = df['label']
print('Padded inputs :\n', padded_inputs)
print('Attention masks :\n', attention_masks)
print('Labels :\n', labels)
print('Tensor shape :', padded_inputs.shape)



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


# # =========================================================
# class ReviewRawDataset(Dataset):
#     # 아직 패딩하지 않은 정수 시퀀스를 보관하는 Dataset
#     def __init__(self, encoded_sequences, labels):
#         self.encoded_sequences = [torch.tensor(seq, dtype=torch.long) for seq in encoded_sequences]
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         return self.encoded_sequences[idx], self.labels[idx]
    

# def collate_fn(batch, pad_value=0):    
#     # batch: list of (seq_tensor, label)    
#     seqs, labels = zip(*batch)  # unzip
#     # 길이 다른 시퀀스를 pad_sequence로 패딩
#     padded_seqs = pad_sequence(
#         seqs,
#         batch_first=True, # (batch, max_len) 형태
#         padding_value=pad_value
#     )

#     # 마스크 생성: 패딩이 아닌 부분(≠pad_value)을 1로
#     attention_mask = (padded_seqs != pad_value).long()
#     labels = torch.stack(labels) # tuple을 tensor로 변경
#     return {
#         "input_ids": padded_seqs,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }
    

# raw_dataset = ReviewRawDataset(encoded_sentences, labels)
# loader2 = DataLoader(
#     raw_dataset,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=lambda batch: collate_fn(batch, pad_value=vocab[PAD_TOKEN])
# )
# batch2 = next(iter(loader2))
# print("동적 패딩 input_ids shape:", batch2["input_ids"].shape)
# print("동적 패딩 attention_mask:\n", batch2["attention_mask"])