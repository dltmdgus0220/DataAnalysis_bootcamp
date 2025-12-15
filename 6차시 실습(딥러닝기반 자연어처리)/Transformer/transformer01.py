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
    

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    batch: [(src, trg), (src, trg), ...]
      src: (src_len,)
      trg: (trg_len,)  -> [<sos>, x1, x2, ..., xL, <eos>]
    리턴:
      - src_batch : (B, S)
      - trg_input : (B, T) [<sos>, x1, ..., xL]
      - trg_output: (B, T) [x1, ..., xL, <eos>]
      - src_key_padding_mask: (B, S)  (PAD 위치=True)
      - tgt_key_padding_mask: (B, T)  (PAD 위치=True)
    """

    src_list, trg_list = zip(*batch)

    # src 패딩
    src_batch = pad_sequence(
        src_list,
        batch_first=True,
        padding_value=PAD_IDX
    ) # (B, S)

    # trg_input, trg_output 분리
    trg_input_list = []
    trg_output_list = []
    for trg in trg_list:
        # trg: [<sos>, x1, ..., xL, <eos>]
        trg_input_list.append(trg[:-1]) # [<sos>, x1, ..., xL]
        trg_output_list.append(trg[1:]) # [x1, ..., xL, <eos>]

    trg_input = pad_sequence(
        trg_input_list,
        batch_first=True,
        padding_value=PAD_IDX
    ) # (B, T)
    trg_output = pad_sequence(
        trg_output_list,
        batch_first=True,
        padding_value=PAD_IDX
    ) # (B, T)
    
    # Transformer는 key_padding_mask에서
    # "True = 가려야 할 위치(PAD)" 로 사용
    src_key_padding_mask = (src_batch == PAD_IDX) # (B, S)
    tgt_key_padding_mask = (trg_input == PAD_IDX) # (B, T)

    return src_batch, trg_input, trg_output, src_key_padding_mask, tgt_key_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model)) # 스케일링을 위한 값, (d_model//2,)

        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 차원
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 차원

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe) # 모델 파라미터들 저장할 때 같이 저장됨.

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x += self.pe[:, :seq_len, :]
        return self.dropout(x)
    

class TransformerCopyModel(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            d_model:int,
            nhead:int,
            num_encoder_layers:int,
            num_decoder_layers:int,
            dim_feedforward:int,
            dropout:float,
            pad_idx:int):
        
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.tok_embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead, # d_model을 nhead만큼 나눠서 q,k,v 따로 attention 계산 후 concat -> 그 다음 linear projection, shape 변화 없음.
            dim_feedforward,
            dropout,
            batch_first=True
        ) # Self-Attention(단어간 관계 학습) -> FFN(단어 각각의 의미 학습), 각 단계마다 Residual(잔차 더해줌) -> LayerNorm(정규화) 수행
        # (batch, seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_encoder_layers
        ) # encoder_layer를 num_encoder_layer 만큼 쌓아서 블록구조로 만들어줌
        # 인코더의 output은 입력 문장의 문맥정보를 요약한 벡터, 이에 어떤 가중치를 곱해주냐에 따라 q,k,v로 활용

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True
        ) # mask mha(문장 내 모든 단어들에 대해 병렬적으로 수행) -> cross mha(인코더의 k,v를 통해 어떤 정보를 참고할지 결정)
        # -> FFN(비선형변환, 표현강화) -> 다음단어 생성
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_decoder_layers
        ) # 디코더의 output은 다음단어를 예측하기 위한 고차원의 의미 벡터

        self.fc_out = nn.Linear(d_model, vocab_size) # 고차원의 의미 벡터를 vocab_size로 차원을 줄여줌. 이후 softmax를 통과하여 확률분포로 만들고 제일 높은 확률의 단어로 예측

    def generate_square_subsequent_mask(self, sz:int):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        # (sz,sz) 짜리 1로 채워진, 즉 true로 채워진 정사각행렬 생성
        # triu:상삼각행렬, diagonal=1 대각선 위 한칸부터 상삼각행렬로 보겠다는 뜻, 이 부분만 true로 남기고 나머지 false로
        # true인 부분이 마스크
        return mask
    
    def forward(
            self,
            src:torch.Tensor,
            tgt_input:torch.Tensor,
            src_key_padding_mask:torch.Tensor,
            tgt_key_padding_mask:torch.Tensor
    ):
        B, S = src.size()
        B2, T = tgt_input.size()

        # 임베딩 + 위치 인코딩
        src_emb = self.tok_embedding(src) * math.sqrt(self.d_model) # 임베딩 정보에 더 가중치를 줌
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tok_embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask) # 인코더 통과

        tgt_mask = self.generate_square_subsequent_mask(T)

        out = self.decoder(
            tgt = tgt_emb,
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask
        ) # 디코더 통과
        
        logits = self.fc_out(out)

        return logits


def indices_to_tokens(indices: List[int]) -> List[str]:
    return [VOCAB_TOKENS[i] for i in indices]

def print_example(src, trg_output, pred_indices):
    """
    src: (S,)
    trg_output: (T,)
    pred_indices: (T,)
    """
    src_tokens = indices_to_tokens(src)
    trg_tokens = indices_to_tokens(trg_output)
    pred_tokens = indices_to_tokens(pred_indices)
    print("-------------------------------------------------")
    print("SRC        :", " ".join(src_tokens))
    print("TRG (gold) :", " ".join(trg_tokens))
    print("PRED       :", " ".join(pred_tokens))
    print("-------------------------------------------------")


def train():
    # 1) 데이터셋 / 데이터로더
    train_dataset = CopyDataset(
        num_samples=NUM_TRAIN_SAMPLES,
        min_len=MIN_SEQ_LEN,
        max_len=MAX_SEQ_LEN,
        vocab_start=3,
        vocab_end=VOCAB_SIZE - 1,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX
    )
    valid_dataset = CopyDataset(
        num_samples=NUM_VALID_SAMPLES,
        min_len=MIN_SEQ_LEN,
        max_len=MAX_SEQ_LEN,
        vocab_start=3,
        vocab_end=VOCAB_SIZE - 1,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX
    )
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

    # 2) 모델 준비
    model = TransformerCopyModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    )

    # 3) 옵티마이저, 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def run_one_epoch(dataloader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.set_grad_enabled(train_mode):
            for (
                src,
                trg_input,
                trg_output,
                src_key_padding_mask,
                tgt_key_padding_mask,
            ) in dataloader:
                
                logits = model(src, trg_input, src_key_padding_mask, tgt_key_padding_mask)
                # logits: (B, T, V)
                B, T, V = logits.size()
                loss = criterion(
                    logits.view(B * T, V),
                    trg_output.reshape(-1)
                )

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                non_pad = (trg_output != PAD_IDX).sum().item()
                total_loss += loss.item() * non_pad
                total_tokens += non_pad

        if total_tokens == 0:
            return 0.0
        
        return total_loss / total_tokens
    
    # 4) Epoch 루프
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_one_epoch(train_loader, train_mode=True)
        valid_loss = run_one_epoch(valid_loader, train_mode=False)
        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f}  valid_loss={valid_loss:.4f}")
        
    # 5) 학습 후 예시 출력
    model.eval()
    with torch.no_grad():
        src, trg_input, trg_output, src_key_padding_mask, tgt_key_padding_mask = next(iter(valid_loader)) # valid셋에서 batch 하나 가져오기
        logits = model(src, trg_input, src_key_padding_mask, tgt_key_padding_mask)
        preds = logits.argmax(dim=-1)  # (B, T)
        num_show = min(3, src.size(0))

        for i in range(num_show):
            src_i = src[i]
            trg_i = trg_output[i]
            pred_i = preds[i]

            # PAD 제거
            src_i_nopad = src_i[src_i != PAD_IDX]
            trg_i_nopad = trg_i[trg_i != PAD_IDX]
            pred_i_nopad = pred_i[pred_i != PAD_IDX]

            # 길이 맞추기
            T_eff = len(trg_i_nopad)
            pred_i_cut = pred_i_nopad[:T_eff]
            print_example(
                src_i_nopad.tolist(),
                trg_i_nopad.tolist(),
                pred_i_cut.tolist()
            )

if __name__ == "__main__":
    train()