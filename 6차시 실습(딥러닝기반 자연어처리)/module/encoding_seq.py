import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
from sklearn.model_selection import train_test_split

# 전역변수
okt = Okt()
counter = Counter()
# 특수 토큰 정의
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# 텍스트 전처리(불용어처리, 정제식, 품사)
def preprocess_text(text: str, stopwords: str) -> list:
    text = text.lower()    
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    morphs = okt.pos(text, stem=True)  

    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:            
            if word not in stopwords:# and len(word) > 1:
                tokens.append(word)

    return tokens

# 인코딩
def encode(tokens, vocab, unk_token=UNK_TOKEN):
    unk_idx = vocab[unk_token]
    return [vocab.get(t, unk_idx) for t in tokens]


