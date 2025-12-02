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



# 인코딩 우선 파일 받고 층화추출하고 패딩은 안하고
def get_encoding_value(
        filename: str, 
        file_type: str, 
        encoding_column_name: str, # 인코딩할 컬럼 이름(함수의 재사용성을 위해: 여러파일에서 벡터대상 컬럼이 다를 수 있으므로)
        label_column_name: str, # 샘플링을 할때 층화추출을 하고 싶다면 그 기준이 되는 칼럼이름
        add_stopwords: set, # 불용어에 추가단어가 있을때
        sample_size:int) -> tuple:
    
    # 데이터로드
    if file_type == "csv" :
        df = pd.read_csv(filename, encoding="utf-8")
    elif file_type == "json" :
        df = pd.read_json(filename, encoding="utf-8")
    else:
        print("지원하지 않는 파일타입입니다.")
        return 
    
