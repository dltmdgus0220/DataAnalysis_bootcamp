import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

from konlpy.tag import Okt
import re

#============================
# 전처리 함수 생성
#============================
okt = Okt()
stopwords = [
    "이", "그", "저", "것", "거", "수",
    "그리고", "그런데", "하지만", "또는",
    "하면", "하고", "같은", "좀", "조금", "너무"
]

def preprocess_text(text:str):
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z가-힣\s]',' ', text).strip() # 정규식 사용 및 양끝 공백 처리
    text = re.sub(r'\s+', ' ', text) # 연속되는 공백 처리

    morphs = okt.pos(text, norm=True, stem=True) # 어간,표제어 처리
    ret = []
    # 불용어 제거
    for word, pos in morphs:
        if pos in ['Noun', 'Verb', 'Adjective']:
            if word not in stopwords and len(word) > 1:
                ret.append(word)
    return ret 


#============================
# 데이터로드
#============================
folder = '5차시 실습(통계기반 자연어처리)/1-1. 여성의류'

dfs = [] # 성능(속도, 메모리)적으로 리스트에 저장해놨다가 나중에 한번에 concat하는게 더 좋음.
for i in range(1,11):
    path = os.path.join(folder, f'1-1.여성의류({i}).json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f) # 100개씩 가지고 있음
            df_tmp = pd.DataFrame(data)
            dfs.append(df_tmp[['RawText','GeneralPolarity']])
df = pd.concat(dfs, ignore_index=True)
# print(df)
# print(df.info())
# print(df.isna().sum())
df = df.dropna() # 결측치 8개 행 제거 (998,2)


#============================
# train/test 분할
#============================
X, y = df['RawText'], df['GeneralPolarity'].astype(int)
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# print(len(x_tr), len(x_te)) # 793, 199

