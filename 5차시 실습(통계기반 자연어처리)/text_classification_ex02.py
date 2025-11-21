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


#============================
# 파이프라인 생성
#============================
# CountVectorizer
nb_ct_clf = Pipeline(steps=[
    ('vect', CountVectorizer(
        tokenizer=preprocess_text,
        token_pattern=None,
        lowercase=False
    )),
    ('model', MultinomialNB())
])
# TfidfVectorizer
nb_tfidf_clf = Pipeline(steps=[
    ('vect', TfidfVectorizer(
        tokenizer=preprocess_text,
        token_pattern=None,
        lowercase=False
    )),
    ('model', MultinomialNB())
])

#============================
# 그리드 서치
#============================
nb_param_grid = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__min_df': [1,2,3],
    'model__alpha': [0.1, 0.5, 1.0, 2.0]
}

# MultinomialNB + CountVectorizer
gs_ct = GridSearchCV(
    nb_ct_clf,
    param_grid=nb_param_grid,
    scoring="f1_macro", # 불균형 데이터 고려하면 macro-F1 추천
    cv=3,
    n_jobs=1 # okt를 쓰게되면 병렬처리가 불가능, 미리 전처리하고 기본 토크나이저를 쓰면 해결가능
)

# 학습
gs_ct.fit(x_tr, y_tr)
print("== MultinomialNB + CountVectorizer ==")
print("Best params:", gs_ct.best_params_)
print("Best macro-F1 (cv):", gs_ct.best_score_)

# 테스트
best_gs_ct = gs_ct.best_estimator_
y_pred = best_gs_ct.predict(x_te)
print(classification_report(y_te, y_pred, digits=3))
macro_f1_ct = f1_score(y_te, y_pred, average='macro')
print(f"macro_f1 : {macro_f1_ct:.3f}")

