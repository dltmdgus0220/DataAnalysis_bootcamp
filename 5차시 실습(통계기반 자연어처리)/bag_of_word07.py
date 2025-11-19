import numpy as np
import pandas as pd

import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 1. 데이터로드
df = pd.read_csv('5차시 실습(통계기반 자연어처리)/movie_reviews.csv', encoding='utf8').dropna()
# print(df)


# 2. train/test 분할
df_sample, _ = train_test_split(df, train_size=10000, random_state=42, stratify=df['label']) # shuffle=True는 기본값이라 생략
text, label = df_sample['document'], df_sample['label']
x_tr, x_te, y_tr, y_te = train_test_split(text, label, test_size=0.2, random_state=42, stratify=label)
# print(len(x_tr), len(x_te))


# 3. 텍스트 전처리 함수
okt = Okt()
stopwords = [
    "이", "그", "저", "것", "거", "수",
    "그리고", "그런데", "하지만", "또는",
    "하면", "하고", "같은", "좀", "조금", "너무"
]

def tokenize_text(text:str):
    # 소문자 변환(영어)
    text_lower = text.lower()
    # 한글/영문/숫자/공백 외 특수문자 제거
    clean_text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text_lower)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip() # 연속적인 공백 제거 및 양끝 공백 제거
    # 형태소 분석 + 품사 태깅
    morphs_text = okt.pos(clean_text, norm=True, stem=True)
    # 사용할 품사만 선택 (명사, 동사, 형용사)
    # 불용어 제거
    content_words = [w for w,p in morphs_text if p in ['Noun', 'Verb', 'Adjective'] and w not in stopwords]

    return content_words


# 4. TfidfVectorizer로 벡터화
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_text,
    token_pattern=None, # 기본 정규식 비활성화
    min_df=1, # 1이면 의미없음
    ngram_range=(1,1) # unigram
)
x_tr_vec = vectorizer.fit_transform(x_tr)
x_te_vec = vectorizer.transform(x_te)
# print(vectorizer.get_feature_names_out().shape) # 10197개의 단어
# print(x_te_vec) # Coords(i,j):i번 문서의 j번 단어, values:TF-IDF 값


# 5. 모델 훈련(LogisticRegression)
clf = LogisticRegression(max_iter=1000) # 기본은 100이지만 nlp처럼 단어수가 많고 희소행렬이 클때, 또는 특징수가 많을 때 더 늘려줌.
clf.fit(x_tr_vec, y_tr) # 학습

y_pred = clf.predict(x_te_vec) # 테스트
print(classification_report(y_te, y_pred, digits=3))