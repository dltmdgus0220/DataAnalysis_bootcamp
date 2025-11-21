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


