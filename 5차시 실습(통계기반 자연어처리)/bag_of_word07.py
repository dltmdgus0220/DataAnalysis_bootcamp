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


