import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


def simple_korea_clean(text):
    text = re.sub(r'[^가-힣0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_csv('5차시 실습(통계기반 자연어처리)/movie_reviews.csv', encoding='utf-8').dropna()

_, df_sample = train_test_split(df, test_size=1000, stratify=df['label'], shuffle=True, random_state=42)

df_sample['clean'] = df_sample['document'].astype(str).apply(simple_korea_clean)
texts = df_sample['clean']

n_topics = 5

pipe_lsa = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        max_df=0.7,
        min_df=5,
        token_pattern=r'(?u)\b\w+\b' 
    )),
    ('svd', TruncatedSVD(
        n_components=n_topics,
        random_state=42
    ))
])

