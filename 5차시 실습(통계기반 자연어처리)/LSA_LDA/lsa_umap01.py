import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap

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


X_lsa = pipe_lsa.fit_transform(texts)
tfidf = pipe_lsa.named_steps['tfidf']
svd = pipe_lsa.named_steps['svd']

neighbors_list = [5, 10, 15]

plt.figure(figsize=(15,4))

for i, n_nb in enumerate(neighbors_list, start=1):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_nb,
        min_dist=0.5,
        metric='cosine',
        random_state=42
    )

    X_umap = reducer.fit_transform(X_lsa)

    plt.subplot(1,3,i)
    sc = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c = df_sample['label'],
        s = 5,
        cmap = 'bwr',
        alpha = 0.7
    )
    plt.title(f'UMAP (n_neighbors = {n_nb})')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('UMAP : n_neighbors에 따른 시각화 비교')
plt.tight_layout()
plt.show()

# 이후에 HDBSCAN이나 KMeans등으로 군집을 생성하고 군집별 데이터를 확인하는 작업이 필요함.