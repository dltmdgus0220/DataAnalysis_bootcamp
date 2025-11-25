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


X_lsa = pipe_lsa.fit_transform(texts)
tfidf = pipe_lsa.named_steps['tfidf']
svd = pipe_lsa.named_steps['svd']

print(f'TF-IDF shape : {tfidf.transform(texts).shape}')
print(f'LSA shape : {X_lsa.shape}')

terms = tfidf.get_feature_names_out()

for topic_idx, comp in enumerate(svd.components_):
    term_idx = comp.argsort()[::-1][:10]
    print(f'\n[토픽 {topic_idx}]')
    print(', '.join(terms[i] for i in term_idx))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_lsa)

df_sample['tsne_x'] = X_2d[:, 0]
df_sample['tsne_y'] = X_2d[:, 1]

plt.figure(figsize=(6,5))
scatter = plt.scatter(
    df_sample['tsne_x'],
    df_sample['tsne_y'],
    c=df_sample['label'],
    s=15,
    alpha=0.7
)
plt.title('문서들의 2D LSA공간(t-SNE)')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar(scatter)
plt.tight_layout()
plt.show()


extract_df = df_sample[(df_sample['tsne_y'] >= -40) & (df_sample['tsne_y'] <= -30)]
# print(len(extract_df))
random10_df = extract_df.sample(n=10, random_state=42)
print(random10_df['document'])