import re
import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Okt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap

import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


okt = Okt()
stopwords = ['하다', '입다', '이다', '같다', '좀', 
             '조금', '있다', '개', '이다', '요',
             '성은', '듭니', '해', '해도', '이렇다', '나다']


def preprocess(text):
    text = re.sub(r'[^가-힣0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    morphs_text = okt.pos(text, norm=True, stem=True)
    content_words = []
    for word, pos in morphs_text:
        if pos in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords: # and len(word) > 1:
                content_words.append(word)

    return ' '.join(content_words)


def fit_and_print_topic_word(pipe, texts):
    X_lsa = pipe.fit_transform(texts)
    tfidf = pipe.named_steps['tfidf']
    svd = pipe.named_steps['svd']

    print(f'TF-IDF shape : {tfidf.transform(texts).shape}')
    print(f'LSA shape : {X_lsa.shape}')

    terms = tfidf.get_feature_names_out()

    for topic_idx, comp in enumerate(svd.components_):
        term_idx = comp.argsort()[::-1][:20]
        print(f'[토픽 {topic_idx}]')
        print(', '.join(terms[i] for i in term_idx))
    
    return (X_lsa, tfidf, svd)


df = pd.read_json('5차시 실습(통계기반 자연어처리)/woman_wear_sample_600.json', encoding='utf-8')
# print(df.shape)
# print(df.head())
df['clean'] = df['RawText'].astype(str).apply(preprocess)
texts = df['clean']
print(texts[:5])

df_neg, df_neu, df_pos = df[df['GeneralPolarity']==-1], df[df['GeneralPolarity']==0], df[df['GeneralPolarity']==1] 
# print(len(df_pos), len(df_neg), len(df_neu))
texts_neg, texts_neu, texts_pos = df_neg['clean'], df_neu['clean'], df_pos['clean']

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

# 이후에 HDBSCAN이나 KMeans등으로 군집을 생성하고 군집별 데이터를 확인하는 작업이 필요함.

