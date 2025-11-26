import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")

plt.rcParams['axes.unicode_minus']=False

okt = Okt()

df = pd.read_json(rf"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\woman_wear_sample_600.json", encoding="utf-8")

df = df.dropna(subset=["RawText", "GeneralPolarity"])
df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)

X = df["RawText"]
y = df["GeneralPolarity"]

print("\n레이블 분포(클래스별 샘플 수)")
print(y.value_counts())

with open(r"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\stopwords-ko.txt", encoding="utf-8") as f:
    stopwords = set(w.strip() for w in f if w.strip())

add_word = set(["하다", "입다", "이다", "않다", "입다"])

stopwords.update(add_word)

def preprocess_text(text: str) -> list:
    text = text.lower()

    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

    morphs = okt.pos(text, norm=True, stem=True)
    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords and len(word) > 1:
                tokens.append(word)
    return tokens

part_df = [ df[df["GeneralPolarity"] == i ] for i in df['GeneralPolarity'].unique()]

plt.figure(figsize=(15, 5))
for i, p_df in enumerate(part_df, start=1):
    n_topics = 5
    pipe_lsa = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            max_df=0.8, 
            min_df=3,
            token_pattern=None, 
            tokenizer=preprocess_text
        )),
        ("svd", TruncatedSVD(
            n_components=n_topics,
            random_state=42
        ))
    ])

    X_lsa = pipe_lsa.fit_transform(p_df["RawText"])
    tfidf = pipe_lsa.named_steps['tfidf']
    svd = pipe_lsa.named_steps['svd']

    # print("TF-IDF shape : ", tfidf.transform(p_df).shape)
    # print("LSA shape : ", X_lsa.shape)

    terms = tfidf.get_feature_names_out()

    for topic_idx, comp in enumerate(svd.components_):
        term_idx = comp.argsort()[::-1][:20]
        print(f"\n[토픽 {topic_idx}]")
        print(", ".join(terms[i] for i in term_idx))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )
    X_umap = reducer.fit_transform(X_lsa)

    plt.subplot(1,3,i)
    sc = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        s = 5,
        cmap="bwr",
        alpha = 0.7
       )
    plt.title(f"UMAP GeneralPolarity ({np.array(p_df['GeneralPolarity'])[0]})")
    plt.xticks([])
    plt.yticks([])


plt.suptitle("UMAP : GeneralPolarity에 따른 시각화 비교")
plt.tight_layout()
plt.show()
        

