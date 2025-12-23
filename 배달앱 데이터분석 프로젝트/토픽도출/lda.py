from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from konlpy.tag import Okt
from kiwipiepy import Kiwi

import numpy as np
import pandas as pd

# --- 1. 데이터 불러오기 ---

df = pd.read_csv('배달앱 데이터분석 프로젝트/데이터/전처리데이터/preprocessed_yogiyo_reviews_playstore.csv', encoding='utf-8-sig')
docs = df['content']

# --- 2. 토크나이저 함수 생성 ---

add_stopwords = ['땡기', '요기']
with open('stopwords-ko.txt', encoding='utf-8') as f:
    stopwords = set(w.strip() for w in f if w.strip())
if add_stopwords is not None:
    stopwords.update(add_stopwords)

okt = Okt()
def okt_tokenizer(text:str) -> list:
    ret = []
    tokens = okt.pos(text, stem=True)
    for w, p in tokens:
        if p in ['Noun', 'Adjective', 'Verb'] and len(w) > 1 and w not in stopwords:
            ret.append(w)
    return ret

kiwi = Kiwi()
def kiwi_tokenizer(text:str) -> list:
    ret = []
    tokens = kiwi.tokenize(text)
    for t in tokens:
        if t.tag.startswith(('N', 'VA', 'VV')) and len(t.form) > 1 and t.form not in stopwords: # 명사, 형용사, 동사
            ret.append(t.form)
    return ret

# --- 3. 벡터화객체 생성 ---

vectorizer = CountVectorizer(
    # tokenizer=okt_tokenizer,
    tokenizer=kiwi_tokenizer,
    ngram_range=(1, 2), # ex 배달, 배달느림 같이 잡기
    min_df=5, # 너무 희귀한 단어 제거
    max_df=0.9, # 너무 흔한 단어 제거
)
X = vectorizer.fit_transform(docs)

# --- 4. LDA 모델 생성 및 학습 ---

n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method='batch',
    random_state=42
)

#  학습 (문서-토픽 분포 θ_hat을 반환)
doc_topic = lda.fit_transform(X)


# --- 5. 결과 출력 ---

# 토픽별 상위 단어 출력 함수
def print_topics(model, feature_names, n_top_word=5):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_word]
        top_words = [feature_names[i] for i in top_indices]
        print(f'Topic {topic_idx} : {", ".join(top_words)}')

feature_names = vectorizer.get_feature_names_out()
print_topics(lda, feature_names, 5)
print()

# 문서별 토픽 비율 보기
# for i, topic_dist in enumerate(doc_topic):
#     print(f"문서 {i} 토픽 분포:", np.round(topic_dist, 3))