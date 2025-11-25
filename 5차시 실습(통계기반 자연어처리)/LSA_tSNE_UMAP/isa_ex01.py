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

# 이후에 HDBSCAN이나 KMeans등으로 군집을 생성하고 군집별 데이터를 확인하는 작업이 필요함.

