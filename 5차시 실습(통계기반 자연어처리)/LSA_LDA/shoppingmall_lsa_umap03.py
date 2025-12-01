import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import platform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.model_selection import train_test_split


if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False


okt = Okt()

def preprocess_text(text: str, local_stopwords: str) -> list:
    text = text.lower()    
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)  
    morphs = okt.pos(text, stem=True)  

    tokens = []
    for word, tag in morphs:
        if tag in ["Noun", "Verb", "Adjective"]:            
            if word not in local_stopwords and len(word) > 1:
                tokens.append(word)

    return tokens

def get_vectorize_value(
        filename: str, 
        file_type: str, 
        vectorize_column_name: str, # 벡터화할 컬럼 이름(함수의 재사용성을 위해: 여러파일에서 벡터대상 컬럼이 다를 수 있으므로)
        label_column_name: str, # 샘플링을 할때 층화추출을 하고 싶다면 그 기준이되는 칼럼이름
        add_stopwords: set, # 불용어에 추가단어가 있을때
        sample_size:int) -> tuple:   
    if file_type == "csv" :
        df = pd.read_csv(filename, encoding="utf-8")
    elif file_type == "json" :
        df = pd.read_json(filename, encoding="utf-8")
    else:
        print("지원하지 않는 파일타입입니다.")
        return 
    
       
    df = df.dropna()
    TEXT_COL = vectorize_column_name

    if label_column_name is not None :
        stratify_column = df[label_column_name]
    else:
        stratify_column = None

    sample_flag = True

    if len(df) <= sample_size :
        sample_flag = False
    
    if sample_flag :
        _, df_sample = train_test_split(
            df,
            test_size=sample_size,
            stratify = stratify_column,
            shuffle=True,      
            random_state=42
        )
    else:
        df_sample = df.copy()

    df_sample = df_sample.reset_index(drop=True)    
    
    with open(r"D:\multicompus\exam\nlp\stat_nlp\common\stopwords-ko.txt",
            encoding="utf-8") as f:
        stopwords = set(w.strip() for w in f if w.strip())

    if add_stopwords is not None:
        stopwords.update(add_stopwords)

    def tokenizer(text: str):        
        return preprocess_text(text, stopwords)   
    

    tfidf = TfidfVectorizer(
    max_df=0.8,
    min_df=1,
    token_pattern=None,
    tokenizer=tokenizer       
    )

    X_tfidf = tfidf.fit_transform(df_sample[TEXT_COL])
    return df_sample, X_tfidf, tfidf

