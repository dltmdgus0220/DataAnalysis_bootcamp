import numpy as np
import pandas as pd
import re
import os
import json

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

okt = Okt()

rows = [] # 아예 딕셔너리 형태로 넣고 나중에 데이터프레임으로 바꿔버리기
i = 1
class_count = {} # 클래스별 데이터 수를 저장할 딕셔너리
while True: # 클래스별 데이터 수가 500개가 될때까지 반복
    if len(class_count) > 0 and all(v >= 600 for v in class_count.values()):
        print("모든 클래스 목표 개수 도달")
        break
    
    path = os.path.join('5차시 실습(통계기반 자연어처리)/1-1. 여성의류', f'1-1.여성의류({i}).json')

    if not os.path.isfile(path):
        print(f"더 이상 읽을 파일 없음.")
        break

    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f) # 100개씩 가지고 있음
            df_tmp = pd.DataFrame(data)[['RawText','GeneralPolarity']].dropna()
            for _, row in df_tmp.iterrows(): # 행으로 읽어옴 ,(인덱스, ['RawText','GeneralPolarity']:series) 형태의 튜플. 따라서 뒤에꺼만 필요함
                c = row['GeneralPolarity']
                count = class_count.get(c,0)
                if count < 500: # 500개가 되지 않으면
                    rows.append({
                        'RawText':row['RawText'],
                        'GeneralPolarity':row['GeneralPolarity']
                    })
                    class_count[c] = count + 1 # 없으면 생성, 있으면 업데이트

print(len(rows))
df = pd.DataFrame(rows)

X = df["RawText"]
y = df["GeneralPolarity"].astype(int)

print("\n레이블 분포(클래스별 샘플 수)")
print(y.value_counts())

with open("5차시 실습(통계기반 자연어처리)/stopwords-ko.txt", encoding="utf-8") as f:
    stopwords = set(w.strip() for w in f if w.strip())

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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n학습데이터 개수 :", len(x_train))
print("테스트데이터 개수:", len(x_test))

def train_and_evaluate(model_name, vectorizer, param_grid=None):
    print("======================")
    print(model_name)

    pipe = Pipeline(steps=[
        ("vect", vectorizer),
        ("clf", MultinomialNB(alpha=1.0)), # class_prior=[1/3,1/3,1/3], fit_prior=False
    ])

    gs = GridSearchCV(
        pipe, 
        param_grid=param_grid,
        cv = 5,
        scoring="f1_macro",
        refit=True
    )

    # sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    gs.fit(x_train, y_train) #, clf__sample_weight = sample_w)

    print("Best Params : ", gs.best_params_)
    print("Best macro F1 : ", gs.best_score_)

    best_meodel = gs.best_estimator_

    y_pred = best_meodel.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"macro-F1 : {macro_f1:.3f}")

    return best_meodel, macro_f1 

param_grid ={
    "vect__ngram_range":[(1,1), (1,2)],   
    # "vect__max_features" : [1000,2000, 3000], 
    # "vect__min_df" : [1,2,3],
    # "clf__alpha" : [0.01, 0.1, 1.0, 10]
}

