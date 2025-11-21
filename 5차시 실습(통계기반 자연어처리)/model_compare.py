import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import seaborn as sns
import konlpy
from konlpy.tag import Okt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

okt = Okt()

df = pd.read_csv(r"D:\multicompus\example\nlp\stat_nlp\document_vector\movie_reviews.csv", encoding="utf-8")
df = df.dropna()

_, df_sample = train_test_split(
    df,
    test_size=10000,
    stratify=df["label"],
    shuffle=True,        # 기본값이라 사실 안 써도 됨
    # random_state=42
)

X = df_sample['document']
y = df_sample['label']

print("데이터 크기:" , df_sample.shape)
print("레이블 분포:", df_sample["label"].value_counts())
print("결측값 개수:", df_sample.isna().sum())

with open(r"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\stopwords-ko.txt", encoding="utf-8") as f:
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

print("학습데이터 개수 :", len(x_train))
print("테스트데이터 개수:", len(x_test))

# def train_and_evaluate(model_name, classifier):
#     print("======================")
#     print(model_name)

#     pipe = Pipeline(steps=[
#         ("tfidf", TfidfVectorizer(
#             tokenizer=preprocess_text,
#             token_pattern=None, 
#             ngram_range=(1,2)
#         )),
#         ("clf",classifier ),

#     ])

#     pipe.fit(x_train, y_train)

#     y_pred = pipe.predict(x_test)
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, digits=3))

#     macro_f1 = f1_score(y_test, y_pred, average="macro")
#     print(f"macro-F1 : {macro_f1:.3f}")

#     return pipe, macro_f1 
# =====================================================================

def train_and_evaluate(model_name, classifier, param_grid=None):
    print("======================")
    print(model_name)

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            tokenizer=preprocess_text,
            token_pattern=None, 
            ngram_range=(1,2)
        )),
        ("clf",classifier ),

    ])

    gs = GridSearchCV(
        pipe, 
        param_grid=param_grid,
        cv = 5,
        scoring="f1_macro",
        refit=True
    )

    gs.fit(x_train, y_train)

    print("Best Params : ", gs.best_params_)
    print("Best macro F1 : ", gs.best_score_)

    best_meodel = gs.best_estimator_

    y_pred = best_meodel.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"macro-F1 : {macro_f1:.3f}")

    return best_meodel, macro_f1 

# =====================================================================
param_grid = {
    "clf__C" : [0.01, 0.1, 1.0, 10.0]
}
log_reg_clf = LogisticRegression(max_iter=1000)
log_reg_pipe, log_reg_f1 = train_and_evaluate("LogisticRedgression", log_reg_clf, param_grid)

param_grid ={
    "tfidf__ngram_range":[(1,1), (1,2)],
    "clf__alpha" : [0.1, 0.5, 1.0, 2.0]
}

nb_clf = MultinomialNB(alpha=1.0)
nb_pipe, nb_f1 = train_and_evaluate("MultinomalNB", nb_clf, param_grid)


param_grid = {
    "clf__C" : [0.01, 0.1, 1.0, 10.0]
}

svm_clf = LinearSVC(C=1.0)
svm_pipe, svm_f1 = train_and_evaluate("LinearSVC", svm_clf, param_grid)


result = pd.DataFrame({
    "model":["LogisticRegression", "MultinomialNB", "LinearSVC"],
    "macro-F1" : [log_reg_f1, nb_f1, svm_f1]
})

print("macro F1 : \n", result)

tfidf_nb = nb_pipe.named_steps["tfidf"]
nb = nb_pipe.named_steps["clf"]

feature_names = np.array(tfidf_nb.get_feature_names_out())

for i, class_label in enumerate(nb.classes_):
    log_prob = nb.feature_log_prob_[i]
    top10_idx = log_prob.argsort()[-10:]
    print(f"===클래스 {class_label} 대표 단어(상위 10개)====")
    print(feature_names[top10_idx])


tfidf_svm = svm_pipe.named_steps["tfidf"]
svm = svm_pipe.named_steps["clf"]


feature_names = np.array(tfidf_svm.get_feature_names_out())

coef = svm.coef_[0]
top10_pos = coef.argsort()[-10:]
top10_neg = coef.argsort()[:10]

print("=== 긍정 쪽으로 크게 기여하는 단어 TOP 10 ===")
print(feature_names[top10_pos])

print("\n=== 부정 쪽으로 크게 기여하는 단어 TOP 10 ===")
print(feature_names[top10_neg])


