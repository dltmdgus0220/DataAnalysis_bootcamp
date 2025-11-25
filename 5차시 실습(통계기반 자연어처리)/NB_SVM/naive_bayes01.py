from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

texts = [
    "서비스가 너무 느리고 불친절했어요",   # 0
    "맛이 없고 양도 너무 적어요",         # 0
    "직원들이 정말 친절하고 음식도 맛있어요", # 1
    "가격도 적당하고 분위기가 좋아요",     # 1
    "다시는 오고 싶지 않아요",             # 0
    "완전 만족스러운 식사였습니다",         # 1,
]
labels = [0, 0, 1, 1, 0, 1]


x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)

nb_clf = Pipeline(steps=[
    ('vect', CountVectorizer()),
    ('nb', MultinomialNB(alpha=1.0))
])

nb_clf.fit(x_tr, y_tr) # CountVectorizer().fit_transform(x_tr), MultinomialNB().fit(x_tr_vectorized, y_tr) 수행
pred = nb_clf.predict(x_te) # CountVectorizer().transform(x_te), MultinomialNB().predict(x_te_vectorized) 수행
# 파이프라인이기 때문에 알아서 자동으로 다 수행

print(classification_report(y_te, pred, digits=3))

probs = nb_clf.predict_proba([
    '배송이 빠르고 친절해서 만족스럽다.',
    '제품이 고장나서 너무 화가 난다.'
])

print(probs)


vect = nb_clf.named_steps['vect']
nb = nb_clf.named_steps['nb']

feature_names = np.array(vect.get_feature_names_out())
print(feature_names)
# print(nb.classes_)
# # print(nb.class_log_prior_)
# print(np.exp(nb.class_log_prior_)) # 데이터의 클래스별 비율
# print(np.exp(nb.feature_log_prob_)) # 각 클래스에서 각 단어가 등장할 조건부확률, (0,0) : 0번 클래스에서 0번째 단어가 등장할 조건부확률

for i, c in enumerate(nb.classes_):
    print(f'===클래스 {c} 대표 단어===')
    log_prob = nb.feature_log_prob_[i]
    top10_idx = log_prob.argsort()[-10:] # 상위 10등
    print(feature_names[top10_idx])
