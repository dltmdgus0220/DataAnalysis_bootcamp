from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


df = pd.read_csv('5차시 실습(통계기반 자연어처리)/movie_reviews.csv', encoding='utf8').dropna()
df_sample, _ = train_test_split(df, train_size=10000, stratify=df['label'], random_state=42)
X = df_sample['document']
y = df_sample['label']

x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

base_svm = LinearSVC()
calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)

svm_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", calibrated_svm)
])

svm_clf.fit(x_tr, y_tr)
y_pred_svm = svm_clf.predict(x_te)
print("== LinearSVC 분류 리포트 ==")
print(classification_report(y_te, y_pred_svm, digits=3))


# param_grid = {
#     'tfidf__ngram_range': [(1,1),(1,2)],
#     'tfidf__min_df': [1,2,3,4,5],
#     'svm__C': [0.1,0.5,1.0,10.0],
# }

# gs = GridSearchCV(
#     svm_clf,
#     param_grid=param_grid,
#     scoring="f1_macro", # 불균형 데이터 고려하면 macro-F1 추천 cv=3,
#     n_jobs=-1
# )

# gs.fit(x_tr, y_tr)
# print("Best params:", gs.best_params_)
# print("Best macro-F1:", gs.best_score_)
# best_model = gs.best_estimator_
# y_pred = best_model.predict(x_te)
# print("== 하이퍼파라미터 튜닝 LinearSVC 분류 리포트 ==")
# print(classification_report(y_te, y_pred, digits=3))

# tfidf = best_model.named_steps['tfidf']
# clf = best_model.named_steps['svm']
tfidf = svm_clf.named_steps['tfidf']
clf = svm_clf.named_steps['svm']
clf = clf.calibrated_classifiers_[0].estimator

feature_names = np.array(tfidf.get_feature_names_out())
coef = clf.coef_ # 값이크면:긍정, 값이작으면:부정, 절댓값이크면:영향력큼
print(coef)
top10_pos = coef[0].argsort()[-10:]
print("== 긍정에 강하게 기여하는 단어 ==")
print(feature_names[top10_pos])

top10_neg = coef[0].argsort()[:10]
print("== 부정에 강하게 기여하는 단어 ==")
print(feature_names[top10_neg])

probas = svm_clf.predict_proba([
    "지루하고 재미 없었다.",
    "배우의 연기가 영화를 살렸다."
])
print(probas)