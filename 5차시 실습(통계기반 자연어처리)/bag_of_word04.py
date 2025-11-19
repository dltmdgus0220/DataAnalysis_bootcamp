from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

docs = [
    '오늘 날씨 정말 좋다',
    '오늘 기분 정말 좋다',
    '오늘은 기분이 좋지 않다'
]

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(docs)
print("vocab:", tfidf.get_feature_names_out())
print("shape:", X_tfidf.shape)
print(X_tfidf.toarray().round(3)) # 단어별 중요도 출력
# tfidf가 높으려면 현 문장에서 자주 나오면서 동시에 다른 문장에서는 많이 안나와야 됨.