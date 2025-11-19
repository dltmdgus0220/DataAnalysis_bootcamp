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


texts = [
    "이 영화 정말 재밌어요 최고에요",
    "별로 재미없고 지루했어요",
    "배우 연기도 좋고 감동적이었어요",
    "스토리가 엉망이고 시간 아까웠어요",
    "완전 최고 강추합니다",
    "돈 아깝고 다시는 보고 싶지 않아요"
]

labels = [1,0,1,0,1,0]

# train/test 분할s
x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.3, random_state=42)
