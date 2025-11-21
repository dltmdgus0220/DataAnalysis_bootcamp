from sklearn.feature_extraction.text import CountVectorizer

docs = [
    '오늘 날씨 정말 좋다',
    '오늘 기분 정말 좋다',
    '오늘은 기분이 좋지 않다'
]

vectorizer = CountVectorizer() # 공백기준 토큰화, 소문자변환(영문기준)

X = vectorizer.fit_transform(docs) # fit:단어집 만들기, transform:희소행렬 만들기
# 희소행렬:(단어인덱스, 값(단어수))
# [0, 0, 3, 0, 0, 1, 0]
# (2,3), (5,1)
# 불필요한 0이 많으므로 메모리 사용을 줄이기 위함.

print(type(X), X.shape) # shape:(문서수, 단어수)
print(vectorizer.get_feature_names_out()) # 단어집
print(X.toarray()) # 밀집행렬로 보기