from sklearn.feature_extraction.text import CountVectorizer

docs = [
    '오늘 날씨 정말 좋다',
    '오늘 기분 정말 좋다',
    '오늘은 기분이 좋지 않다'
]

# unigram (단어 한개씩)
cv_uni = CountVectorizer(ngram_range=(1, 1))
X_uni = cv_uni.fit_transform(docs)
print("unigram vocab:", cv_uni.get_feature_names_out())
print(X_uni.toarray())
print()

