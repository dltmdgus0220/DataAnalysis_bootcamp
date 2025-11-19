import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [ "오늘 날씨가 좋아서 산책을 했다",
        "오늘은 비가 와서 우울하다",
        "점심에 맛있는 파스타를 먹었다",
        "저녁에 산책하면서 음악을 들었다"
        ]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())

# 쿼리(검색 문장)
query = ["산책을 하면서 날씨를 즐겼다"]
X_query = vectorizer.transform(query)
sim = cosine_similarity(X_query, X)  # (1, 문서수)
print("similarity:", sim)

most_sim_idx = np.argmax(sim[0])
print("Query:", query[0])
print("Most similar doc:", docs[most_sim_idx])