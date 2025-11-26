from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

docs = [
    "배송이 빠르고 포장이 깔끔해서 좋았어요",
    "배송이 느리고 박스가 찢어져 와서 별로였어요",
    "가격이 생각보다 저렴해서 가성비가 좋아요",
    "가격이 비싼 편인데 품질이 좋아서 만족해요",
    "품질이 안 좋아서 실망했어요",
    "배송은 빠르지만 품질이 너무 안 좋아요",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method='batch',
    random_state=42
)

#  학습 (문서-토픽 분포 θ_hat을 반환)
doc_topic = lda.fit_transform(X)
# print(doc_topic)
# print(doc_topic.sum(axis=1))
# print(lda.components_) # 각 토픽의 단어 분포, 크면 클수록 해당 토픽에서 그 단어가 등장할 가능성이 높다
