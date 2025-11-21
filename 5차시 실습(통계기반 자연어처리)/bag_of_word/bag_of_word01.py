import numpy as np

# bag_of_word : 문장에서 단어의 순서나 문맥은 무시하고, 단순히 카운팅하여 수치로 표현
# 영화 진짜 재밌다. / 진짜 영화 재밌다. => 같은 문장으로 봄.

docs = [
    '오늘 날씨 정말 좋다',
    '오늘 기분 정말 좋다',
    '오늘은 기분이 좋지 않다'
]

tokenized_docs = [doc.split() for doc in docs]
print(tokenized_docs)

# 단어집 만들기
vocab = sorted({word for doc in tokenized_docs for word in doc}) # {}:집합, sorted(set):리스트
print(vocab)
# print(type(vocab))

# 단어: 인덱스 매핑
word_to_idx = {word: i for i, word in enumerate(vocab)}
print(word_to_idx)

bow_vectors = []

for doc in tokenized_docs:
    vec = np.zeros(len(vocab), dtype=int)

    for w in doc:
        idx = word_to_idx[w]
        vec[idx] += 1 # 카운팅

    bow_vectors.append(vec) # 각 문장별 벡터 저장

bow_matrix = np.vstack(bow_vectors) # 행방향으로 쌓기, 즉 위에서 아래로 행렬 추가
print(bow_matrix)