from gensim.models import Word2Vec
import torch
import torch.nn as nn

sentences = [["이", "영화", "정말", "최고", "였다"],
    ["배우", "연기", "가", "최고", "이다"],
    ["스토리", "가", "지루하다"],
    ["내용", "이", "지루하고", "별로", "였다"],
    ["이", "브랜드", "디자인", "이", "세련되다"],
    ["이", "브랜드", "가격", "은", "비싸다"]
]

model = Word2Vec(
    sentences=sentences,
    vector_size=50, # 임베딩 차원
    window=3, # 문맥 범위
    min_count=1, # 최소 등장 횟수
    sg=1, # 1=Skip-gram, 0=CBOW
    workers=1, # 실습 환경에서는 1로 해도 충분
    epochs=100 # 에폭수를 늘려서 더 수렴시키기
)

# 단어 벡터 확인
print("단어 '최고' 벡터 크기:", model.wv["최고"].shape)
print("벡터 앞 5개 값:", model.wv["최고"][:5])

# 비슷한 단어 찾기
print("\n[단어 '최고'와 비슷한 단어 Top-5]")
for w, score in model.wv.most_similar("최고", topn=5):
    print(f"{w:10s} 유사도: {score:.3f}")

# 유사도 예시
print("\n[단어 간 유사도]")
print("최고 vs 별로 :", model.wv.similarity("최고", "별로"))
print("지루하다 vs 별로 :", model.wv.similarity("지루하다", "별로"))

word_index = model.wv.key_to_index
index_word = model.wv.index_to_key

vocab_size = len(word_index)
embed_dim = model.vector_size

print("vocab_size : ", vocab_size)
print("embed_dim : ", embed_dim)

# 단어 벡터들 (shape: (vocab_size, embed_dim))
pretrained_weights = model.wv.vectors # numpy array
print("pretrained_weights.shape:", pretrained_weights.shape)

