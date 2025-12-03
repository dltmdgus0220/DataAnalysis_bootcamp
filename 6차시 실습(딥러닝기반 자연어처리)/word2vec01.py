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

