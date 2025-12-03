from gensim.models import FastText

sentences = [
    ["이", "영화", "정말", "최고", "였다"],
    ["배우", "연기", "가", "최고", "이다"],
    ["스토리", "가", "지루하다"],
    ["내용", "이", "지루하고", "별로", "였다"],
    ["이", "브랜드", "디자인", "이", "세련되다"],
    ["이", "브랜드", "가격", "은", "비싸다"],
]

# FastText 모델 학습
ft_model = FastText(
    sentences,
    vector_size=50, window=3,
    min_count=1,
    sg=1,        # Skip-gram
    workers=1,
    epochs=100
)


# 기존 단어 벡터
print("단어 '브랜드' 벡터 크기:", ft_model.wv["브랜드"].shape)
oov_word = "브랜드맛집"   # 말뭉치에 없는 단어
oov_vec = ft_model.wv[oov_word]
print("OOV 단어 '브랜드맛집' 벡터 크기:", oov_vec.shape)
print("브랜드 vs 브랜드맛집 유사도:", ft_model.wv.similarity("브랜드", "브랜드맛집"))

