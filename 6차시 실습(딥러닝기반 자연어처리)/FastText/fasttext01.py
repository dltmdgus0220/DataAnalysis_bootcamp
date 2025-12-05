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

print(ft_model.wv.vectors_ngrams.shape)

word = '지루하고'
idx = ft_model.wv.key_to_index[word]
print(f'word index : {idx}')

# ngram으로 만들어진 subword들의 인덱스 가져오기, <브랜, 브랜드, 랜드> 이런 것들의 인덱스, 해시로 저장되기 때문에 실제 단어는 확인불가
bucket_indices = ft_model.wv.buckets_word[idx] 
print(f'bucket indices for this word : {bucket_indices}')

subword_vectors = ft_model.wv.vectors_ngrams[bucket_indices] # 서브워드들의 임베딩된 벡터값
idx_to_word = ft_model.wv.index_to_key
print(len(idx_to_word)) # vocab 단어 수
print(f'num subword : {subword_vectors.shape[0]}') # 그냥 shape은 (서브워드수, 임베딩차원)
print(f'one subword vector (first 5) : {subword_vectors[0][:5]}')