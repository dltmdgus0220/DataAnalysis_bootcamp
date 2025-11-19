from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

okt = Okt()

def tokenize_korean(text):
    morphs = okt.pos(text, stem=True)
    tokens = [word for word, tag in morphs if tag in ["Noun", "Verb", "Adjective"]]
    return tokens


docs = [
    "이 영화 정말 재밌어요 최고에요",
    "별로 재미없고 지루했어요",
    "배우 연기도 좋고 감동적이었어요",
    "스토리가 엉망이고 시간 아까웠어요",
    "완전 최고 강추합니다",
    "돈 아깝고 다시는 보고 싶지 않아요"
]

vectorizer = TfidfVectorizer(
    tokenizer=tokenize_korean,
    token_pattern=None, # 기본 정규식 비활성화
    min_df=1, # 1이면 의미없음
    ngram_range=(1,1) # unigram
)
# 기본정규식:r"(?u)\b\w\w+\b"
# (?u):유니코드 모드, \w의 인식범위를 확장(영어+숫자+몇몇 유니코드)
# \b:단어경계(공백,문장부호,문자열 시작과 끝을 기준)
# \w\w+:문자한개+문자이상, 즉 문자 두개이상 받음

