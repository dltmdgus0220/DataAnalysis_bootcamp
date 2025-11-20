from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

texts = [
    "서비스가 너무 느리고 불친절했어요",   # 0
    "맛이 없고 양도 너무 적어요",         # 0
    "직원들이 정말 친절하고 음식도 맛있어요", # 1
    "가격도 적당하고 분위기가 좋아요",     # 1
    "다시는 오고 싶지 않아요",             # 0
    "완전 만족스러운 식사였습니다",         # 1,
]
labels = [0, 0, 1, 1, 0, 1]