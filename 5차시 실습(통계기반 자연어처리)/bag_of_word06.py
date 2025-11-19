from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

okt = Okt()

def tokenize_korean(text):
    morphs = okt.pos(text, stem=True)
    tokens = [word for word, tag in morphs if tag in ["Noun", "Verb", "Adjective"]]
    return tokens


