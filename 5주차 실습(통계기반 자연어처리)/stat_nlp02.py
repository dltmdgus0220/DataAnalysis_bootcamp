from konlpy.tag import Okt

okt = Okt()

with open('E:/데이터분석가 부트캠프/실습/DataAnalysis_bootcamp/5주차 실습(통계기반 자연어처리)/stopwords-ko.txt', encoding='utf-8') as f:
    basic_stopwords = set(w.strip() for w in f if w.strip()) # strip: 양쪽 끝 공백 혹은 파라미터로 받은 문자 제거

