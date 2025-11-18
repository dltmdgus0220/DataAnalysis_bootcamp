from konlpy.tag import Okt
import re

okt = Okt()

reviews = ['거듭할수록 트릭이 드러난다는 핸디캡을 안고',
           '시건방진 초짜 야바위꾼들',
           '호스맨 기다렸다고ㅠㅠ 꼭 보세요.. 시리즈 팬들도 충분히 만족할만함',
           ' 1 2 극장에서 진짜 재밌게봤어서 맨날 다시보는 영화인데요, 3편도 기대한만큼 정말 재밌었어요!!! 꼭 봐야합니다!']

with open('E:/데이터분석가 부트캠프/실습/DataAnalysis_bootcamp/5차시 실습(통계기반 자연어처리)/stopwords-ko.txt', encoding='utf-8') as f:
    stopwords = set(w.strip() for w in f if w.strip())

