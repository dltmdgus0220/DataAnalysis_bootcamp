from konlpy.tag import Okt

okt = Okt()

text = '자연어 처리는 정말 재밌네욬ㅋㅋㅋ'
tokens = okt.morphs(text, norm=True, stem=True) # 형태소 분석
print(tokens)
print(okt.pos(text, norm=True, stem=True, join=True)) # (형태소, 품사)


