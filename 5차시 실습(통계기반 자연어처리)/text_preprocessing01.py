from konlpy.tag import Okt

okt = Okt()

text = '자연어 처리는 정말 재밌네욬ㅋㅋㅋ'
tokens = okt.morphs(text, norm=True, stem=True) # 형태소 분석
print(tokens)
print(okt.pos(text, norm=True, stem=True, join=True)) # (형태소, 품사)


import re

text = "오늘은 2025-11-16, 가격은 30,000원입니다!!!"
# 1) 모두 소문자 (영문만 있는 경우)
text_lower = text.lower()
# 2) 숫자를 특수 토큰으로 치환 (예: "NUM")
text_num = re.sub(r"\d+", "NUM", text_lower)
# 3) 특수문자 제거 (한글/영문/숫자/공백만 남기기)
text_clean = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text_num)
print(text_clean)


text = '문의: admin@example.com 또는 help@my-service.com.kr로 보내주세요.'
pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(pattern, text)
print(emails)


text = '연락처는 010-1234-5678입니다. 예전 번호는 011-222-3333은 사용 안해요'
pattern = r'01[016789]-\d{3,4}-\d{4}'
phones = re.findall(pattern, text)
print(phones)


text = 'Python으로 한국어 NLP를 배우는 중입니다. 자연어처리 재밌다.'
pattern = r'[가-힣]+'
korean_word = re.findall(pattern, text)
print(korean_word)


text = '오늘도 #자연어처리 #Python #딥러닝 공부중!!'
pattern = r'#([0-9a-zA-Z가-힣_]+)'
tags = re.findall(pattern, text)
print(tags)


text = '<p>안녕하세요, <b>엘리스</b>님!!</p>'
pattern = r'<.*?>' # 태그들 찾음, <span> 이런 것도 가능
clean = re.sub(pattern, '', text)
print(clean)


text = 'Error: code=404, message=Not found'
m = re.search(r'code=(\d+)', text)
print(m)
print(m.group(0)) # 매칭된거 전부다
print(m.group(1)) # 캡처한 거 중 1번째


m = re.search(r'\d+', 'abc123def')
print(m.group())
print(m.start())
print(m.end())
print(m.span())


m = re.search(r'(\d+)-(\d+)-(\d+)', 'tel:010-1234-5678')
print(m.group())
print(m.group(0))
print(m.group(1))
print(m.group(2))


# 불용어 처리
stopwords = set([
"은", "는", "이", "가", "을", "를", "에", "에서", "으로", "그리고", "그러나", "하지만", "또는",
"도", "만", "와", "과", "저", "그", "이", "것", "수", "들", "등"
])

sentence = "하지만 이런 불용어는 분석에 큰 도움이 되지 않습니다."

# 1단계: 형태소 분석
tokens = okt.morphs(sentence)
print("토큰:", tokens)

# 2단계: 불용어 제거
tokens_clean = [t for t in tokens if t not in stopwords]
print("불용어 제거 후:", tokens_clean)
