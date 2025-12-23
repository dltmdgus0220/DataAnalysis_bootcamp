import json
import pandas as pd
import numpy as np
from collections import Counter
from kiwipiepy import Kiwi
from konlpy.tag import Okt

STATE = ["느림", "빠름", "나쁨", "좋음", "비쌈", "쌈", "불친절", "친절", "안됨", "있음", "없음", "적음", "많음", "편함", "편리함"] # 상태
ASPECT = {"배달", "쿠폰", "주문", "할인", "어플", "사용", "가게", "결제", "이용", "센터", "지역",
          "음식", "리뷰", "이벤트", "취소", "수수료", "혜택", "매장", "전화", "오류", "가격", "업체", "상품권",
          "메뉴", "광고", "접속", "카드", "소비자", "연결", "기능", "서비스", "기사", "업데이트", "문의", "지연", "설정",
          "배차", "배송", "라이더", "삭제", "업데이트", "환불", "시스템", "도착", "위치", "매장", "보상", "음식점", "양",
          "배민보다", "쿠팡이츠보다", "쿠팡보다", "요기요보다", "땡겨요보다",
          "알뜰배달", "한집배달", "배달특급", "배민클럽", "배민", "쿠팡이츠", "쿠팡", "요기요", "땡겨요", "요기패스", "와우회원", "지역화폐", "온누리상품권", "고객센터"} # 대상
SPECIAL_PATTERNS = [
    (r"알뜰\s*배달", "알뜰배달"),
    (r"한\s*집\s*배달", "한집배달"),
    (r"배달\s*특급", "배달특급"),
    (r"배민\s*클럽", "배민클럽"),
    (r"배달의\s*민족", "배민"),
    (r"쿠팡\s*이\s*츠", "쿠팡이츠"),
    (r"요기\s*요", "요기요"),
    (r"땡겨\s*요", "땡겨요"),
    (r"요기\s*패스", "요기패스"),
    (r"와우\s*회원", "와우회원"),
    (r"지역\s*화폐", "지역화폐"),
    (r"온누리\s*상품권", "온누리상품권"),
    (r"고객\s*센터", "고객센터"),
    (r"써비\s*스", "서비스"),
    (r"써비\s*쓰", "서비스"),
    (r"서비\s*쓰", "서비스"),
    (r"배민\s*보다", "배민보다"),
    (r"쿠팡\s*보다", "쿠팡보다"),
    (r"쿠팡이츠\s*보다", "쿠팡이츠보다"),
    (r"요기요\s*보다", "요기요보다"),
    (r"땡겨요\s*보다", "땡겨요보다"),
    (r'굿+', "좋음"),
    (r'안\s*좋(?:다|아|아요|네요|음)?', "나쁨")
]

# --- 0. 토크나이저 테스트 ---
kiwi = Kiwi()
okt = Okt()
# for s in STATE:
#     print(f"====={s}=====")
#     for t in kiwi.tokenize(s):
#         print(t.form, t.tag) # 없음-> 없/음, 느림-> 느리/ㅁ 이렇게 분리하기 때문에 맞지않음
#     for w, p in okt.pos(s):
#         print(w, p) 


# --- 1. 불용어 및 매핑사전 ---

# 불용어
add_stopwords = ['땡기', '요기']
with open('stopwords-ko.txt', encoding='utf-8') as f:
    STOPWORDS = set(w.strip() for w in f if w.strip())
if add_stopwords is not None:
    STOPWORDS.update(add_stopwords)

# 매핑
with open("canon.json", "r", encoding="utf-8") as f:
    CANON = json.load(f)


# --- 2. 데이터로드 ---

df = pd.read_csv(r'배달앱 데이터분석 프로젝트\데이터\전처리데이터\preprocessed_ddangyo_reviews_playstore.csv', encoding='utf-8-sig')


# --- 3. 토큰화 및 매핑 ---

def tokenize_and_mapping(text:str) -> str:
    text = text.split() # 공백기준토큰화
    text = [CANON.get(token, token) for token in text] # 1차 매핑
    # okt 토큰화를 위한 토큰결합
    text = " ".join(text)
    text = [w for w, p in okt.pos(text) if p in ("Noun", "Adjective", "Verb") and p not in STOPWORDS] # okt 토큰화
    text = [CANON.get(token, token) for token in text] # 2차 매핑

    return text

print(tokenize_and_mapping('배달이 느리고 혜택도 없어서 안좋다.'))

# --- 4. 키워드결합 ---

def keyword_combine(keywords:list) -> list:
    # ['배달', '느림', '혜택', '없음', '나쁨']
    # ['배달느림', '혜택없음', '나쁨']
    ret = []
    used = [False] * len(keywords)
    for i in range(len(keywords) - 1):
        a, b = keywords[i], keywords[i+1]

        # 대상+상태
        if (b in STATE) and (a not in STATE):
            ret.append(a + b) # "배달"+"느림" -> "배달느림"
            used[i] = True
            used[i+1] = True
    
    # 남은 키워드 처리
    for i, keyword in enumerate(keywords):
        if used[i]:
            continue
        ret.append(keyword) # 남은 키워드 다 추가

    return ret
    # 키워드 빈도수 파악을 위해 중복제거는 하지 않음

data = data.apply(lambda x : keyword_combine(x))
print("\n===== 키워드결합 =====")
print(data[0]) # 키워드결합

