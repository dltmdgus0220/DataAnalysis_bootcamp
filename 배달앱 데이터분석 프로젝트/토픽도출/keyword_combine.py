import json
import pandas as pd
from kiwipiepy import Kiwi
from konlpy.tag import Okt

STATE = ["느림", "빠름", "나쁨", "비쌈", "쌈", "불친절", "친절", "안됨", "있음", "없음", "적음", "많음"]
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

df.loc[0, 'content'] = '배달이 느리고 혜택도 없어서 안좋다.' # 예제
data = df['content'].apply(lambda x : x.split()) # 공백기준토큰화
print("===== 공백기준 토큰화 =====")
print(data[0]) # 토큰화 결과

data = data.apply(lambda tokens: [CANON.get(token, token) for token in tokens]) # 1차 매핑
data = data.apply(lambda x : " ".join(x))
print("\n===== 1차 매핑 =====")
print(data[0]) # 1차 매핑 결과

data = data.apply(lambda x : [w for w, p in okt.pos(x) if p in ("Noun", "Adjective", "Verb")]) # okt 토큰화
print("\n===== Okt 토큰화 =====")
print(data[0]) # 토큰화 결과

data = data.apply(lambda tokens: [CANON.get(token, token) for token in tokens]) # 2차 매핑
print("\n===== 2차 매핑 =====")
print(data[0]) # 2차 매핑 결과

