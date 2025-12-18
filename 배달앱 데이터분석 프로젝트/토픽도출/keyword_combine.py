import json
import pandas as pd
from kiwipiepy import Kiwi
from konlpy.tag import Okt

STATE = ["느림", "빠름", "나쁨", "비쌈", "쌈", "불친절", "친절", "안됨", "있음", "없음", "적음", "많음"]
kiwi = Kiwi()
okt = Okt()
for s in STATE:
    print(f"====={s}=====")
    for t in kiwi.tokenize(s):
        print(t.form, t.tag) # 없음-> 없/음, 느림-> 느리/ㅁ 이렇게 분리하기 때문에 맞지않음
    for w, p in okt.pos(s):
        print(w, p) 

