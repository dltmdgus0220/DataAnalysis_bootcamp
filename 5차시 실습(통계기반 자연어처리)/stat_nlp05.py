from konlpy.tag import Okt
import re
import matplotlib.pyplot as plt

# Okt 형태소 분석기 객체 생성
okt = Okt()

reviews = ["진짜 겁나 신박함ㅋㅋㅋ첨엔 적응안됐는데 배우들 연기도 좋고 넘나 신선해서 시간가는줄 모르고 봄.그냥 신난다",
"와 속도감 최고임. 색다른 영화는 분명한것 같다..! 지창욱 연기 앞으로도 왕기대!!!!",
"와우~~개봉첫날~조조로보고 왔습니다 기대이상으로 재미난영화 입니다~~대박 강추~~스피드한 액션~^^ 지창욱 연기넘멋졌네요~다들배우들 연기도 굿이었답니다"]

stopwords = [
    "이", "그", "저", "것", "거", "수",
    "그리고", "그런데", "하지만", "또는",
    "하면", "하고", "같은", "좀", "조금", "너무"
]

# print(" aa     bb    cc ".split(" "))

# for text in reviews:
#     clean = re.sub(r"[^가-힣0-9\s]", " ", text)
#     clean = re.sub(r"\s+", " ", clean).strip()
#     tokens = clean.split(" ")
#     tokens_no_stop = [ w for w in tokens if w not in stopwords]
#     print(tokens_no_stop)


# for text in reviews:
#     morphs_text = okt.pos(text)
#     print("\n기본 POS:\n", morphs_text)
#     morphs_norm = okt.pos(text, norm=True)
#     print("\n정규화 POS:\n", morphs_norm)
#     morphs_stem = okt.pos(text, norm=True, stem=True)
#     print("\n정규화, 어간/표제어 추출 POS:\n",morphs_stem)


def preprocess_korean(text:str):
    clean = re.sub(r"[^가-힣0-9\s]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    morphs_text = okt.pos(clean, norm=True, stem=True)

    content_words = []
    for word, pos in morphs_text:
        if pos in ["Noun", "Verb", "Adjective"]:
            if word not in stopwords:
                content_words.append(word)

    return {
        "clean_text" : clean,
        "morphs_text" : morphs_text,
        "content_words": content_words
    }   

 

for text in reviews:
    result = preprocess_korean(text)
    for k, v in result.items():
        print(f"{k} : {v}")