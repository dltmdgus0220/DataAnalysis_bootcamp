from konlpy.tag import Okt

okt = Okt()

with open('E:/데이터분석가 부트캠프/실습/DataAnalysis_bootcamp/5주차 실습(통계기반 자연어처리)/stopwords-ko.txt', encoding='utf-8') as f:
    for w in f:
        print(w.strip())
    basic_stopwords = set(w.strip() for w in f if w.strip()) # strip: 양쪽 끝 공백 혹은 파라미터로 받은 문자 제거

def korean_tokenize_okt(
        text: str,
        tagger=okt,
        stopwords: set | None=None, # set으로 받거나 아무것도 안받으면 None으로 받을거다
        remove_pos=('Josa', 'Eomi', 'Punctuation', 'Suffix'),
        min_len: int = 1,
):
    if stopwords is None:
        stopwords = basic_stopwords

    tokens = []
    for word, pos in tagger.pos(text, norm=True, stem=True):
        if pos in remove_pos: # 품사확인
            continue
        if word in stopwords: # 단어확인
            continue
        if len(word) < min_len: # 길이확인
            continue
        tokens.append(word)

    return tokens

if __name__ == '__main__':
    sentence = '오늘은 날씨가 너무 좋습니다. 그래서 기분이 더 좋습니다.'
    print('원문:', sentence)
    print('Okt 품사:', okt.pos(sentence, norm=True, stem=True))
    print('전처리 결과:', korean_tokenize_okt(sentence))