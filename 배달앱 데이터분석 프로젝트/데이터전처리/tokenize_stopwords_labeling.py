import pandas as pd
from konlpy.tag import Okt
from collections import Counter


# --- 0. 초기 설정 및 파일 경로 ---
okt = Okt()
FILE_PATH_INPUT = r"배달앱 데이터분석 프로젝트\데이터\원시데이터\ddangyo_reviews_playstore_7000.csv"
FILE_PATH_OUTPUT = r"배달앱 데이터분석 프로젝트\데이터\전처리데이터\preprocessed_ddangyo_reviews_playstore.csv"

# --- 1. 데이터 로드 및 전처리 ---
try:
    df = pd.read_csv(FILE_PATH_INPUT)
    print(f"--- 1. 파일 로드 완료: {FILE_PATH_INPUT} (총 {len(df)}개 리뷰) ---")
except FileNotFoundError:
    print(f"오류: '{FILE_PATH_INPUT}' 파일을 찾을 수 없습니다.")
    exit()

# 2. 토큰화 및 품사 태깅 함수 정의 (norm=True, stem=True 사용)
def simple_tokenize_and_tag(text):
    if pd.isna(text) or text.strip() == "":
        return []
    return okt.pos(str(text), norm=True, stem=True)

# 3. '쿠팡이츠' 사후 결합 함수 정의 (data_preprocessing.py 기반) #브랜드명 고유 명사 처리
def coupang_eats_post_processing(tagged):
    result = []
    i = 0
    while i < len(tagged):
        if i + 2 < len(tagged):
            w1, p1 = tagged[i]
            w2, p2 = tagged[i+1]
            w3, p3 = tagged[i+2]
            
            # '쿠팡' '이' '츠'가 연속될 경우 '쿠팡이츠'로 결합
            if w1 == '쿠팡' and w2 == '이' and w3 == '츠':
                result.append(('쿠팡이츠', 'Noun'))
                i += 3
                continue

        result.append(tagged[i])
        i += 1
    return result

# 4. 토큰화, 사후 결합 실행
df['tagged_content'] = df['content'].apply(simple_tokenize_and_tag)
df['tagged_content'] = df['tagged_content'].apply(coupang_eats_post_processing)
print("--- 2. 토큰화 및 '쿠팡이츠' 사후 결합 완료 ---")


# --- 5. 불용어 처리 로직 (stopword_processing.py 기반) ---

# 불용어 사전 정의
# 특정 품사 제거 목록 (요청하신 목록에 해당하지 않는 품사)
REMOVE_POS = ("Josa", "Eomi", "Punctuation", "Suffix", "Adverb", "Conjunction", "URL", "Hashtag", "Foreign", "Alpha", "Number", "KoreanParticle")
# 특정 단어 제거 목록 (요청하신 목록)
CUSTOM_STOPWORDS = ['되다', '이다', '하다', '것', '있다',"돼다",'요','자다','좋다','배민','쿠팡','와우','쿠팡이츠','어플','앱',"최고","한국","회원",'굿굿','굿굿굿','짱짱','사용','탈퇴'] 


# 불용어 처리 함수 정의
def stopword_processing_final(tagged_list, remove_pos=REMOVE_POS, stopword=CUSTOM_STOPWORDS):
    final_keywords = []
    
    for word, pos in tagged_list:
        
        # A. 품사 필터링 (명사, 동사, 형용사 외 제거)
        if pos in remove_pos:
            continue
   
        if len(word) <= 1:
            continue
            
        # C. 사용자 정의 단어 불용어 필터링
        if word in stopword:
            continue
            
        # D. 최종 키워드 추가 ('쿠팡이츠'는 이미 명사로 통과됨)
        final_keywords.append(word)
    
    final_keywords = list(set(final_keywords))
    return final_keywords

# --- 6. 불용어 처리 함수 적용 ---
df['filtered_keywords'] = df['tagged_content'].apply(stopword_processing_final)
print("--- 3. 최종 불용어 처리 완료 (filtered_keywords 컬럼 생성) ---")

# --- 7. 별점 기반 라벨링 ---

# 긍정(1) : 별점 4,5점 / 중립(0) : 별점 3점 / 부정(-1) : 별점 1,2점
df.loc[df['score'].isin([4, 5]), 'sentiment_label'] = 1 # 긍정
df.loc[df['score'] == 3, 'sentiment_label'] = 0 # 중립
df.loc[df['score'].isin([1, 2]), 'sentiment_label'] = -1 # 부정

df.loc[df['score'].isin([4, 5]), 'sentiment'] = 'positive' # 긍정
df.loc[df['score'] == 3, 'sentiment'] = 'neutral' # 중립
df.loc[df['score'].isin([1, 2]), 'sentiment'] = 'negative' # 부정

# --- 8. 최종 결과 저장 ---

# 저장할 데이터프레임 구성: 원본 content와 최종 키워드 목록
df_output = df.copy()
df_output = df_output.drop('tagged_content', axis=1)

original_rows = len(df_output)
df_output = df_output[df_output['filtered_keywords'].apply(len) > 0]
deleted_rows = original_rows - len(df_output)

# CSV 저장 시 리스트 형태가 문자열로 저장
df_output.to_csv(FILE_PATH_OUTPUT, index=False, encoding="utf-8-sig")

print(f"\n--- 4. 최종 파일 저장 완료! ({FILE_PATH_OUTPUT}) ---")
print(f"키워드가 0개인 리뷰 {deleted_rows}개를 삭제했습니다. (최종 {len(df_output)}개 리뷰 남음)")

# 최종 결과 확인 (옵션)
print("\n--- 저장된 파일 미리보기 ---")
print(df_output.head())